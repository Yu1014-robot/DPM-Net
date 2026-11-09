import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out 
from timm.models.layers import to_2tuple, trunc_normal_
from models import DepthNet  
from models.UNet import UNet 
from dc import  *
import torch.fft as fft
import torch
from einops import rearrange
import torch.nn as nn


class SBM(nn.Module):
    def __init__(self, dim, kernel_sizes=[3, 5, 7], dilations=[1, 1, 1], bias=False):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.num_branches = len(kernel_sizes)
        self.dim = dim
        
        # 计算所有分支卷积核展平后的总参数量
        out_dim = sum(k ** 2 for k in kernel_sizes)
        self.out_dim = out_dim
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 用于根据特征相似度生成动态权重的线性层
        self.weights_proj = nn.Linear(dim, out_dim, bias=bias)
        self.act = nn.Softmax(dim=-1)
        
        # 初始化权重投影层
        nn.init.kaiming_normal_(self.weights_proj.weight, mode='fan_out', nonlinearity='relu')
        
        
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * self.num_branches, dim * self.num_branches, kernel_size=3, stride=1, padding=1,
                      groups=dim * self.num_branches, bias=bias),
            nn.Conv2d(dim * self.num_branches, self.num_branches, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Softmax(dim=1)
        )
        # ==================================================
        
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.inside_all = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        

        self.final_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x_dec, x_enc):
        B, C, H, W = x_enc.shape

        enc_vec = self.gap(x_enc).squeeze(-1).squeeze(-1) # [B, C]
        dec_vec = self.gap(x_dec).squeeze(-1).squeeze(-1) # [B, C]

        with torch.amp.autocast(device_type='cuda', enabled=False):

            sim = torch.bmm(enc_vec.float().unsqueeze(2), dec_vec.float().unsqueeze(1)) 

            sim = torch.clamp(sim, min=-65500.0, max=65500.0)
     
        sim_flat = sim.view(B * C, C).type_as(enc_vec) 
        conv_weights = self.weights_proj(sim_flat) 
        conv_weights = torch.clamp(conv_weights, min=-65500.0, max=65500.0)
        
        branch_outs = []
        param_idx = 0
        
        
        for i in range(self.num_branches):
            kernel_size = self.kernel_sizes[i]
            dilation = self.dilations[i]
            param_size = kernel_size ** 2
            
            params = conv_weights[:, param_idx:param_idx + param_size]
            params = self.act(params) 
            param_idx += param_size

            params = params.view(B * C, 1, kernel_size, kernel_size)

            x_enc_res = x_enc.reshape(1, B * C, H, W)

            branch_out = F.conv2d(
                x_enc_res,
                weight=params,
                stride=1,
                padding=((kernel_size - 1) // 2) * dilation,
                dilation=dilation,
                groups=B * C
            )
            branch_out = branch_out.view(B, C, H, W)
            branch_outs.append(branch_out)

        fused = torch.cat(branch_outs, dim=1) 
        with torch.amp.autocast(device_type='cuda', enabled=False):
            masks = self.fuse(fused.float()) 
        masks = masks.type_as(fused)
        # ===================================
        
        stacked = torch.stack(branch_outs, dim=1) 
        low_part = (stacked * masks.unsqueeze(2)).sum(dim=1)
        
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(x_enc)
        out_low = out_low * self.lamb_l[None, :, None, None]
        
        out_high = (x_enc) * (self.lamb_h[None, :, None, None] + 1.)

        x_enc_new = out_low + out_high
        final_out = self.final_conv(torch.cat([x_enc_new, x_dec], dim=1))

        final_out = torch.nan_to_num(final_out, nan=0.0, posinf=10.0, neginf=-10.0)
        final_out = torch.clamp(final_out, min=-10.0, max=10.0)


        
        return final_out

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel,filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel,filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DepthLocalStatsExtractor(nn.Module):
    def __init__(self, window_size=9, eps=1e-6, moment_eps=1e-2, moment_clip=3.0):
        super(DepthLocalStatsExtractor, self).__init__()
        self.window_size = window_size
        self.eps = eps

        
        self.moment_eps = moment_eps

        self.moment_clip = moment_clip

        self.avg_pool = nn.AvgPool2d(
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2,
            count_include_pad=True  
        )
        self.avg_pool_sq = nn.AvgPool2d(
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2,
            count_include_pad=True
        )

    def forward(self, depth_feat, return_higher_order=False):
       
        B, C, H, W = depth_feat.shape

       
        with torch.amp.autocast(device_type='cuda', enabled=False):
            depth_feat_fp32 = depth_feat.float()

            
            depth_local_mean = self.avg_pool(depth_feat_fp32)  
            depth_feat_sq = depth_feat_fp32 ** 2
            depth_local_mean_sq = self.avg_pool_sq(depth_feat_sq)
            depth_local_var = depth_local_mean_sq - (depth_local_mean) ** 2
            depth_local_var = torch.clamp(depth_local_var, min=self.eps)

            higher_order_prior = None

           
            if return_higher_order:
                with torch.no_grad():
                    depth_feat_cubed = depth_feat_sq * depth_feat_fp32        # X^3
                    depth_feat_quad = depth_feat_sq * depth_feat_sq           # X^4

                    depth_local_mean_cubed = self.avg_pool(depth_feat_cubed)  # E[X^3]
                    depth_local_mean_quad = self.avg_pool(depth_feat_quad)    # E[X^4]

            
                    stable_var = torch.clamp(depth_local_var, min=self.moment_eps)
                    stable_std = torch.sqrt(stable_var)

                    skew_num = (
                        depth_local_mean_cubed
                        - 3.0 * depth_local_mean * depth_local_mean_sq
                        + 2.0 * (depth_local_mean ** 3)
                    )

                    kurt_num = (
                        depth_local_mean_quad
                        - 4.0 * depth_local_mean * depth_local_mean_cubed
                        + 6.0 * (depth_local_mean ** 2) * depth_local_mean_sq
                        - 3.0 * (depth_local_mean ** 4)
                    )

                    skewness = skew_num / (stable_std ** 3 + self.moment_eps)
                    kurtosis = kurt_num / (stable_var ** 2 + self.moment_eps)

                    kurtosis = kurtosis - 3.0

                    skewness = torch.tanh(skewness)
                    kurtosis = torch.tanh(kurtosis / 3.0)

                    higher_order_prior = skewness + kurtosis
                    higher_order_prior = torch.clamp(
                        higher_order_prior,
                        min=-self.moment_clip,
                        max=self.moment_clip
                    )

        del depth_feat_sq, depth_local_mean_sq
        if return_higher_order:
            del depth_feat_cubed, depth_feat_quad, depth_local_mean_cubed, depth_local_mean_quad
            del stable_var, stable_std, skew_num, kurt_num, skewness, kurtosis

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_higher_order:
            return (
                depth_local_mean.type_as(depth_feat),
                depth_local_var.type_as(depth_feat),
                higher_order_prior.type_as(depth_feat)
            )
        return depth_local_mean.type_as(depth_feat), depth_local_var.type_as(depth_feat)


class DRDB(nn.Module):
 
    def __init__(self, in_ch=1, out_ch=24, growth_rate=8):  
        super(DRDB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.growth_rate = growth_rate

        self.branch1 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch3 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=3, dilation=3, bias=False)

        self.concat_ch = growth_rate * 3  
        self.enhance_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.concat_ch, self.concat_ch, kernel_size=3, padding=2, dilation=2, groups=self.concat_ch, bias=False),
                nn.Conv2d(self.concat_ch, self.concat_ch, kernel_size=1, bias=False),
                nn.GELU()
            )
            for _ in range(3)  
        ])

        self.match_conv = nn.Conv2d(in_ch, self.concat_ch, kernel_size=1, bias=False)
        self.output_conv = nn.Conv2d(self.concat_ch, out_ch, kernel_size=1, bias=False)
        self.input_residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.final_activation = nn.GELU()

    def forward(self, x):
        b1 = F.gelu(self.branch1(x))
        b2 = F.gelu(self.branch2(x))
        b3 = F.gelu(self.branch3(x))
        F_concat = torch.cat([b1, b2, b3], dim=1)

       
        x_resized = self.match_conv(x)
        out = F_concat + x_resized
        for block in self.enhance_blocks:
            residual = out
            out = block(out)
            out = out + residual
        F_enhance = out

        out = x_resized + F_concat + F_enhance
        out = self.output_conv(out)
        out = self.final_activation(out)

        x_residual = self.input_residual_conv(x)
        out = out + x_residual

        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        out = torch.clamp(out, min=-10.0, max=10.0)

        return out



class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y)
        return y.view(B, C, 1, 1)

class GroupBatchNorm2d(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.view(N, self.num_groups, C // self.num_groups, H, W)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        x = x * self.scale.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return x

class PerformanceEnhancedMaskGating(nn.Module):
    def __init__(self, in_channels: int, gate_threshold: float = 0.5,
                 high_init: float = 0.8, low_init: float = 0.2,
                 num_groups: int = 24, soft_gate_beta: float = 10.0):
        super().__init__()
        self.group_norm = GroupBatchNorm2d(in_channels, num_groups=num_groups)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

        self.high_scale = nn.Parameter(torch.tensor(high_init, dtype=torch.float32))
        self.low_scale = nn.Parameter(torch.tensor(low_init, dtype=torch.float32))
        self.fine_tune_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.soft_gate_beta = soft_gate_beta

    def forward(self, x: torch.Tensor, higher_order_prior: torch.Tensor = None) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_fp32 = x.float()
            norm_feat = self.group_norm(x_fp32)

            channel_scale = self.group_norm.scale.float()
            channel_weights = channel_scale / (channel_scale.sum() + 1e-6)
            channel_weights = channel_weights.view(1, -1, 1, 1)

            info_scores = self.sigmoid(norm_feat * channel_weights)

            high_info_mask = torch.sigmoid((info_scores - self.gate_threshold) * self.soft_gate_beta)
            low_info_mask = 1.0 - high_info_mask

            high_scale = torch.clamp(self.high_scale, min=0.0, max=1.0)
            low_scale = torch.clamp(self.low_scale, min=0.0, max=1.0)

            spatial_weight = (
                high_info_mask * high_scale * info_scores +
                low_info_mask * low_scale * info_scores
            )

            spatial_weight = torch.nan_to_num(spatial_weight, nan=0.0, posinf=1.0, neginf=0.0)
            spatial_weight = torch.clamp(spatial_weight, min=0.0, max=1.0)

            if higher_order_prior is not None:
                higher_order_prior_fp32 = torch.nan_to_num(
                    higher_order_prior.float(), nan=0.0, posinf=0.0, neginf=0.0
                )

                fine_tune_term = torch.tanh(self.fine_tune_weight) * torch.tanh(higher_order_prior_fp32)
                fine_tune_term = torch.clamp(fine_tune_term, min=-0.05, max=0.05)

                spatial_weight = spatial_weight * (1.0 + fine_tune_term)

                spatial_weight = torch.nan_to_num(spatial_weight, nan=0.0, posinf=1.0, neginf=0.0)
                spatial_weight = torch.clamp(spatial_weight, min=0.0, max=1.0)

        return spatial_weight.type_as(x)

class FinalSeparateStatsDehazer(nn.Module):
    def __init__(self, in_channels=3, out_channels=24, hidden_channels=24, init_threshold=0.5):
        super(FinalSeparateStatsDehazer, self).__init__()
        self.hidden_channels = hidden_channels

        self.fog_feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.se_attention = SEAttention(hidden_channels)
        self.depth_encoder = DRDB(in_ch=1, out_ch=hidden_channels)
        self.depth_stats_extractor = DepthLocalStatsExtractor(window_size=9)

        self.mean_gating = PerformanceEnhancedMaskGating(
            in_channels=hidden_channels, gate_threshold=init_threshold, num_groups=hidden_channels
        )
        self.var_gating = PerformanceEnhancedMaskGating(
            in_channels=hidden_channels, gate_threshold=init_threshold, num_groups=hidden_channels
        )

        self.residual_conv = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        self.feat_refiner = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.output_layer = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.raw_img_to_24ch = nn.Conv2d(3, 24, 1, bias=False)

    def forward(self, fog_image, depth_map):
        fog_feat = self.fog_feat_extractor(fog_image)
        fog_feat = fog_feat * self.se_attention(fog_feat)
        fog_feat_residual = self.residual_conv(fog_feat)

        depth_feat_24ch = self.depth_encoder(depth_map)

        depth_mean, depth_var, depth_higher_order = self.depth_stats_extractor(
            depth_feat_24ch, return_higher_order=True
        )

        mean_position_weight = self.mean_gating(depth_mean, depth_higher_order)
        var_position_weight = self.var_gating(depth_var, None)

        fog_feat_mean = fog_feat * mean_position_weight
        fog_feat_var = fog_feat * var_position_weight
        weight_sum = torch.clamp(mean_position_weight + var_position_weight, 0, 1)
        fog_feat_residual_weighted = fog_feat_residual * (1 - weight_sum)
        modulated_fog_feat = fog_feat_mean + fog_feat_var + fog_feat_residual_weighted

        modulated_fog_feat = torch.nan_to_num(modulated_fog_feat, nan=0.0, posinf=10.0, neginf=-10.0)
        modulated_fog_feat = torch.clamp(modulated_fog_feat, min=-10.0, max=10.0)

        refined_feat = self.feat_refiner(modulated_fog_feat)
        dehazed_feat_24ch = self.output_layer(refined_feat)

        dehazed_feat_24ch = torch.nan_to_num(dehazed_feat_24ch, nan=0.0, posinf=10.0, neginf=-10.0)
        dehazed_feat_24ch = torch.clamp(dehazed_feat_24ch, min=-10.0, max=10.0)

        return dehazed_feat_24ch


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class DIACMPN(nn.Module):
    def __init__(self): 
        super(DIACMPN, self).__init__()  
        
        num_res = 8
        base_channel = 32
        self.sbm=SBM(dim=64) 
        self.sbmm=SBM(dim=32) 
        
        self.reduce_conv = nn.Conv2d(
                     in_channels=56,
                     out_channels=32,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False
                         )

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),  #64跳跃链接降通道
            BasicConv(base_channel, base_channel, kernel_size=1, relu=True, stride=1),  #32跳跃连接
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )
        
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.dpn = DepthNet.DN()             #深度估计网络
        self.xy  = FinalSeparateStatsDehazer()  #先验选择


    def forward(self, x):
        _, _, H_ori, W_ori = x.shape
        pad_h = (16 - H_ori % 16) % 16
        pad_w = (16 - W_ori % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        H, W = x.shape[2:]  
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)     
        dp = self.dpn(x)       
        dp = dp[:, :, :H, :W]   
        re=self.xy(x,dp)   
        res=torch.cat([re, res1], dim=1)   
        res2=self.reduce_conv(res) 
        
        # 128尺度特征提取
        z = self.feat_extract[1](res2)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        
        # 64尺度特征提取
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # 解码+64尺度输出（4倍下采样）
        z = self.Decoder[0](z)
        z_64 = self.ConvsOut[0](z)
        z_temp = self.feat_extract[3](z)  
        out_4x = z_64 + x_4

        # 128尺度输出（2倍下采样）
        z =self.sbm(z_temp,res2)  
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_128 = self.ConvsOut[1](z)
        z_temp = self.feat_extract[4](z)  
        out_2x = z_128 + x_2

        # 256尺度输出（原始尺度）
        z=self.sbmm(z_temp,res1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        out_1x = z + x

        if pad_h > 0 or pad_w > 0:
            out_1x = out_1x[:, :, :H_ori, :W_ori]
            out_2x = out_2x[:, :, :H_ori//2, :W_ori//2]
            out_4x = out_4x[:, :, :H_ori//4, :W_ori//4]
        
        return out_1x, out_2x, out_4x
def build_net():  
    return DIACMPN()

