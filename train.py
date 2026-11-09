import os


import argparse
import json
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from depth.networks import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import DepthNet, DIACMPN, build_net
from models.UNet import UNet
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.loader import PairLoader
from loss.CR_loss import ContrastLoss as crloss
import setproctitle
setproctitle.setproctitle("Dehaze")
#from loss.CR_loss import FreqPolarAttentionPlus #
from pytorch_msssim import ms_ssim, ssim


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DIACMPN-dehaze-Indoor', type=str, help='dehaze model name')
parser.add_argument('--model_depth', default='DIACMPN-depth-Indoor', type=str, help='depth model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')  #16

parser.add_argument('--no_autocast', action='store_true', default=True, help='disable autocast')


parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='', type=str, help='path to dataset')   
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')

CUDA_VISIBLE_DEVICES异常 =====
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, network, DEPTH_NET, criterion_dehaze, criterion_dehaze_cr, criterion_depth,
          optimizer_dehaze, optimizer_depth):

    Dehaze_loss = AverageMeter()
    Depth_loss = AverageMeter()

    

    torch.cuda.empty_cache()
    network.train()
    DEPTH_NET.train()

    for batch in train_loader:
        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)

        
        dehaze_output_img, d128, d64 = network(source_img)

        
        dehaze_output_img_2_depth_img = DEPTH_NET(dehaze_output_img)

        
        with torch.no_grad():
            real_img_2_depth_img = depth_decoder(encoder(target_img))  # 真实无雾深度图
            real_img_2_depth_img = real_img_2_depth_img[("disp", 0)]   # 真实无雾深度图
        

        label_img2 = F.interpolate(target_img, scale_factor=0.5, mode='bilinear')      #1/2真值
        label_img4 = F.interpolate(target_img, scale_factor=0.25, mode='bilinear')     #1/4真值

       
        loss_depth_consis_w = criterion_depth(dehaze_output_img_2_depth_img, real_img_2_depth_img)

        loss_total_depth = loss_depth_consis_w #+ 0.1*loss_depth_consis         #+0.1*loss_depth_consis     #polar损失

        loss_dehaze_consis = criterion_dehaze(dehaze_output_img, target_img)  #256原尺寸的损失


        loss_dehaze_u1 = criterion_dehaze(d128, label_img2)                  ########128的损失
        loss_dehaze_u2 = criterion_dehaze(d64, label_img4)                   #########64的损失

        
        label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(d64, dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        pred_fft2 = torch.fft.fft2(d128, dim=(-2, -1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(target_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(dehaze_output_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
        

        f1 = criterion_dehaze(pred_fft1, label_fft1)
        f2 = criterion_dehaze(pred_fft2, label_fft2)
        f3 = criterion_dehaze(pred_fft3, label_fft3)
        loss_fft = f1 + f2 + f3

        loss_dehaze_total = loss_dehaze_consis + loss_dehaze_u1 + loss_dehaze_u2 + 0.1 * loss_fft
        total_loss = loss_dehaze_total + loss_total_depth

        
        if (not torch.isfinite(loss_dehaze_total)) or (not torch.isfinite(loss_total_depth)) or (not torch.isfinite(total_loss)):
            print("Skip batch because non-finite loss:",
                  "dehaze =", loss_dehaze_total,
                  "depth =", loss_total_depth,
                  "total =", total_loss)
            optimizer_dehaze.zero_grad(set_to_none=True)
            optimizer_depth.zero_grad(set_to_none=True)
            continue
        

        Dehaze_loss.update(loss_dehaze_total.item())
        Depth_loss.update(loss_total_depth.item())

        
        optimizer_dehaze.zero_grad(set_to_none=True)
        optimizer_depth.zero_grad(set_to_none=True)
        
        total_loss.backward()

        
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(DEPTH_NET.parameters(), max_norm=1.0)
        

        optimizer_dehaze.step()
        optimizer_depth.step()

    return Dehaze_loss.avg, Depth_loss.avg


def valid(val_loader, network):

    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)

        with torch.no_grad():
            output = network(source_img)[0].clamp_(-1,1)

            

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))
        output_01 = (output + 1) / 2
        target_01 = (target_img + 1) / 2
        ssim_value = ssim(output_01, target_01, data_range=1.0, size_average=True)
        SSIM.update(ssim_value.item(), source_img.size(0))

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    with torch.no_grad():
        model_path = os.path.join("./depth/models", 'RA-Depth')
        assert os.path.isdir(model_path), \
            "Cannot find a folder at {}".format(model_path)
        print("-> Loading weights from {}".format(model_path))

        encoder_path = os.path.join(model_path, "encoder.pth")
        decoder_path = os.path.join(model_path, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        encoder = hrnet18(False)
        depth_decoder = DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        
        for p in encoder.parameters():
            p.requires_grad = False
        for p in depth_decoder.parameters():
            p.requires_grad = False
       

    
    network = build_net().cuda() 
    DEPTH_NET = DepthNet.DN().cuda()
    deep_estimate_net = UNet().cuda()

    criterion_dehaze = nn.L1Loss()
    criterion_dehaze_cr = crloss()
    criterion_depth = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer_dehaze = torch.optim.Adam(network.parameters(), lr=setting['lr_dehaze'])
        optimizer_depth = torch.optim.Adam(DEPTH_NET.parameters(), lr=setting['lr_depth'])
    elif setting['optimizer'] == 'adamw':
        optimizer_dehaze = torch.optim.AdamW(network.parameters(), lr=setting['lr_dehaze'])
        optimizer_depth = torch.optim.AdamW(DEPTH_NET.parameters(), lr=setting['lr_depth'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler_dehaze = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dehaze, T_max=setting['epochs'], eta_min=1e-6)
    scheduler_depth = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_depth, T_max=setting['epochs'], eta_min=1e-6)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(
        dataset_dir,
        'train',
        'train',
        (256,256),       
        setting['edge_decay'],
        setting['only_h_flip']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=setting['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):

        print('==> Start training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):
            dehaze_loss, depth_loss = train(
                train_loader, network, DEPTH_NET,
                criterion_dehaze, criterion_dehaze_cr, criterion_depth,
                optimizer_dehaze, optimizer_depth
            )

            writer.add_scalar('dehaze_train_loss', dehaze_loss, epoch)
            writer.add_scalar('depth_train_loss', depth_loss, epoch)

            print('Dehaze_loss: ', dehaze_loss, epoch)
            print('Depth_loss: ', depth_loss, epoch)
            scheduler_dehaze.step()
            scheduler_depth.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr, avg_ssim = valid(val_loader, network)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                print('valid_psnr', avg_psnr, epoch)
                print('valid_ssim', avg_ssim, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    Dehaze_state_dict = {
                        'dehaze_net': network.state_dict(),
                        'dehaze_optimizer': optimizer_dehaze.state_dict(),
                        'epoch_dehaze': epoch,
                        'dehaze_net_best_psnr': best_psnr
                    }
                    depth_state_dict = {
                        'depth_net': DEPTH_NET.state_dict(),
                        'depth_optimizer': optimizer_depth.state_dict(),
                        'epoch_depth': epoch
                    }
                    torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model + '.pth'))
                    torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))
                print('best_psnr', best_psnr, epoch)

                writer.add_scalar('best_psnr', best_psnr, epoch)

                with open('./checkpoint/psnr_loss.txt', 'a') as file:
                    file.write('Epoch [{}/{}], Loss_Dehaze: {:.4f}, Loss_Depth: {:.4f}'
                               .format(epoch + 1, setting['epochs'], dehaze_loss, depth_loss))
                    file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
                    file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
                    file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
                    file.write('\n')
    else:
        print('==> 1')


        dehaze_checkpoint = torch.load(os.path.join(save_dir, args.model + '.pth'))
        depth_checkpoint = torch.load(os.path.join(save_dir, args.model_depth + '.pth'))

       
        dehaze_state_dict = dehaze_checkpoint['dehaze_net']
        model_state_dict = network.state_dict()
        
        for key in list(dehaze_state_dict.keys()):
            if 'ln1.weight' in key or 'ln1.bias' in key:
                if key in model_state_dict and dehaze_state_dict[key].shape != model_state_dict[key].shape:
                    del dehaze_state_dict[key]
                    print(f"已删除形状不匹配的参数：{key}")
        
        network.load_state_dict(dehaze_state_dict, strict=False)
        DEPTH_NET.load_state_dict(depth_checkpoint['depth_net'], strict=False)

        best_psnr = dehaze_checkpoint['dehaze_net_best_psnr']
        start_epoch = dehaze_checkpoint['epoch_dehaze'] + 1

        print('Load start_epoch {} ！'.format(start_epoch))
        print('Load depth_optimizer {} ！'.format(optimizer_dehaze))
        print('Load depth_optimizer {} ！'.format(optimizer_depth))

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
            dehaze_loss, depth_loss = train(
                train_loader, network, DEPTH_NET,
                criterion_dehaze, criterion_dehaze_cr, criterion_depth,
                optimizer_dehaze, optimizer_depth
            )

            writer.add_scalar('dehaze_train_loss', dehaze_loss, epoch)
            writer.add_scalar('depth_train_loss', depth_loss, epoch)

            print('Dehaze_loss: ', dehaze_loss, epoch)
            print('Depth_loss: ', depth_loss, epoch)
            scheduler_dehaze.step()
            scheduler_depth.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr, avg_ssim = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                print('valid_psnr', avg_psnr, epoch)
                print('valid_ssim', avg_ssim, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    Dehaze_state_dict = {
                        'dehaze_net': network.state_dict(),
                        'dehaze_optimizer': optimizer_dehaze.state_dict(),
                        'epoch_dehaze': epoch,
                        'loss_dehaze': dehaze_loss,
                        'dehaze_net_best_psnr': best_psnr
                    }
                    depth_state_dict = {
                        'depth_net': DEPTH_NET.state_dict(),
                        'depth_optimizer': optimizer_depth.state_dict(),
                        'epoch_depth': epoch,
                        'loss_depth': depth_loss
                    }
                    torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model + '.pth'))
                    torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)
                print('best_psnr', best_psnr, epoch)

                with open('./checkpoint/psnr_loss.txt', 'a') as file:
                    file.write('Epoch [{}/{}], Loss_Dehaze: {:.4f}, Loss_Depth: {:.4f}'
                               .format(epoch + 1, setting['epochs'], dehaze_loss, depth_loss))
                    file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
                    file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
                    file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
                    file.write('\n')



