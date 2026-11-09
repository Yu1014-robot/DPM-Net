import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

from models import DepthNet, DIACMPN, build_net
from pytorch_ssim import ssim
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from utils.metrics import psnr
import time
import cv2

import setproctitle
setproctitle.setproctitle("Dehaze_Test")

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--dehaze_result_dir', default='./results/dehaze_result/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def test_dehaze(test_loader, network, dehaze_result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(dehaze_result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(dehaze_result_dir, 'dehaze_results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            # 原图输出
            output1 = network(input)[0].clamp_(-1, 1)

            # 水平翻转 TTA
            flipped_input = torch.flip(input, dims=[3])
            flipped_output = network(flipped_input)[0].clamp_(-1, 1)
            flipped_output = torch.flip(flipped_output, dims=[3])  # 翻回来

            # 平均融合输出
            output = (output1 + flipped_output) / 2.0

            # 映射回 [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            # 防止溢出，提高 PSNR 精度
            output = output.clamp(0, 1)

            # 更精确 PSNR 计算
            psnr_val = psnr(output, target)

            # 更精确 SSIM（可选缩放）
            ssim_val = ssim(output, target, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test_dehaze: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(dehaze_result_dir, 'imgs', filename), out_img)

    f_result.close()
    os.rename(os.path.join(dehaze_result_dir, 'dehaze_results.csv'),
              os.path.join(dehaze_result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)))#原配


if __name__ == '__main__':
    network = build_net()
    DEPTH_NET = DepthNet.DN().cuda()
    network.cuda()
    DEPTH_NET.cuda()

    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

    if os.path.exists(saved_model_dir):

        print('==> Start testing, current model name: ' + args.model)

        dehaze_state_dict = torch.load(saved_model_dir)
        depth_state_dict = torch.load('/root/autodl-tmp/DCMPNet-main/saved_models/indoor/DIACMPN-depth-Indoor.pth')   
        network.load_state_dict(dehaze_state_dict['dehaze_net'])
        DEPTH_NET.load_state_dict(depth_state_dict['depth_net'])

    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    dehaze_result_dir = os.path.join(args.dehaze_result_dir, args.dataset, args.model)
    test_dehaze(test_loader, network, dehaze_result_dir)