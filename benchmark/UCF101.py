import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--oldmodel', dest='use_old_model', action='store_true', 
                    help='Use the old model in the RIFE repo')
parser.add_argument('--hd', action='store_true', help='Use newer HD model')
parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')
parser.add_argument('--count', type=int, default=-1, help='Evaluate on the first count images')
parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                    help='Output M groups of flow')                      
parser.add_argument('--ctxmergeflow', dest='ctx_use_merged_flow', action='store_true', 
                    help='Use merged flow for contextnet.')

args = parser.parse_args()
args.multi = [ int(m) for m in args.multi.split(",") ]

print(f"Args:\n{args}")

if args.use_old_model:
    model = Model(use_old_model=True)
    model.load_model('rife_checkpoint/flownet.pth')
elif args.hd:
    from train_log.RIFE_HDv3 import Model
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    # -1: rank. If rank <= 0, remove "module" prefix from state_dict keys.
    model.load_model('rife_hd_checkpoint/flownet.pth', -1)
    print("Loaded 3.x/4.x HD model.")
else:
    model = Model(multi=args.multi, 
                  ctx_use_merged_flow=args.ctx_use_merged_flow)
    model.load_model(args.cp)

model.eval()
model.device()

path = 'UCF101/ucf101_interp_ours/'
dirs = os.listdir(path)
psnr_list = []
ssim_list = []
total_triplets = len(dirs)
for i, d in enumerate(dirs):
    img0 = (path + d + '/frame_00.png')
    img1 = (path + d + '/frame_02.png')
    gt = (path + d + '/frame_01_gt.png')
    img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    pred = model.inference(img0, img1)[0]
    ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("{}/{} PSNR {:.3f} Avg {:.3f}, SSIM {:.3f} Avg {:.3f}".format( \
          i+1, total_triplets, psnr, np.mean(psnr_list), ssim, np.mean(ssim_list)))
