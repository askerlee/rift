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

names = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
psnr_list = []
ssim_list = []

for i, name in enumerate(names):
    i0 = cv2.imread('middlebury/other-data/{}/frame10.png'.format(name)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread('middlebury/other-data/{}/frame11.png'.format(name)).transpose(2, 0, 1) / 255.
    gt = (torch.tensor(cv2.imread('middlebury/other-gt-interp/{}/frame10i11.png'.format(name)).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    h, w = i0.shape[1], i0.shape[2]
    imgs = torch.zeros([1, 6, 480, 640]).to(device)
    ph = (480 - h) // 2
    pw = (640 - w) // 2
    imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float().to(device)
    imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float().to(device)
    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]
    pred = model.inference(I0, I2)[0]
    ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    IE = np.abs((out - gt * 1.0)).mean()
    IE_list.append(IE)
    print(np.mean(IE_list))

    print("{}/{} PSNR {:.3f} Avg {:.3f}, SSIM {:.3f} Avg {:.3f}, IE {:.3f} Avg {:.3f}".format( \
          i+1, len(name), psnr, np.mean(psnr_list), ssim, np.mean(ssim_list), IE, np.mean(IE_list)))
