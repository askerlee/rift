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
from model.RIFT import RIFT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--oldmodel', dest='use_old_model', action='store_true', 
                    help='Use the old model in the RIFE repo')
parser.add_argument('--hd', action='store_true', help='Use newer HD model')
parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')
parser.add_argument('--count', type=int, default=-1, help='Evaluate on the first count images')
parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                    help='Output M groups of flow')                      
parser.add_argument('--each', dest='out_summary', action='store_false', 
                    help='Output the scores of each frame instead of outputting summary only')

args = parser.parse_args()
args.multi = [ int(m) for m in args.multi.split(",") ]
if args.out_summary:
    endl = "\r"
else:
    endl = "\n"

print(f"Args:\n{args}")

if args.use_old_model:
    model = RIFT(use_old_model=True)
    model.load_model('checkpoints/rife.pth')
elif args.hd:
    from model.rife_new.v4_0.RIFE_HDv3 import Model
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    # -1: rank. If rank <= 0, remove "module" prefix from state_dict keys.
    model.load_model('checkpoints/rife-hd.pth', -1)
    print("Loaded 3.x/4.x HD model.")
else:
    model = RIFT(multi=args.multi)
    model.load_model(args.cp)

model.eval()
model.device()

names = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
psnr_list = []
ssim_list = []

for i, name in enumerate(names):
    i0 = cv2.imread('data/middlebury/other-data/{}/frame10.png'.format(name)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread('data/middlebury/other-data/{}/frame11.png'.format(name)).transpose(2, 0, 1) / 255.
    gt = (torch.tensor(cv2.imread('data/middlebury/other-gt-interp/{}/frame10i11.png'.format(name)).transpose(2, 0, 1))).to(device).float().unsqueeze(0)
    gt_norm = gt / 255.
    h, w = i0.shape[1], i0.shape[2]
    imgs = torch.zeros([1, 6, 480, 640]).to(device)
    ph = (480 - h) // 2
    pw = (640 - w) // 2
    imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float().to(device)
    imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float().to(device)
    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]
    pred = model.inference(I0, I2)[0]
    pred = pred[:, :h, :w]
    ssim = ssim_matlab(gt_norm, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255)
    out_norm = out / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    gt_norm = gt / 255.
    psnr = -10 * math.log10(((gt_norm - out_norm) * (gt_norm - out_norm)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    IE = np.abs((out - gt * 1.0)).mean()
    IE_list.append(IE)

    print("{}/{} PSNR {:.3f} Avg {:.3f}, SSIM {:.3f} Avg {:.3f}, IE {:.3f} Avg {:.3f}".format( \
          i+1, len(names), psnr, np.mean(psnr_list), ssim, np.mean(ssim_list), IE, np.mean(IE_list)), end=endl)

if args.out_summary:
    print()
