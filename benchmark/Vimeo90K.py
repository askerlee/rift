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
parser.add_argument('--trans', dest='trans_layer_indices', default="-1", type=str, 
                    help='Which IFBlock to apply transformer (default: "-1", not to use transformer in any blocks)')
parser.add_argument('--paper', action='store_true', help='Use the model in the RIFE paper')
parser.add_argument('--hd', action='store_true', help='Use newer HD model')
parser.add_argument('--rife', dest='use_rife_settings', action='store_true', help='Use rife settings')
parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')

args = parser.parse_args()
args.trans_layer_indices = [ int(idx) for idx in args.trans_layer_indices.split(",") ]

if args.paper:
    model = Model(use_rife_settings=True)
    model.load_model('rife_checkpoint/flownet.pth')
elif args.hd:
    from train_log.RIFE_HDv3 import Model
    model = Model(use_rife_settings=True)
    if not hasattr(model, 'version'):
        model.version = 0
    # -1: rank. If rank <= 0, remove "module" prefix from state_dict keys.
    model.load_model('rife_hd_checkpoint/flownet.pth', -1)
    print("Loaded 3.x/4.x HD model.")
else:
    model = Model(use_rife_settings=args.use_rife_settings, 
                  trans_layer_indices=args.trans_layer_indices)
    model.load_model(args.cp)

model.eval()
model.device()

path = 'vimeo_triplet/'
f = open(path + 'tri_testlist.txt', 'r')
psnr_list = []
ssim_list = []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'sequences/' + name + '/im1.png')
    I0 = cv2.imread(path + 'sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + 'sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + 'sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    mid = model.inference(I0, I2)[0]
    ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.    
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
