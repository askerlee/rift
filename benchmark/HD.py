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
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read,YUV_Write
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--oldmodel', dest='use_old_model', action='store_true', 
                    help='Use the old model in the RIFE repo')
parser.add_argument('--hd', action='store_true', help='Use newer HD model')
parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')
parser.add_argument('--count', type=int, default=-1, help='Evaluate on the first count images')
parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                    help='Output M groups of flow')

args = parser.parse_args()
args.multi = [ int(m) for m in args.multi.split(",") ]

print(f"Args:\n{args}")

if args.use_old_model:
    model = Model(use_old_model=True)
    model.load_model('checkpoints/rife.pth')
elif args.hd:
    from v4_0.RIFE_HDv3 import Model
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    # -1: rank. If rank <= 0, remove "module" prefix from state_dict keys.
    model.load_model('checkpoints/rife-hd.pth', -1)
    print("Loaded 3.x/4.x HD model.")
else:
    model = Model(multi=args.multi)
    model.load_model(args.cp)

model.eval()
model.device()

name_list = [
    ('HD_dataset/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
    ('HD_dataset/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
    ('HD_dataset/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
    ('HD_dataset/HD1080p_GT/BlueSky.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
    ('HD_dataset/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
    ('HD_dataset/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
    ('HD_dataset/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
]
tot = 0.
for data in name_list:
    psnr_list = []
    name = data[0]
    h = data[1]
    w = data[2]
    if 'yuv' in name:
        Reader = YUV_Read(name, h, w, toRGB=True)
    else:
        Reader = cv2.VideoCapture(name)
    _, lastframe = Reader.read()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(name + '.mp4', fourcc, 30, (w, h))
    for index in range(0, 100, 2):
        if 'yuv' in name:
            IMAGE1, success1 = Reader.read(index)
            gt, _ = Reader.read(index + 1)
            IMAGE2, success2 = Reader.read(index + 2)
            if not success2:
                break
        else:
            success1, gt = Reader.read()
            success2, frame = Reader.read()
            IMAGE1 = lastframe
            IMAGE2 = frame
            lastframe = frame
            if not success2:
                break
        I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        
        if h == 720:
            pad = 24
        elif h == 1080:
            pad = 4
        else:
            pad = 16
        pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
        I0 = pader(I0)
        I1 = pader(I1)
        with torch.no_grad():
            pred = model.inference(I0, I1)
            pred = pred[:, :, pad: -pad]
        out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
        # video.write(out)
        if 'yuv' in name:
            diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
            mse = np.mean((diff_rgb - 128.0) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        else:
            breakpoint()    # obsolete code.
        psnr_list.append(psnr)
    print(np.mean(psnr_list))
    tot += np.mean(psnr_list)
print('avg psnr', tot / len(name_list))
