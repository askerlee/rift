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

video_structs = [
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
def inference(I0, I1, pad, multi=2, arbitrary=True):
    img = [I0, I1]
    if not arbitrary:
        for i in range(multi):
            res = [I0]
            for j in range(len(img) - 1):
                res.append(model.inference(img[j], img[j + 1]))
                res.append(img[j + 1])
            img = res
    else:
        img = [I0]
        p = 2**multi
        for i in range(p-1):
            img.append(model.inference(I0, I1, timestep=(i+1)*(1./p)))
        img.append(I1)
    for i in range(len(img)):
        img[i] = img[i][0][:, pad: -pad]
    return img[1: -1]

video_psnr_list = []        
tot = []

for i, video_struct in video_structs:
    frame_psnr_list = []
    name = video_struct[0]
    h = video_struct[1]
    w = video_struct[2]
    if 'yuv' in name:
        Reader = YUV_Read(name, h, w, toRGB=True)
    else:
        Reader = cv2.VideoCapture(name)
    _, lastframe = Reader.read()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(name + '.mp4', fourcc, 30, (w, h))    
    for index in range(0, 100, 4):
        gt = []
        if 'yuv' in name:
            IMAGE1, success1 = Reader.read(index)
            IMAGE2, success2 = Reader.read(index + 4)
            if not success2:
                break
            for i in range(1, 4):
                tmp, _ = Reader.read(index + i)
                gt.append(tmp)
        else:
            print('Not Implement')
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
            pred = inference(I0, I1, pad)
        for i in range(4 - 1):
            out = (np.round(pred[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
            if 'yuv' in name:
                diff_rgb = 128.0 + rgb2yuv(gt[i] / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
                mse = np.mean((diff_rgb - 128.0) ** 2)
                PIXEL_MAX = 255.0
                psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            else:
                breakpoint()    # obsolete code.
            frame_psnr_list.append(psnr)

    video_psnr = np.mean(frame_psnr_list)
    video_psnr_list.append(video_psnr)
    print("{}/{} PSNR {:.3f}".format(i+1, len(video_structs), video_psnr))

print('Avg: {:.3f}. 544*1280: {:.3f}, 720p: {:.3f}, 1080p: {:.3f}'.format(\
      np.mean(video_psnr_list), np.mean(video_psnr_list[7:11]), 
      np.mean(video_psnr_list[:3]), np.mean(video_psnr_list[3:7])))
