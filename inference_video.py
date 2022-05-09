from ctypes import resize
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from model.RIFT import RIFT

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1, 
                    help='increase the multiply ratio with 2**exp')
parser.add_argument('--mul', dest='mul', type=int, default=2, help='multiply fps by this ratio')
# RIFT model options
parser.add_argument('--oldmodel', dest='use_old_model', action='store_true', 
                    help='Use the old model in the RIFE repo')
parser.add_argument('--hd', action='store_true', help='Use newer HD model')
parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')
parser.add_argument('--count', type=int, default=-1, help='Evaluate on the first count images')
parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                    help='Output M groups of flow')                      

args = parser.parse_args()
if args.exp != 1:
    args.mul = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True
args.multi = [ int(m) for m in args.multi.split(",") ]
print(f"Args:\n{args}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

if args.use_old_model:
    model = RIFT(use_old_model=True)
    if not hasattr(model, 'version'):
        model.version = 0    
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
    model.version = 1.0
        
model.eval()
model.device()

if args.video is not None:
    if os.path.splitext(args.video)[1] != '.yuv':
        videoCapture = cv2.VideoCapture(args.video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        videogen = skvideo.io.vreader(args.video)
        lastframe = next(videogen)
    else: # '.yuv'
        from benchmark.yuv_frame_io import YUV_Read,YUV_Write
        video_structs_dict = {
            'HD_dataset/HD720p_GT/parkrun_1280x720_50.yuv': [720, 1280],
            'HD_dataset/HD720p_GT/shields_1280x720_60.yuv': [720, 1280],
            'HD_dataset/HD720p_GT/stockholm_1280x720_60.yuv': [720, 1280],
            'HD_dataset/HD1080p_GT/BlueSky.yuv': [1080, 1920],
            'HD_dataset/HD1080p_GT/Kimono1_1920x1080_24.yuv': [1080, 1920],
            'HD_dataset/HD1080p_GT/ParkScene_1920x1080_24.yuv': [1080, 1920],
            'HD_dataset/HD1080p_GT/sunflower_1080p25.yuv': [1080, 1920],
            'HD_dataset/HD544p_GT/Sintel_Alley2_1280x544.yuv': [544, 1280],
            'HD_dataset/HD544p_GT/Sintel_Market5_1280x544.yuv': [544, 1280],
            'HD_dataset/HD544p_GT/Sintel_Temple1_1280x544.yuv': [544, 1280],
            'HD_dataset/HD544p_GT/Sintel_Temple2_1280x544.yuv': [544, 1280]
        }
        name = args.video
        h = video_structs_dict[name][0]
        w = video_structs_dict[name][1]
        Reader = YUV_Read(name, h, w, toRGB=True)
        lastframe, lastframe_bool = Reader.read()
        videogen = []
        for index in range(Reader.frame_length):
            img, success = Reader.read(index)
            if not success:
                break
            videogen.append(img)
        videogen = videogen[1:]
        fps = args.fps if args.fps is not None else 15 # fps of files in HD544p_GT seems to be 15
        tot_frame = 100 # refer HD.py L75: for index in range(0, 100, 2):
    
    if args.fps is None:
        fpsNotAssigned = True
    else:
        fpsNotAssigned = False
    args.fps = fps * args.mul
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.mul, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def resize_image(img, size):
    if(args.fp16):
        return F.interpolate(img, size, mode='bilinear', align_corners=False).half()
    else:
        return F.interpolate(img, size, mode='bilinear', align_corners=False)

if args.montage:
    left = w // 4
    w = w // 2
# Resize images to multiple of 128, to facilitate model processing.
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = resize_image(I1, (ph, pw))
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = resize_image(I1, (ph, pw))
    # I0_small, I1_small are only used to compute ssim.
    I0_small = resize_image(I0, (32, 32))
    I1_small = resize_image(I1, (32, 32))
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = resize_image(I1, (ph, pw))
        I1 = model.inference(I0, I1, args.scale)
        # I1_small is only used to compute ssim.
        I1_small = resize_image(I1, (32, 32))
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        # frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
    if ssim < 0.2:
        output = []
        for i in range(args.mul - 1):
            output.append(I0)
        '''
        output = []
        step = 1 / args.mul
        alpha = 0
        for i in range(args.mul - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(I0, I1, args.mul-1)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = resize_image(mid, (h, w))
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = resize_image(mid, (h, w))
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid)
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
