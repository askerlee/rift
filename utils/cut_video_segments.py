import cv2
import os
import shutil
import datetime
import argparse
import torch
from vgg_perceptual_loss import VGGPerceptualLoss


def transform(x, device):
    '''
    x: np.array(H, W, C)
    return: torch.Tensor(1, C, H, W)
    '''
    x = torch.Tensor(x).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def video_scene_transition(video_path, model, loss_thres=500):
    _, f_ext = os.path.splitext(video_path)
    assert f_ext in ['.mp4', '.avi', '.mkv'], 'Video file extension is not supported'
    assert os.path.exists(video_path), 'Video file not found'
    device = model.mean.device

    timestamps = [0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_exists, frame1 = cap.read()
    image1 = transform(frame1, device)
    dt = 1000 / fps
    triplet_t = dt * 12

    t = 0
    num_seg = 0
    while(cap.isOpened()):
        frame_exists, frame2 = cap.read()
        if frame_exists:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) #current timestamp (millisecond)
            image2 = transform(frame2, device)
            perceptual_loss = model(image1, image2, feature_layers=[3])
            # tt = str(datetime.timedelta(milliseconds=t))
            # with open('frame_loss_cos.txt', 'a') as f:
            #     f.writelines(tt+'\t'+str(perceptual_loss.item())+'\n')
            delta_t = t - timestamps[-1]
            if perceptual_loss > loss_thres and delta_t > triplet_t:
                num_seg += 1
                print('{}: Loss {}, Segment timestamp {}'.format(num_seg, perceptual_loss, t))
                timestamps.append(t)
                tt = str(datetime.timedelta(milliseconds=t))
                with open('timestamp.txt', 'a') as f:
                    f.writelines(str(t)+','+tt+'\n')
            image1 = image2
        else:
            break
    cap.release()
    # add ending timestamp to list
    timestamps.append(t)
    timestamps = sorted(timestamps)[1:]
    return timestamps


def cut_video_timestamp(video_path:str, timestamps:list, save_root:str):
    v_path = video_path.split('/')[-1]
    v_name, ext = os.path.splitext(v_path)
    save_dir = os.path.join(save_root, v_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if timestamps is None:
        shutil.copyfile(video_path, os.path.join(save_dir, '0'+ext))
        return
    else:
        print('Cut video into {} pieces'.format(len(timestamps)-1))
    start = "00:00:00"
    for i, t in enumerate(timestamps):
        if isinstance(t, float):
            end = str(datetime.timedelta(milliseconds=t))
        elif isinstance(t, str):
            end = t
        print('Cut video from {} to {}'.format(start, end))
        save_file = os.path.join(save_dir, str(i+1)+ext)
        if os.path.exists(save_file):
            start = end
            continue
        os.system("ffmpeg -i {} -ss {} -to {} -async 1 -qscale 0 {} -y".format(video_path, start, end, save_file))
        start = end


def cut_video(video_path, start, end):
    v_path = video_path.split('/')[-1]
    name, ext = os.path.splitext(v_path)
    save_dir = video_path[:-len(v_path)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, name+'_cut'+ext)
    if os.path.exists(save_file):
        return save_file
    # os.system("ffmpeg -i {} -ss {} -to {} -acodec copy -vcodec copy {} -y".format(video_path, start, end, save_file)) # correct ending, broke start ts
    # os.system("ffmpeg -i {} -ss {} -to {} -c copy {} -y".format(video_path, start, end, save_file)) # broke frames, wrong start ts
    # os.system("ffmpeg -i {} -ss {} -to {} -async 1 {} -y".format(video_path, start, end, save_file)) # cut at correct start ts, quality degraded
    os.system("ffmpeg -i {} -ss {} -to {} -async 1 -qscale 0 {} -y".format(video_path, start, end, save_file)) # cut at correct start ts, quality reserved but file size increased
    print('Saved file', save_file)
    return save_file


def read_timestamp_from_file(f_name):
    timestamp = []
    timestamp_str = []
    with open(f_name, 'r') as f:
        lines = f.readlines()
    for line in lines:
        l = float(line.split(',')[0])
        s = line.split(',')[1]
        s = s.split('\n')[0]
        timestamp.append(l)
        timestamp_str.append(s)
    return timestamp, timestamp_str


def main(args):
    model = VGGPerceptualLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if args.video_path != '':
        video_path = args.video_path
    else:
        video_path = os.path.join('..', os.getcwd(), 'demo.mp4')
    
    cut_vid = cut_video(video_path, start="00:00:04", end="00:12:23")
    # timestamps = video_scene_transition(cut_vid, model, loss_thres=args.loss_thres)
    timestamps, timestamp_str = read_timestamp_from_file('timestamps.txt')
    cut_video_timestamp(cut_vid, timestamp_str, args.save_root)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', dest='video_path', default='')
    parser.add_argument('--loss_thres', help='VGG perceptual loss threshold', type=float, default=10)
    parser.add_argument('--save_root', help='folder to save video clips', default='data/cut/')
    args = parser.parse_args()

    main(args)