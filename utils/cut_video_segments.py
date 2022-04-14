import cv2
import os
import shutil
import datetime
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

    timestamps = set()
    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frame_exists, frame1 = cap.read()
    image1 = transform(frame1, device)
    
    while(cap.isOpened()):
        frame_exists, frame2 = cap.read()
        if frame_exists:
            image2 = transform(frame2, device)
            perceptual_loss = model(image1, image2)
            if perceptual_loss > loss_thres:
                # record current timestamp (millisecond)
                t = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                print('Segment timestamp {}'.format(t))
                timestamps.add(t)
            image1 = image2
        else:
            break
    cap.release()
    return sorted(timestamps)


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
        end = str(datetime.timedelta(milliseconds=t))
        print('Cut video from {} to {}'.format(start, end))
        save_file = os.path.join(save_dir, str(i)+ext)
        os.system("ffmpeg -i {} -ss {} -to {} -async 1 {} -y".format(video_path, start, end, save_file))
        start = end


if __name__=='__main__':
    model = VGGPerceptualLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    video_path = os.path.join('..', os.getcwd(), 'demo.mp4')
    timestamps = video_scene_transition(video_path, model, loss_thres=500)
    save_root = 'data/cut/'
    cut_video_timestamp(video_path, timestamps, save_root)