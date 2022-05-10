import cv2
import os
import argparse
import numpy as np
import re


SINTEL_BLANK_CLIPS = [25, 37, 105, 184, 186]

def is_dark_frame(frame):
    '''frame: np.array(H, W, C)'''
    frame_mean = np.mean(frame, axis=-1)
    pixel_max = np.max(frame_mean)
    pixel_mean = np.mean(frame_mean)
    if pixel_max < 10 and pixel_mean < 10:
        return True
    else:
        return False


def read_video(video_path:str, out_dir:str, v_count:int):
    '''
    Given a single video, filter out short video (less than 3 frames) and empty video (dark frames).
    Save triplet frames into separate folders.
    '''
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_exists, frame1 = cap.read()
    if total_frame < 3 or frame_exists==False:
        return False
    # dark = is_dark_frame(frame1)
    # if dark:
    #     print('Dark Frame!!!')
    #     cv2.imwrite(str(datetime.time), frame1)
    #     return False
    
    new_folder = os.path.join(out_dir, str(v_count).zfill(5)) #eg. 00001
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    seq_count = 1
    new_subfolder = os.path.join(new_folder, str(seq_count).zfill(4)) #eg.0001
    if not os.path.exists(new_subfolder):
        os.makedirs(new_subfolder)
    cv2.imwrite(new_subfolder + '/im1.png', frame1)

    f_count = 1
    while(cap.isOpened()):
        frame_exists, frame = cap.read()
        if frame_exists:
            f_count += 1
            if f_count == 1:
                new_subfolder = os.path.join(out_dir, str(v_count).zfill(5), str(seq_count).zfill(4))
                if not os.path.exists(new_subfolder):
                    os.makedirs(new_subfolder)
                cv2.imwrite(new_subfolder + '/im1.png', frame)
            elif f_count == 2:
                cv2.imwrite(new_subfolder + '/im2.png', frame)
            else: # 3rd frame
                cv2.imwrite(new_subfolder + '/im3.png', frame)
                seq_count += 1
                f_count = 0
        else:
            break
    cap.release()
    print('Sequence:', seq_count)
    assert seq_count < 1e5, "sequence > 10000"
    if f_count != 0:
        print('Folder {} has less than 3 images'.format(new_subfolder))
        os.system("rm -rf {}".format(new_subfolder))
    return True


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    videos = sorted(os.listdir(args.video_root), key=natural_keys)
    v_count = 1
    for v in videos:
        v_n, _ = os.path.splitext(v)
        v_n = int(v_n)
        if v_n in SINTEL_BLANK_CLIPS:
            continue
        v_path = os.path.join(args.video_root, v)
        valid = read_video(v_path, args.out_dir, v_count)
        if valid:
            v_count += 1
    print("Total {} valid videos".format(v_count))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', dest='video_root', default='data/cut/sintel_cut/')
    parser.add_argument('--out_dir', help='folder to save output data', default='data/sintel_triplet/')
    args = parser.parse_args()

    main(args)