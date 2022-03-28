# RIFT - <ins>R</ins>obust V<ins>I</ins>deo <ins>F</ins>rame Interpolation with Mul<ins>T</ins>iple Intermediate Flow 见缝插帧


## Introduction
This project is the implement of "RIFT - <ins>R</ins>obust V<ins>I</ins>deo <ins>F</ins>rame Interpolation with Mul<ins>T</ins>iple Intermediate Flow" 见缝插帧. The paper is still **under preparation**.

### Installation

```
git clone https://github.com/askerlee/rift
cd rift
pip3 install -r requirements.txt
```

### Run

**Video Frame Interpolation**

You can use demo.mp4 or your own video. 
```
python3 inference_video.py --exp=1 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --exp=2 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5
```
(If your video has very high resolution such as 4K, we recommend set --scale=0.5 (default 1.0). If you generate disordered pattern on your videos, try set --scale=2.0. This parameter control the process resolution for optical flow model.)
```
python3 inference_video.py --exp=2 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)
```
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
(add slomo effect, the audio will be removed)
```
python3 inference_video.py --video=video.mp4 --montage --png
```
(if you want to montage the origin video and save the png format output)

**Image Interpolation**

```
python3 inference_img.py --img img0.png img1.png --exp=4
```
(2^4=16X interpolation results)
After that, you can use pngs to generate mp4:
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
```
You can also use pngs to generate gif:
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

## Evaluation

**UCF101**: Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow) at ./UCF101/ucf101_interp_ours/

**Vimeo90K**: Download [Vimeo90K dataset](http://toflow.csail.mit.edu/) at ./vimeo_triplet

**MiddleBury**: Download [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/) at ./middlebury/other-data and ./middlebury/other-gt-interp

**HD**: Download [HD dataset](https://github.com/baowenbo/MEMC-Net) at ./HD_dataset. The RIFE authors also provide a [google drive download link](https://drive.google.com/file/d/1iHaLoR2g1-FLgr9MEv51NH_KQYMYz-FA/view?usp=sharing).
```
# RIFT
python3 benchmark/UCF101.py --cp checkpoints/rift-02171158.pth
# "PSNR: 35.331 SSIM: 0.969"
python3 benchmark/Vimeo90K.py --cp checkpoints/rift-02171158.pth
# "PSNR: 35.781 SSIM: 0.979"
python3 benchmark/MiddleBury_Other.py --cp checkpoints/rift-02171158.pth
# "PSNR: 37.822 SSIM: 0.986 IE: 1.935"
python3 benchmark/HD.py --cp checkpoints/rift-02171158.pth
# "PSNR: 32.321. 544*1280: 25.704, 720p: 33.766, 1080p: 37.855"
```

## Training and Reproduction
Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

We use 2 GPUs and 20G RAM each GPU for training: 
```
torchrun train.py --consshiftprob 0.1 --multi 8,8,4
```

## Citation
We have not released the RIFT paper on arxiv yet. Before that, please consider citing our precursor, the RIFE paper as follows:
```
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```

## Reference
Our code is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE/). We thank [hzwer](https://github.com/hzwer) for his tremendous help during the development.
