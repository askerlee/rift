import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import imgaug.augmenters as iaa
import imgaug as ia
from torchvision import transforms

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32, aug_shift_prob=0, shift_sigmas=(16,10)):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = 'vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()
        self.aug_scheme = 'new'
        if self.dataset_name == 'train':
            if self.aug_scheme == 'new':
                tgt_height, tgt_width = 224, 224
                # Mean crop size at any side of the image. delta = 16.
                delta = (self.h - tgt_height) // 2
                affine_prob     = 0.2

                self.geo_aug_func =   iaa.Sequential(
                        [
                            # Randomly crop to 256*256.
                            iaa.CropToAspectRatio(aspect_ratio=1, position='uniform'),
                            # Mean crop length is delta (at one side). So the average output image size
                            # is (self.h - 2*delta) * (self.w - 2*delta).
                            iaa.Crop(px=(0, 2*delta), keep_size=False),
                            # resize the image to the shape of orig_input_size
                            iaa.Resize({'height': tgt_height, 'width': tgt_width}),  
                            # iaa.Sometimes(0.5, iaa.CropAndPad(
                            #     percent=crop_percents,
                            #     pad_mode='constant', # ia.ALL,
                            #     pad_cval=0
                            # )),
                            # apply the following augmenters to most images
                            iaa.Fliplr(0.5),  # Horizontally flip 50% of all images
                            iaa.Flipud(0.5),  # Vertically flip 50% of all images
                            iaa.Sometimes(0.2, iaa.Rot90((1,3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                            # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                            iaa.Sometimes(affine_prob, iaa.Affine(
                                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                                    shear=(-16, 16), # shear by -16 to +16 degrees
                                    order=1,
                                    cval=(0,255),
                                    mode='reflect'
                            )),
                            iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades.
                            # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                            # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                            #iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),
                            #iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                        ])
                        
        self.aug_shift_prob = aug_shift_prob
        self.shift_sigmas = shift_sigmas
                        
    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
            
    # random crop
    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    # img0, img1 and gt are 3D np arrays of (H, W, 3). gt is the middle frame.
    def random_shift(self, img0, img1, gt, reversed_01=False, shift_sigmas=(16,10)):
        u_shift_sigma, v_shift_sigma = shift_sigmas
        # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
        # and [-2*v_shift_sigma, 2*v_shift_sigma].
        dx = np.random.laplace(0, u_shift_sigma)
        dy = np.random.laplace(0, v_shift_sigma)
        # Make sure dx and dy are even numbers.
        dx = (int(dx) // 2) * 2
        dy = (int(dy) // 2) * 2
        # Shift gt by half of (dy, dx).
        dx2 = dx // 2
        dy2 = dy // 2

        # Do not bother to make a special case to handle 0 offsets. 
        # Just discard such shift params.
        if dx == 0 or dy == 0:
            return img0, img1, gt

        if dx > 0 and dy > 0:
            # img0 is cropped at the bottom-right corner. 
            img0a   = img0[:-dy, :-dx]
            # img1 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
            img1a   = img1[dy:,  dx:]
            # gt is shifted by (dx2, dy2) to the left and up, and is also cropped at the bottom-right corner.
            gta     = gt[  dy2:-dy2, dx2:-dx2]
        if dx > 0 and dy < 0:
            # img0 is cropped at the right side, and shifted to the up.
            img0a   = img0[-dy:, :-dx]
            # img1 is shifted to the left and cropped at the bottom.
            img1a   = img1[:dy,  dx:]
            # gt is shifted by (dx2, -dy2) to the left and up, and is also cropped at the bottom-right corner.
            gta     = gt[  -dy2:dy2, dx2:-dx2]
        if dx < 0 and dy > 0:
            # img0 is shifted to the left, and cropped at the bottom.
            img0a   = img0[:-dy, -dx:]
            # img1 is cropped at the right side, and shifted to the up.
            img1a   = img1[dy:,  :dx]
            # gt is shifted by (-dx2, dy2) to the left and up, and is also cropped at the bottom-right corner.
            gta     = gt[  dy2:-dy2, -dx2:dx2]
        if dx < 0 and dy < 0:
            # img0 is shifted by (-dx, -dy) to the left and up.
            img0a   = img0[-dy:, -dx:]
            # img1 is cropped at the bottom-right corner.
            img1a   = img1[:dy,  :dx]
            # gt is shifted by (-dx2, -dy2) to the left and up, and is also cropped at the bottom-right corner.
            gta     = gt[  -dy2:dy2, -dx2:dx2]

        dx2, dy2 = abs(dx2), abs(dy2)
        img0a = np.pad(img0a, ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
        img1a = np.pad(img1a, ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
        gta   = np.pad(gta,   ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')

        if reversed_01:
            img0a, img1a = img1a, img0a
        return img0a, img1a, gta

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
        if self.dataset_name == 'train':
            if self.aug_scheme == 'old':
                img0, gt, img1 = self.aug(img0, gt, img1, 224, 224)
                # reverse the order of the RGB channels
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                # vertical flip
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                # horizontal flip
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]
                # swap img0 and img1
                if random.uniform(0, 1) < 0.5:
                    img0, img1 = img1, img0
            else:                        
                # A fake 9-channel image, so as to apply the same geometric augmentation to img0, img1 and gt.
                comb_img = np.concatenate((img0, gt, img1), axis=2)
                comb_img = self.geo_aug_func.augment_image(comb_img)
                # Separate the fake 9-channel image into 3 normal images.
                img0, gt, img1 = comb_img[:,:,0:3], comb_img[:,:,3:6], comb_img[:,:,6:9]
                # reverse the order of the RGB channels
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]

                # swap img0 and img1
                if random.uniform(0, 1) < 0.5:
                    img0, img1 = img1, img0

        if self.aug_shift_prob > 0:
            rand = random.random()
            if rand < self.aug_shift_prob / 2:
                img0, img1, gt = self.random_shift(img0, img1, gt, False, self.shift_sigmas)
            elif rand >= self.aug_shift_prob / 2 and rand < self.aug_shift_prob:
                img0, img1, gt = self.random_shift(img1, img0, gt, True,  self.shift_sigmas)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)
