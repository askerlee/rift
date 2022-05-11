import os
import cv2
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from torchvision.transforms import ColorJitter

#cv2.setNumThreads(1)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img0, img1, mid_gt are 3D np arrays of (H, W, 3).
def random_shift(img0, img1, mid_gt, shift_sigmas=(10,8)):
    u_shift_sigma, v_shift_sigma = shift_sigmas
    # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    # Make sure at most one of dx, dy is large. Otherwise the shift is too difficult.
    if random.random() > 0.5:
        dx = np.random.laplace(0, u_shift_sigma / 4)
        dy = np.random.laplace(0, v_shift_sigma)
    else:
        dx = np.random.laplace(0, u_shift_sigma)
        dy = np.random.laplace(0, v_shift_sigma / 4)
        
    # Make sure dx and dy are even numbers.
    dx = (int(dx) // 2) * 2
    dy = (int(dy) // 2) * 2

	# If flow=0, pixels at (dy, dx)_0a <-> (0, 0)_1a.
    if dx >= 0 and dy >= 0:
        # img0 is cropped at the bottom-right corner.               img0[:-dy, :-dx]
        img0_bound = (0,  img0.shape[0] - dy,  0,  img0.shape[1] - dx)
        # img1 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
        #                                                           img1[dy:,  dx:]
        img1_bound = (dy, img0.shape[0],       dx, img0.shape[1])
    if dx >= 0 and dy < 0:
        # img0 is cropped at the right side, and shifted to the up. img0[-dy:, :-dx]
        img0_bound = (-dy, img0.shape[0],      0,  img0.shape[1] - dx)
        # img1 is shifted to the left and cropped at the bottom.    img1[:dy,  dx:]
        img1_bound = (0,   img0.shape[0] + dy, dx, img0.shape[1])
        # (dx, 0)_0 => (dx, dy)_0a, (dx, 0)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.        
    if dx < 0 and dy >= 0:
        # img0 is shifted to the left, and cropped at the bottom.   img0[:-dy, -dx:]
        img0_bound = (0,   img0.shape[0] - dy, -dx, img0.shape[1])
        # img1 is cropped at the right side, and shifted to the up. img1[dy:,  :dx]
        img1_bound = (dy,  img0.shape[0],      0,   img0.shape[1] + dx)
        # (0, dy)_0 => (dx, dy)_0a, (0, dy)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.            
    if dx < 0 and dy < 0:
        # img0 is shifted by (-dx, -dy) to the left and up. img0[-dy:, -dx:]
        img0_bound = (-dy, img0.shape[0],      -dx, img0.shape[1])
        # img1 is cropped at the bottom-right corner.       img1[:dy,  :dx]
        img1_bound = (0,   img0.shape[0] + dy, 0,   img0.shape[1] + dx)

    # Swapping the shifts to img0 and img1, to increase diversity.
    reversed_01 = random.random() > 0.5
    if reversed_01:
        img0_bound, img1_bound = img1_bound, img0_bound

    dx2, dy2 = abs(dx) // 2, abs(dy) // 2
    # T, B, L, R: top, bottom, left, right boundary.
    T1, B1, L1, R1 = img0_bound
    T2, B2, L2, R2 = img1_bound
    # TM, BM, LM, RM: new boundary of the middle frame.
    TM, BM, LM, RM = dy2, img0.shape[0] - dy2, dx2, img0.shape[1] - dx2

    img0a   = img0[T1:B1, L1:R1]
    img1a   = img1[T2:B2, L2:R2]
    mid_gta = mid_gt[TM:BM, LM:RM]

    # Pad img0a, img1a, mid_gta by half of (dy, dx), to the original size.
    img0a   = np.pad(img0a,  ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
    img1a   = np.pad(img1a,  ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
    mid_gta = np.pad(mid_gta,    ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')

    return img0a, img1a, mid_gta


class BaseDataset(Dataset):
    def __init__(self, h, w, tgt_height, tgt_width, aug_shift_prob=0, shift_sigmas=(10,8), aug_jitter_prob=0):
        super(BaseDataset, self).__init__()
        self.h = h
        self.w = w
        # Mean crop size at any side of the image. For Vimeo, delta = 16. For Sintel, delta = 32.
        delta = (self.h - tgt_height) // 2
        affine_prob     = 0.1
        perspect_prob   = 0.1 

        self.geo_aug_func = iaa.Sequential(
                [
                    # Resize the image to the size (h, w). When the original image is too big, 
                    # the first resizing avoids cropping too small fractions of the whole image.
                    # For Sintel, (h, w) = (288, 680), around 2/3 of the original image size (436, 1024).
                    # For Vimeo,  (h, w) = (256, 488) is the same as the original image size.
                    iaa.Resize({ 'height': self.h, 'width': self.w }),
                    # As tgt_width=tgt_height=224, the aspect ratio is always 1.
                    # Randomly crop to 256*256 (Vimeo) or 288*288 (Sintel).
                    iaa.CropToAspectRatio(aspect_ratio=tgt_width/tgt_height, position='uniform'),
                    # Crop a random length from uniform(0, 2*delta) (equal length at four sides). 
                    # The mean crop length is delta, and the mean size of the output image is
                    # (self.h - 2*delta) * (self.h - 2*delta) = tgt_height * tgt_height (=tgt_width).
                    iaa.Crop(px=(0, 2*delta), keep_size=False),
                    # Resize the image to the shape of target size.
                    iaa.Resize({'height': tgt_height, 'width': tgt_width}),
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # Horizontally flip 50% of all images
                    iaa.Flipud(0.5),  # Vertically flip 50% of all images
                    iaa.Sometimes(0.2, iaa.Rot90((1, 3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                    # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                    iaa.Sometimes(affine_prob, iaa.Affine(
                            rotate=(-45, 45), # rotate by -45 to +45 degrees
                            shear=(-16, 16), # shear by -16 to +16 degrees
                            order=1,
                            cval=(0,255),
                            mode='constant'  
                            # Previously mode='reflect' and no PerspectiveTransform => worse performance.
                            # Which is the culprit? maybe mode='reflect'? 
                            # But PerspectiveTransform should also have positive impact, as it simulates
                            # a kind of scene changes due to motion.
                    )),
                    iaa.Sometimes(perspect_prob, 
                                  iaa.PerspectiveTransform(scale=(0.01, 0.15), cval=(0,255), mode='constant')), 
                    iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades?
                    # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                    # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                    # iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),
                    # iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                ])
        self.aug_shift_prob     = aug_shift_prob
        self.shift_sigmas       = shift_sigmas
        self.aug_jitter_prob    = aug_jitter_prob
        self.asym_jitter_prob   = 0.2
        self.color_fun          = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)


class VimeoDataset(BaseDataset):
    def __init__(self, dataset_name, batch_size=32, aug_shift_prob=0, shift_sigmas=(10,8), 
                 aug_jitter_prob=0, h=256, w=448):
        super(VimeoDataset, self).__init__(h, w, 224, 224, aug_shift_prob, shift_sigmas, aug_jitter_prob)
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = 'data/vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()
        
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
        print('Loaded {} Vimeo {} triplets.'.format(len(self.meta_data), self.dataset_name))

    # random crop
    def aug(self, img0, mid_gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        mid_gt = mid_gt[x:x+h, y:y+w, :]
        return img0, mid_gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        mid_gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, mid_gt, img1
            
    def __getitem__(self, index):        
        img0, mid_gt, img1 = self.getimg(index)
        if self.dataset_name == 'train':        
            # A fake 9-channel image, so as to apply the same geometric augmentation to img0, img1 and mid_gt.
            comb_img = np.concatenate((img0, mid_gt, img1), axis=2)
            comb_img = self.geo_aug_func.augment_image(comb_img)
            # Separate the fake 9-channel image into 3 normal images.
            img0, mid_gt, img1 = comb_img[:,:,0:3], comb_img[:,:,3:6], comb_img[:,:,6:9]
            # reverse the order of the RGB channels
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                mid_gt = mid_gt[:, :, ::-1]

            # swap img0 and img1
            if random.uniform(0, 1) < 0.5:
                img0, img1 = img1, img0

        if self.aug_shift_prob > 0 and random.random() < self.aug_shift_prob:
            img0, img1, mid_gt = random_shift(img0, img1, mid_gt, self.shift_sigmas)

        # H, W, C => C, H, W
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        mid_gt = torch.from_numpy(mid_gt.copy()).permute(2, 0, 1)

        # A small probability to do individual jittering. 
        # More challenging, therefore smaller prob.        
        if self.aug_jitter_prob > 0 and random.random() < self.aug_jitter_prob:
            if random.random() < self.asym_jitter_prob:
                img0 = self.color_fun(img0)
                img1 = self.color_fun(img1)

                if mid_gt.shape[1] == 3:
                    mid_gt   = self.color_fun(mid_gt)
            else:
                # imgs: 3, C, H, W
                imgs = torch.stack((img0, img1, mid_gt), 0)
                imgs = self.color_fun(imgs)
                # img0, img1, mid_gt: C, H, W
                img0, img1, mid_gt = imgs[0], imgs[1], imgs[2]

        imgs = torch.cat((img0, img1, mid_gt), 0)    
        return imgs


class SintelDataset(BaseDataset):
    def __init__(self, data_root='data/Sintel/', sample_rate=1, aug_shift_prob=0, 
                 shift_sigmas=(10,8), aug_jitter_prob=0, h=288, w=680):
                 # The original size of Sintel images is (436, 1024). First resize to (288, 680), 
                 # a size similar to Vimeo's (256, 448).
                 # Then do extra cropping and augmentation.
        super(SintelDataset, self).__init__(h, w, 224, 224, aug_shift_prob, shift_sigmas, aug_jitter_prob)
        self.data_root = data_root
        self.sample_rate = sample_rate
        sub_roots   = [ 'training/clean', 'training/final', 'test/clean', 'test/final' ]
        folders     = [ os.path.join(self.data_root, d) for d in sub_roots ]
        self.sub_folders = []
        for d in folders:
            self.sub_folders = self.sub_folders + sorted(glob.glob(d + '/*'))
        self.image_paths = [ sorted(glob.glob(d + '/*.png')) for d in self.sub_folders ]
        self.sample_triplet(sample_rate)

    def sample_triplet(self, sample_rate):
        self.triplets = []
        for paths in self.image_paths:
            for i, p in enumerate(paths[:-sample_rate*2]):
                image_path0 = p
                image_path1 = paths[i + sample_rate]
                image_path2 = paths[i + sample_rate * 2]
                self.triplets.append([image_path0, image_path1, image_path2])

        print('Loaded {} Sintel triplets.'.format(len(self.triplets)))

    def __len__(self):
        return len(self.triplets)

    def getimg(self, index):
        imgpaths = self.triplets[index]
        # Load images
        img0    = cv2.imread(imgpaths[0])
        mid_gt  = cv2.imread(imgpaths[1])
        img1    = cv2.imread(imgpaths[2])
        return img0, mid_gt, img1

    def __getitem__(self, index):
        img0, mid_gt, img1 = self.getimg(index)
      
        # A fake 9-channel image, so as to apply the same geometric augmentation to img0, img1 and mid_gt.
        comb_img = np.concatenate((img0, mid_gt, img1), axis=2)
        comb_img = self.geo_aug_func.augment_image(comb_img)
        # Separate the fake 9-channel image into 3 normal images.
        img0, mid_gt, img1 = comb_img[:,:,0:3], comb_img[:,:,3:6], comb_img[:,:,6:9]
        # reverse the order of the RGB channels
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            mid_gt = mid_gt[:, :, ::-1]

        # swap img0 and img1
        if random.uniform(0, 1) < 0.5:
            img0, img1 = img1, img0

        if self.aug_shift_prob > 0 and random.random() < self.aug_shift_prob:
            img0, img1, mid_gt = random_shift(img0, img1, mid_gt, self.shift_sigmas)

        # H, W, C => C, H, W
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        mid_gt = torch.from_numpy(mid_gt.copy()).permute(2, 0, 1)

        # A small probability to do individual jittering. 
        # More challenging, therefore smaller prob.        
        if self.aug_jitter_prob > 0 and random.random() < self.aug_jitter_prob:
            if random.random() < self.asym_jitter_prob:
                img0 = self.color_fun(img0)
                img1 = self.color_fun(img1)

                if mid_gt.shape[1] == 3:
                    mid_gt   = self.color_fun(mid_gt)
            else:
                # imgs: 3, C, H, W
                imgs = torch.stack((img0, img1, mid_gt), 0)
                imgs = self.color_fun(imgs)
                # img0, img1, mid_gt: C, H, W
                img0, img1, mid_gt = imgs[0], imgs[1], imgs[2]

        imgs = torch.cat((img0, img1, mid_gt), 0)    
        return imgs


if __name__=='__main__':
    ds1 = SintelDataset(sample_rate=2)
    print(len(ds1))
    ds1 = SintelDataset(sample_rate=4)
    print(len(ds1))    
    imgs = ds1[0]
    print(imgs.shape)
    ds2 = VimeoDataset(dataset_name='train')
    print(len(ds2))
    imgs = ds2[0]
    print(imgs.shape)
