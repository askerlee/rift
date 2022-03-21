import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warp import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
from model.IFNet_rife import IFNet_rife
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
import random

device = torch.device("cuda")

# img0, img1, gt are 4D tensors of (B, 3, 256, 448). gt are the middle frames.
def random_shift(img0, img1, gt, shift_sigmas=(16, 10)):
    B, C, H, W = img0.shape
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
    dx2 = dx // 2
    dy2 = dy // 2
    
    # If flow=0, pixels at (dy, dx)_0a <-> (0, 0)_1a.
    if dx >= 0 and dy >= 0:
        # img0 is cropped at the bottom-right corner.               img0[:-dy, :-dx]
        img0_bound = (0,  H - dy,  0,  W - dx)
        # img1 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
        #                                                           img1[dy:,  dx:]
        img1_bound = (dy, H,       dx, W)
    if dx >= 0 and dy < 0:
        # img0 is cropped at the right side, and shifted to the up. img0[-dy:, :-dx]
        img0_bound = (-dy, H,      0,  W - dx)
        # img1 is shifted to the left and cropped at the bottom.    img1[:dy,  dx:]
        img1_bound = (0,   H + dy, dx, W)
        # (dx, 0)_0 => (dx, dy)_0a, (dx, 0)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.          
    if dx < 0 and dy >= 0:
        # img0 is shifted to the left, and cropped at the bottom.   img0[:-dy, -dx:]
        img0_bound = (0,   H - dy, -dx, W)
        # img1 is cropped at the right side, and shifted to the up. img1[dy:,  :dx]
        img1_bound = (dy,  H,      0,   W + dx)
        # (0, dy)_0 => (dx, dy)_0a, (0, dy)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.         
    if dx < 0 and dy < 0:
        # img0 is shifted by (-dx, -dy) to the left and up. img0[-dy:, -dx:]
        img0_bound = (-dy, H,      -dx, W)
        # img1 is cropped at the bottom-right corner.       img1[:dy,  :dx]
        img1_bound = (0,   H + dy, 0,   W + dx)

    # Swapping the shifts to img0 and img1, to increase diversity.
    reversed_01 = random.random() > 0.5
    # Make the shifted img0, img1, gt shifted copies of the same image. Performs slightly worse.
    do_identity_shift = False
    
    # dxy is the motion of the middle frame. It's always half of the relative motion between frames 0 and 1.
    if reversed_01:
        img0_bound, img1_bound = img1_bound, img0_bound
        if do_identity_shift:
            img0, img1, gt = img1, img1, img1
        # Shifting to img0 & img1 are swapped.
        # dxy: offsets (from old to new flow) for two directions.
        # Note the middle frame is shifted by *half* of dx, dy.
        # Note the flows are for backward warping (from middle to 0/1).
        # From 0.5 -> 0: negative delta (from the old flow). old 0.5->0 flow - (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: positive delta (from the old flow). old 0.5->1 flow + (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([-dx2, -dy2,  dx2,  dy2], dtype=float, device=img0.device)
    else:
        if do_identity_shift:
            img0, img1, gt = img0, img0, img0        
        # Note the middle frame is shifted by *half* of dx, dy.
        # From 0.5 -> 0: positive delta (from the old flow). old 0.5->0 flow + (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: negative delta (from the old flow). old 0.5->1 flow - (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([ dx2,  dy2, -dx2, -dy2], dtype=float, device=img0.device)

    # T, B, L, R: top, bottom, left, right boundary.
    T1, B1, L1, R1 = img0_bound
    T2, B2, L2, R2 = img1_bound
    # For the middle frame, the numbers of cropped pixels at the left and right, or the up and the bottom are equal.
    # Therefore, after padding, the middle frame doesn't shift. It's just cropped at the center and 
    # zero-padded at the four sides.
    # This property makes it easy to compare the flow before and after shifting.
    dx2, dy2 = abs(dx2), abs(dy2)
    # TM, BM, LM, RM: new boundary of the middle frame.
    TM, BM, LM, RM = dy2, H - dy2, dx2, W - dx2
    img0a = img0[:, :, T1:B1, L1:R1]
    img1a = img1[:, :, T2:B2, L2:R2]
    gta   = gt[:, :, TM:BM, LM:RM]

    # Pad img0a, img1a, gta by half of (dy, dx), to the original size.
    # Note the pads are ordered as (x1, x2, y1, y2) instead of (y1, y2, x1, x2). 
    # The order is different from np.pad().
    img0a = F.pad(img0a, (dx2, dx2, dy2, dy2))
    img1a = F.pad(img1a, (dx2, dx2, dy2, dy2))
    gta   = F.pad(gta,   (dx2, dx2, dy2, dy2))

    dxy = dxy.view(1, 4, 1, 1)

    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    # mask for the middle frame. Both directions have the same mask.
    mask = torch.zeros(mask_shape, device=img0.device, dtype=bool)
    mask[:, :, TM:BM, LM:RM] = True
    return img0a, img1a, gta, mask, dxy

class Model:
    def __init__(self, local_rank=-1, use_old_model=False, grad_clip=-1, 
                 distill_loss_weight=0.015, 
                 multi=(8,8,4), 
                 weight_decay=1e-3,
                 cons_shift_prob=0,
                 shift_sigmas=(16,10),
                 consist_loss_weight=0.05,
                 debug=False):
        #if arbitrary == True:
        #    self.flownet = IFNet_m()
        if use_old_model:
            self.flownet = IFNet_rife()
        else:
            self.flownet = IFNet(multi)
        self.device()

        conv_param_groups, trans_param_groups = [], []
        for name, param in self.flownet.named_parameters():
            if 'trans' in name:
                trans_param_groups.append(param)
            else:
                conv_param_groups.append(param)

        # lr here is just a placeholder. Will be overwritten in update(), 
        # where the actual LR is obtained from train.py:get_learning_rate().
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=weight_decay)

        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1 and (not debug):
            self.flownet = DDP(self.flownet, device_ids=[local_rank], 
                               output_device=local_rank,
                               find_unused_parameters=True)
        self.distill_loss_weight = distill_loss_weight
        self.grad_clip = grad_clip
        self.cons_shift_prob = cons_shift_prob
        self.shift_sigmas = shift_sigmas
        self.consist_loss_weight = consist_loss_weight

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load(path)))
        else:
            self.flownet.load_state_dict(torch.load(path))

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/ep{:03}.pth'.format(path, epoch))

    def inference(self, img0, img1, scale=1, TTA=False, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]        
        flow, mask, merged_img_list, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        stu_pred = merged_img_list[2]
        if TTA == False:
            return stu_pred
        else:
            flow2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (stu_pred + merged_img_list2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, weight_decay=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
            param_group['weight_decay'] = weight_decay
            
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged_img_list, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale_list=[4, 2, 1])
        if self.cons_shift_prob > 0 and random.random() < self.cons_shift_prob:
            # smask: shift mask. dxy: (dx, dy).
            img0a, img1a, gta, smask, dxy = random_shift(img0, img1, gt, self.shift_sigmas)

            if dxy is not None:
                imgsa = torch.cat((img0a, img1a), 1)
                flow2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(torch.cat((imgsa, gta), 1), scale_list=[4, 2, 1])
                loss_consist_stu = 0
                # s enumerates all scales.
                loss_on_scales = np.arange(len(flow))
                for s in loss_on_scales:
                    loss_consist_stu += torch.abs(flow[s] + dxy - flow2[s])[smask].mean()

                loss_consist_tea = torch.abs(flow_teacher + dxy - flow_teacher2)[smask].mean()
                loss_consist = (loss_consist_stu / len(loss_on_scales) + loss_consist_tea) / 2
                mean_shift = dxy.abs().mean().item()
            else:
                loss_consist = 0
                mean_shift = 0
                loss_distill2 = 0
        else:
            loss_consist = 0
            mean_shift = 0
            loss_distill2 = 0

        only_calc_final_loss = True
        if only_calc_final_loss:
            stu_pred = merged_img_list[2]
            loss_stu = (self.lap(stu_pred, gt)).mean()
        else:
            loss_stu = 0
            for stu_pred in merged_img_list:
                # lap: laplacian pyramid loss.
                loss_stu += (self.lap(stu_pred, gt)).mean()
            loss_stu = loss_stu / len(merged_img_list)

        # loss_tea: laplacian pyramid loss between warped image by teacher's flow & the ground truth image
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            CONS_DISTILL_DISCOUNT = 2
            # loss_distill: L1 loss between the teacher's flow and the student's flow.
            # loss_distill2: the distillation loss when the input is shifted. 
            # Discounted by 2, so the effective weight is 0.01.
            loss_G = loss_stu + loss_tea + (loss_distill + loss_distill2 / CONS_DISTILL_DISCOUNT) * self.distill_loss_weight \
                     + loss_consist * self.consist_loss_weight
            loss_G.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), self.grad_clip)

            self.optimG.step()
        else:
            flow_teacher = flow[2]

        return stu_pred, {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_stu': loss_stu,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'loss_consist': loss_consist,
            'mean_shift': mean_shift
            }
