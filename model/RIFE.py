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
def random_shift(img0, img1, gt, shift_sigmas=(16,10)):
    B, C, H, W = img0.shape
    u_shift_sigma, v_shift_sigma = shift_sigmas
    # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    dx = np.random.laplace(0, u_shift_sigma)
    dy = np.random.laplace(0, v_shift_sigma)
    # Make sure dx and dy are even numbers.
    dx = (int(dx) // 2) * 2
    dy = (int(dy) // 2) * 2

    # Do not bother to make a special case to handle 0 offsets. 
    # Just discard such shift params.
    if dx == 0 or dy == 0:
        return img0, img1, gt, None, None

    if dx > 0 and dy > 0:
        # img0 is cropped at the bottom-right corner.               img0[:-dy, :-dx]
        img0_bound = (0,  img0.shape[0] - dy,  0,  img0.shape[1] - dx)
        # img1 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
        #                                                           img1[dy:,  dx:]
        img1_bound = (dy, img0.shape[0],       dx, img0.shape[1])
    if dx > 0 and dy < 0:
        # img0 is cropped at the right side, and shifted to the up. img0[-dy:, :-dx]
        img0_bound = (-dy, img0.shape[0],      0,  img0.shape[1] - dx)
        # img1 is shifted to the left and cropped at the bottom.    img1[:dy,  dx:]
        img1_bound = (0,   img0.shape[0] + dy, dx, img0.shape[1])
    if dx < 0 and dy > 0:
        # img0 is shifted to the left, and cropped at the bottom.   img0[:-dy, -dx:]
        img0_bound = (0,   img0.shape[0] - dy, -dx, img0.shape[1])
        # img1 is cropped at the right side, and shifted to the up. img1[dy:,  :dx]
        img1_bound = (dy,  img0.shape[0],      0,   img0.shape[1] + dx)
    if dx < 0 and dy < 0:
        # img0 is shifted by (-dx, -dy) to the left and up. img0[-dy:, -dx:]
        img0_bound = (-dy, img0.shape[0],      -dx, img0.shape[1])
        # img1 is cropped at the bottom-right corner.       img1[:dy,  :dx]
        img1_bound = (0,   img0.shape[0] + dy, 0,   img0.shape[1] + dx)

    # Swapping the shifts to img0 and img1, to increase diversity.
    reversed_01 = random.random() > 0.5
    # Shift gt (middle frame) by half of (dy, dx).
    dx2, dy2 = abs(dx) // 2, abs(dy) // 2

    if reversed_01:
        img0_bound, img1_bound = img1_bound, img0_bound
        # Shifting to img0 & img1 are swapped.
        # dxy: offsets (from old to new flow) for two directions.
        # Take half of dx, dy as this is the shift for the middle frame.
        # Note the flows are for backward warping (from middle to 0/1).
        # From 0.5 -> 0: negative delta (from the old flow). old 0.5->0 flow - (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: positive delta (from the old flow). old 0.5->1 flow + (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([-dx2, -dy2,  dx2,  dy2], dtype=float, device=img0.device)
    else:
        # From 0.5 -> 0: positive delta (from the old flow). old 0.5->0 flow + (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: negative delta (from the old flow). old 0.5->1 flow - (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([ dx2,  dy2, -dx2, -dy2], dtype=float, device=img0.device)

    T1, B1, L1, R1 = img0_bound
    T2, B2, L2, R2 = img1_bound
    TM, BM, LM, RM = dy2, img0.shape[0] - dy2, dx2, img0.shape[1] - dx2
    img0a = img0[:, :, T1:B1, L1:R1]
    img1a = img1[:, :, T2:B2, L2:R2]
    gta   = gt[:, :, TM:BM, LM:RM]

    # pad img0a, img1a, gta to the original size.
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
                 ctx_use_merged_flow=False,
                 conv_weight_decay=1e-3,
                 cons_shift_prob=0,
                 shift_sigmas=(10,6),
                 consist_loss_weight=0.05):
        #if arbitrary == True:
        #    self.flownet = IFNet_m()
        if use_old_model:
            self.flownet = IFNet_rife()
        else:
            self.flownet = IFNet(multi, ctx_use_merged_flow)
        self.device()

        conv_param_groups, trans_param_groups = [], []
        for name, param in self.flownet.named_parameters():
            if 'trans' in name:
                trans_param_groups.append(param)
            else:
                conv_param_groups.append(param)

        # Use a large weight decay may avoid NaN loss, but reduces transformer performance.
        # lr here is just a placeholder. Will be overwritten in update(), 
        # where the actual LR is obtained from train.py:get_learning_rate().
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=conv_weight_decay)

        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
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
        
    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/ep{:03}.pth'.format(path, epoch))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged_img_list, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        stu_pred = merged_img_list[2]
        if TTA == False:
            return stu_pred
        else:
            flow2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (stu_pred + merged_img_list2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
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
        else:
            loss_consist = 0
            mean_shift = 0

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
            # loss_distill: L1 loss between the teacher's flow and the student's flow.
            loss_G = loss_stu + loss_tea + loss_distill * self.distill_loss_weight \
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
