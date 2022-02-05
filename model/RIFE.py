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

# img (img0 or img1) and gt are 4D tensors of (B, 3, 256, 448). gt are the middle frames.
# t_img: 0 or 1, indicating which img to shift.
def random_shift(img, gt, t_img, shift_sigmas=(10,6)):
    B, C, H, W = img.shape
    u_shift_sigma, v_shift_sigma = shift_sigmas
    # 90% of delta_x and delta_y are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    delta_x = np.random.laplace(0, u_shift_sigma)
    delta_y = np.random.laplace(0, v_shift_sigma)
    delta_x, delta_y = int(delta_x), int(delta_y)
    # Make sure delta_x and delta_y are even numbers.
    delta_x = (delta_x // 2) * 2
    delta_y = (delta_y // 2) * 2
    # Shift gt (middle frame) by half of (delta_y, delta_x).
    # 90% of gt shift are within [-u_shift_sigma, u_shift_sigma] 
    # and [-v_shift_sigma, v_shift_sigma].
    delta_x2 = delta_x // 2
    delta_y2 = delta_y // 2

    # Do not bother to make a special case to handle 0 offsets. 
    # Just discard such shift params.
    if delta_x == 0 or delta_y == 0:
        return img, gt, None, None, None, None

    img2    = torch.zeros_like(img)
    gt2     = torch.zeros_like(gt)
    mask_shape = list(img.shape)
    mask_shape[1] = 2   # For 2 flow channels (one of the two directions).
    # mask for img
    # mask2 for the other img.
    mask    = torch.zeros(mask_shape, device=img.device, dtype=bool)
    mask2   = torch.zeros(mask_shape, device=img.device, dtype=bool)

    if delta_x > 0 and delta_y > 0:
        img2[:, :, delta_y:, delta_x:]    = img[:, :, :-delta_y,  :-delta_x]
        gt2[ :, :, delta_y2:, delta_x2:]  = gt[ :, :, :-delta_y2, :-delta_x2]
        # mask is for img -> middle (gt).
        # If img=img1, then for img0 -> middle, mask2 is the opposite 
        # block of the mask for img1 -> middle 
        # (continuing the shift from img1 -> middle until reaching the opposite edge).
        # If img=img0, then for img1 -> middle, mask2 is the opposite 
        # block of the mask for img0 -> middle.
        # (continuing the shift from img0 -> middle until reaching the opposite edge).
        mask[:, :,  delta_y:,  delta_x:]  = 1
        mask2[:, :, :-delta_y, :-delta_x] = 1
    if delta_x > 0 and delta_y < 0:
        img2[:,  :, :delta_y,  delta_x:]   = img[:, :, -delta_y:,  :-delta_x]
        gt2[ :,  :, :delta_y2, delta_x2:]  = gt[ :, :, -delta_y2:, :-delta_x2]
        mask[:,  :,  :delta_y,  delta_x:]  = 1
        mask2[:, :, -delta_y:, :-delta_x]  = 1
    if delta_x < 0 and delta_y > 0:
        img2[:,  :, delta_y:, :delta_x]    = img[:, :, :-delta_y,  -delta_x:]
        gt2[ :,  :, delta_y2:, :delta_x2]  = gt[ :, :, :-delta_y2, -delta_x2:]
        mask[:,  :,  delta_y:,  :delta_x]  = 1
        mask2[:, :, :-delta_y, -delta_x:]  = 1
    if delta_x < 0 and delta_y < 0:
        # Note delta_y and delta_x are negative. 
        # So :delta_y / :delta_x removes the last few rows/columns of pixels.
        img2[:,  :, :delta_y, :delta_x]    = img[:, :, -delta_y:,  -delta_x:]
        gt2[ :,  :, :delta_y2, :delta_x2]  = gt[ :, :, -delta_y2:, -delta_x2:]
        mask[:,  :,  :delta_y,  :delta_x]  = 1
        mask2[:, :, -delta_y:, -delta_x:]  = 1
    if t_img == 0:
        # delta_xy0， delta_xy1: offsets (from old to new flow) for two directions.
        # Take half of delta_x, delta_y as this is the shift for the middle frame.
        # Note the flows are for backward warping (from middle from to 0/1).
        # From 0.5 -> 0: positive delta (from the old flow). old 0.5->0 flow + (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: negative delta (from the old flow). old 0.5->1 flow - (dx, dy) = new 0.5->1 flow.
        delta_xy0 = torch.tensor([ delta_x2,  delta_y2], dtype=float, device=img.device)
        delta_xy1 = torch.tensor([-delta_x2, -delta_y2], dtype=float, device=img.device)
        mask0, mask1 = mask, mask2
    else:
        # From 0.5 -> 0: negative delta (from the old flow). old 0.5->0 flow - (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: positive delta (from the old flow). old 0.5->1 flow + (dx, dy) = new 0.5->1 flow.
        delta_xy0 = torch.tensor([-delta_x2, -delta_y2], dtype=float, device=img.device)
        delta_xy1 = torch.tensor([ delta_x2,  delta_y2], dtype=float, device=img.device)
        mask0, mask1 = mask2, mask

    delta_xy0 = delta_xy0.view(1, 2, 1, 1)
    delta_xy1 = delta_xy1.view(1, 2, 1, 1)
    return img2, gt2, mask0, mask1, delta_xy0, delta_xy1

class Model:
    def __init__(self, local_rank=-1, use_old_model=False, grad_clip=-1, 
                 distill_loss_weight=0.015, 
                 multi=(8,8,4), 
                 ctx_use_merged_flow=False,
                 conv_weight_decay=1e-3,
                 cons_shift_prob=0,
                 cons_shift_sigmas=(14,8)):
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
        self.consist_loss_weight = 0.5
        self.cons_shift_sigmas = cons_shift_sigmas

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
        rand = random.random()
        if self.cons_shift_prob > 0 and rand < self.cons_shift_prob:
            if rand < self.cons_shift_prob / 2:
                # smask: shift mask. dxy: (delta_x, delta_y).
                # *_0: for img0. *_1: for img1.
                img0a, gt2, smask0, smask1, dxy0, dxy1 = random_shift(img0, gt, 0, self.cons_shift_sigmas)
                img1a = img1
            elif rand >= self.cons_shift_prob / 2:
                img1a, gt2, smask0, smask1, dxy0, dxy1 = random_shift(img1, gt, 1, self.cons_shift_sigmas)
                img0a = img0

            if dxy0 is not None:
                imgs2 = torch.cat((img0a, img1a), 1)
                flow2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(torch.cat((imgs2, gt2), 1), scale_list=[4, 2, 1])
                loss_consist_stu = 0
                # s enumerates all scales.
                for s in range(len(flow)):
                    consist_0m = torch.abs(flow[s][:, :2] + dxy0 - flow2[s][:, :2])[smask0].mean()
                    consist_1m = torch.abs(flow[s][:, 2:] + dxy1 - flow2[s][:, 2:])[smask1].mean()
                    loss_consist_stu += (consist_0m + consist_1m) / 2

                consist_0m_tea = torch.abs(flow_teacher[:, :2] + dxy0 - flow_teacher2[:, :2])[smask0].mean()
                consist_1m_tea = torch.abs(flow_teacher[:, 2:] + dxy1 - flow_teacher2[:, 2:])[smask1].mean()
                loss_consist_tea = (consist_0m_tea + consist_1m_tea) / 2
                loss_consist = (loss_consist_stu / len(flow) + loss_consist_tea) / 2
                mean_shift = dxy0.abs().mean().item()
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
