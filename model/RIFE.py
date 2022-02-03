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

# img1 and gt are 4D tensors of (B, 3, H, W). gt are the middle frames.
def random_shift(img, gt):
    B, C, H, W = img.shape
    max_u_shift = W // 16
    max_v_shift = H // 16
    x_shift = np.random.randint(-max_u_shift, max_u_shift)
    y_shift = np.random.randint(-max_v_shift, max_v_shift)
    # Make sure x_shift and y_shift are even numbers.
    x_shift = (x_shift // 2) * 2
    y_shift = (y_shift // 2) * 2
    # Shift gt by half of (y_shift, x_shift).
    x_shift2 = x_shift // 2
    y_shift2 = y_shift // 2

    # Do not bother to make a special case to handle 0 offsets. 
    # Just discard such shift params.
    if x_shift == 0 or y_shift == 0:
        return img, gt, None

    img2    = torch.zeros_like(img)
    gt2     = torch.zeros_like(gt)
        
    if x_shift > 0 and y_shift > 0:
        img2[:, :, y_shift:, x_shift:]    = img[:, :, :-y_shift,  :-x_shift]
        gt2[ :, :, y_shift2:, x_shift2:]  = gt[ :, :, :-y_shift2, :-x_shift2]
    if x_shift > 0 and y_shift < 0:
        img2[:, :, :y_shift,  x_shift:]   = img[:, :, -y_shift:,  :-x_shift]
        gt2[ :, :, :y_shift2, x_shift2:]  = gt[ :, :, -y_shift2:, :-x_shift2]
    if x_shift < 0 and y_shift > 0:
        img2[:, :, y_shift:, :x_shift]    = img[:, :, :-y_shift,  -x_shift:]
        gt2[ :, :, y_shift2:, :x_shift2]  = gt[ :, :, :-y_shift2, -x_shift2:]
    if x_shift < 0 and y_shift < 0:
        img2[:, :, :y_shift, :x_shift]    = img[:, :, -y_shift:,  -x_shift:]
        gt2[ :, :, :y_shift2, :x_shift2]  = gt[ :, :, -y_shift2:, -x_shift2:]

    xy_shift = torch.tensor([x_shift, y_shift], dtype=float, device=img.device)
    xy_shift = xy_shift.view(1, 2, 1, 1)
    return img2, gt2, xy_shift

class Model:
    def __init__(self, local_rank=-1, use_old_model=False, grad_clip=-1, 
                 distill_loss_weight=0.015, 
                 multi=(8,8,4), 
                 ctx_use_merged_flow=False,
                 conv_weight_decay=1e-3,
                 cons_shift_prob=0):
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
        self.consist_loss_weight = 0.1

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
            imgs2, gt2, xy_shift = random_shift(imgs, gt)
            if xy_shift is not None:
                flow2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(torch.cat((imgs2, gt2), 1), scale_list=[4, 2, 1])
                consist_loss_stu = torch.abs(flow2 + xy_shift - flow).mean()
                consist_loss_tea = torch.abs(flow_teacher2 + xy_shift - flow_teacher).mean()
                consist_loss = (consist_loss_stu + consist_loss_tea) / 2
            else:
                consist_loss = 0
        else:
            consist_loss = 0

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
                     + consist_loss * self.consist_loss_weight
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
            }
