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
from model.loss import *
from model.laplacian import *
from model.refine import *
import random
from .augment_consist_loss import *

device = torch.device("cuda")


class Model:
    def __init__(self, local_rank=-1, use_old_model=False, grad_clip=-1, 
                 distill_loss_weight=0.02, 
                 multi=(8,8,4), 
                 weight_decay=1e-3,
                 cons_shift_prob=0,
                 shift_sigmas=(16,10),
                 cons_flip_prob=0,
                 cons_rot_prob=0,
                 consist_loss_weight=0.02,
                 mixed_precision=False,
                 debug=False):
        #if arbitrary == True:
        #    self.flownet = IFNet_m()
        if use_old_model:
            self.flownet = IFNet_rife()
        else:
            self.flownet = IFNet(multi, mixed_precision=mixed_precision)
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
        self.cons_flip_prob = cons_flip_prob
        self.cons_rot_prob = cons_rot_prob
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
        flow_list, mask, merged_img_list, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        stu_pred = merged_img_list[2]
        if TTA == False:
            return stu_pred
        else:
            flow_list2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
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
        flow_list, mask, merged_img_list, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale_list=[4, 2, 1])

        args = dict(img0=img0, img1=img1, gt=gt, flow_list=flow_list, flow_teacher=flow_teacher,
                    model=self.flownet, shift_sigmas=self.shift_sigmas)
        if self.cons_shift_prob > 0 and random.random() < self.cons_shift_prob:
            args["aug_handler"]  = random_shift
            args["flow_handler"] = flow_adder
            loss_consist, loss_distill2, mean_tidbit = calculate_consist_loss(**args)
        elif self.cons_flip_prob > 0 and random.random() < self.cons_flip_prob:
            args["aug_handler"]  = random_flip
            args["flow_handler"] = flow_flipper
            loss_consist, loss_distill2, mean_tidbit = calculate_consist_loss(**args)
        elif self.cons_rot_prob > 0 and random.random() < self.cons_rot_prob:
            args["aug_handler"]  = random_rotate
            args["flow_handler"] = flow_rotator
            loss_consist, loss_distill2, mean_tidbit = calculate_consist_loss(**args)
        else:
            loss_consist = 0
            mean_tidbit = 0
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
            flow_teacher = flow_list[2]

        return stu_pred, {
                'merged_tea': merged_teacher,
                'mask': mask,
                'mask_tea': mask,
                'flow': flow_list[2][:, :2],
                'flow_tea': flow_teacher,
                'loss_stu': loss_stu,
                'loss_tea': loss_tea,
                'loss_distill': loss_distill,
                'loss_consist': loss_consist,
                'mean_tidbit': mean_tidbit
               }
