import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=False, lr=1e-6, grad_clip=-1, 
                 distill_loss_weight=0.01, use_rife_settings=False, 
                 mask_score_res_weight=-1,
                 trans_layer_indices=(), trans_weight_decay=1e-5):
        if arbitrary == True:
            self.flownet = IFNet_m(use_rife_settings, mask_score_res_weight, trans_layer_indices)
        else:
            self.flownet = IFNet(use_rife_settings, mask_score_res_weight, trans_layer_indices)
        self.device()

        conv_param_groups, trans_param_groups = [], []
        for name, param in self.flownet.named_parameters():
            if 'trans' in name:
                trans_param_groups.append(param)
            else:
                conv_param_groups.append(param)

        # Use a large weight decay may avoid NaN loss, but reduces transformer performance.
        # lr here doesn't really matter. Will be overwritten in update(), 
        # where the actual LR is obtained from train.py:get_learning_rate().
        self.optimG = AdamW( [ { 'params': conv_param_groups,  'lr': lr, 'weight_decay': 1e-3 }, 
                               { 'params': trans_param_groups, 'lr': lr, 'weight_decay': trans_weight_decay } 
                             ] )

        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], 
                               output_device=local_rank,
                               find_unused_parameters=True)
        self.distill_loss_weight = distill_loss_weight
        self.grad_clip = grad_clip

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
        stu_pred = merged_img_list[2]
        # lap: laplacian pyramid loss.
        loss_stu = (self.lap(stu_pred, gt)).mean()
        # loss_tea: laplacian pyramid loss between warped image by teacher's flow & the ground truth image
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            # loss_distill: L1 loss between the teacher's flow and the student's flow.
            loss_G = loss_stu + loss_tea + loss_distill * self.distill_loss_weight
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
