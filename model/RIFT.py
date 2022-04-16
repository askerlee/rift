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

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class RIFT:
    def __init__(self, local_rank=-1, use_old_model=False, 
                 esti_sofi=False,
                 grad_clip=-1, 
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

        self.esti_sofi = esti_sofi

        if use_old_model:
            self.flownet = IFNet_rife()
        else:
            self.flownet = IFNet(multi, esti_sofi)
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
                               find_unused_parameters=not esti_sofi)
        self.distill_loss_weight = distill_loss_weight
        self.grad_clip = grad_clip
        self.cons_shift_prob = cons_shift_prob
        self.shift_sigmas = shift_sigmas
        self.cons_flip_prob = cons_flip_prob
        self.cons_rot_prob = cons_rot_prob
        self.consist_loss_weight = consist_loss_weight
        # Even if crude_loss_weight=0.01, it still slightly reduces performance.
        self.crude_loss_weight = 0.0   
        self.mixed_precision = mixed_precision

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
        flow_list, mask, crude_img_list, refined_img_list, flow_teacher, \
            merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        stu_pred = refined_img_list[2]
        if TTA == False:
            return stu_pred
        else:
            flow_list2, mask2, crude_img_list2, refined_img_list2, flow_teacher2, \
                merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (stu_pred + refined_img_list2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
            
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        
        with autocast(enabled=self.mixed_precision):
            flow_list, mask, crude_img_list, refined_img_list, flow_teacher, \
                merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale_list=[4, 2, 1])

        # flow_list is of length 4.
        args = dict(model=self.flownet, img0=img0, img1=img1, gt=gt, 
                    flow_list=flow_list, flow_teacher=flow_teacher, num_rift_scales=3,
                    shift_sigmas=self.shift_sigmas, mixed_precision=self.mixed_precision)
        do_consist_loss = True
        if self.cons_shift_prob > 0 and random.random() < self.cons_shift_prob:
            args["aug_handler"]  = random_shift
            args["flow_handler"] = flow_adder
        elif self.cons_flip_prob > 0 and random.random() < self.cons_flip_prob:
            args["aug_handler"]  = random_flip
            args["flow_handler"] = flow_flipper
        elif self.cons_rot_prob > 0 and random.random() < self.cons_rot_prob:
            args["aug_handler"]  = random_rotate
            args["flow_handler"] = flow_rotator
        else:
            loss_consist = 0
            mean_tidbit = 0
            loss_distill2 = 0
            loss_consist_str = '-'
            do_consist_loss = False

        if do_consist_loss:
            loss_consist, loss_distill2, mean_tidbit = calculate_consist_loss(**args)
            loss_consist_str = f"{loss_consist:.2f}/{mean_tidbit}"
            
        only_calc_refined_loss = True
        stu_pred = refined_img_list[2]
        loss_stu = (self.lap(stu_pred, gt)).mean()
        if not only_calc_refined_loss:
            for stu_crude_pred in crude_img_list[:3]:
                # lap: laplacian pyramid loss.
                loss_stu += (self.lap(stu_crude_pred, gt)).mean()
            loss_stu = loss_stu / 4

        if self.esti_sofi:
            refined_img0        = refined_img_list[3]
            refined_img1        = refined_img_list[4]
            crude_img0          = crude_img_list[3]
            crude_img1          = crude_img_list[4]
            loss_refined_img0   = (self.lap(refined_img0, img0)).mean()
            loss_refined_img1   = (self.lap(refined_img1, img1)).mean()

            if self.crude_loss_weight > 0:
                loss_crude_img0     = (self.lap(crude_img0, img0)).mean()
                loss_crude_img1     = (self.lap(crude_img1, img1)).mean()
            else:
                loss_crude_img0, loss_crude_img1 = 0, 0

            # crude_loss_weight = 0.01. loss on crude images is highly inaccurate. So assign a tiny weight.
            loss_sofi           = (loss_refined_img0 + loss_refined_img1 + 
                                   (loss_crude_img0 + loss_crude_img1)* self.crude_loss_weight
                                  ) / 2
        else:
            loss_sofi = torch.tensor(0, device=imgs.device)

        # loss_tea: laplacian pyramid loss between warped image by teacher's flow & the ground truth image
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            CONS_DISTILL_DISCOUNT = 2
            # loss_distill: L1 loss between the teacher's flow and the student's flow.
            # loss_distill2: the distillation loss when the input is shifted. 
            # Discounted by 2, so the effective weight is 0.01.
            loss_G = loss_stu + loss_tea + (loss_distill + loss_distill2 / CONS_DISTILL_DISCOUNT) * self.distill_loss_weight \
                     + loss_consist * self.consist_loss_weight + loss_sofi
            loss_G.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), self.grad_clip)

            self.optimG.step()
        else:
            flow_teacher = flow_list[2]

        return refined_img_list[2], {
                'merged_tea':   merged_teacher,
                'mask':         mask,
                'mask_tea':     mask,
                'flow':         flow_list[2][:, :2],    # :2 means only one direction of the flow is passed.
                'flow_tea':     flow_teacher,
                'flow_sofi':    flow_list[-1],          # Keep both directions of sofi flow.
                # If not esti_sofi, crude_img0, crude_img1, refined_img0, refined_img1 are all None.
                'crude_img0':   crude_img_list[3],
                'crude_img1':   crude_img_list[4],
                'refined_img0': refined_img_list[3],    
                'refined_img1': refined_img_list[4],
                'loss_stu':     loss_stu,
                'loss_tea':     loss_tea,
                'loss_sofi':    loss_sofi,
                'loss_distill': loss_distill,
                'loss_consist_str': loss_consist_str,
                'mean_tidbit':  mean_tidbit
               }

class SOFI_Wrapper(IFNet):
    def __init__(self, multi=(8,8,4)):
        super(SOFI_Wrapper, self).__init__(multi, esti_sofi=True)

    # Simulate the interface of CRAFT.
    def forward(self, image0, image1, iters=12, flow_init=None, upsample=True, test_mode=1):
        # Only support test_mode = 1.
        if test_mode != 1:
            breakpoint()

        scale_list = [4, 2, 1]        
        imgs = torch.cat([image0, image1], dim=1)
        flow_list, mask, crude_img_list, refined_img_list, flow_teacher, \
            merged_teacher, loss_distill = super()(imgs, scale_list, timestep=0.5)

        flow_sofi = flow_list[3]
        flow_01   = flow_sofi[:, 2:4]
        return None, flow_01
