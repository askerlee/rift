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
                 is_big_model=False,
                 esti_sofi=False,
                 grad_clip=-1, 
                 distill_loss_weight=0.02, 
                 smooth_loss_weight=0.001,
                 sofi_loops=2,
                 multi=(8,8,4), 
                 weight_decay=1e-3,
                 consistency_args={},
                 use_edge_aware_smooth_loss=False,
                 mixed_precision=False,
                 debug=False):

        self.esti_sofi = esti_sofi

        if use_old_model:
            self.flownet = IFNet_rife()
        else:
            self.flownet = IFNet(multi, is_big_model, esti_sofi, sofi_loops)
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
        self.use_edge_aware_smooth_loss = use_edge_aware_smooth_loss    # default: False, no edge awareness.
        self.smooth_loss_weight  = smooth_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.grad_clip = grad_clip
        self.cons_shift_prob    = consistency_args.get("shift_prob", 0.2)
        self.shift_sigmas       = consistency_args.get("shift_sigmas", [24, 16])
        self.cons_flip_prob     = consistency_args.get("flip_prob", 0.1)
        self.cons_rot_prob      = consistency_args.get("rot_prob", 0.1)
        self.cons_jitter_prob   = consistency_args.get("jitter_prob", 0.2)
        self.cons_erase_prob    = consistency_args.get("erase_prob", 0.5)
        self.cons_scale_prob    = consistency_args.get("scale_prob", 0.3)
        self.consist_loss_weight = consistency_args.get("consist_loss_weight", 0.02)
        self.consistency_args   = consistency_args

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
        flow_list, mask, crude_img_list, refined_img_list, teacher_dict, loss_distill \
                = self.flownet(imgs, scale_list, timestep=timestep)
        stu_pred = refined_img_list[2]
        if TTA == False:
            return stu_pred
        else:
            flow_list2, mask2, crude_img_list2, refined_img_list2, teacher_dict, loss_distill2 \
                = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (stu_pred + refined_img_list2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, mid_gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
            
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        
        with autocast(enabled=self.mixed_precision):
            flow_list, mask, crude_img_list, refined_img_list, teacher_dict, loss_distill = \
                    self.flownet(imgs, mid_gt, scale_list=[4, 2, 1])

        flow_teacher    = teacher_dict['flow_teacher']
        merged_teacher  = teacher_dict['merged_teacher']

        num_rift_scales = 3
        # flow_list is of length 4.
        args = dict(model=self.flownet, img0=img0, img1=img1, mid_gt=mid_gt, 
                    flow_list=flow_list, flow_teacher=flow_teacher, num_rift_scales=num_rift_scales,
                    shift_sigmas=self.shift_sigmas, mixed_precision=self.mixed_precision)
        do_consist_loss = True
        # 0.2
        if self.cons_shift_prob > 0 and random.random() < self.cons_shift_prob:
            args["aug_handler"]     = random_shift
            args["aug_type"]        = "shift"
        # 0.8 * 0.1 = 0.08
        elif self.cons_flip_prob > 0 and random.random() < self.cons_flip_prob:
            args["aug_handler"]     = random_flip
            args["aug_type"]        = "flip"
        # 0.72 * 0.1 = 0.072
        elif self.cons_rot_prob > 0 and random.random() < self.cons_rot_prob:
            args["aug_handler"]     = random_rotate
            args["aug_type"]        = "rotate"
        # 0.648 * 0.1 = 0.0648
        elif self.cons_jitter_prob > 0 and random.random() < self.cons_jitter_prob:
            args["aug_handler"]     = color_jitter
            args["aug_type"]        = "jitter"
        # 0.5832 * 0.3 = 0.175
        elif self.cons_erase_prob > 0 and random.random() < self.cons_erase_prob:
            args["aug_handler"]     = random_erase
            args["aug_type"]        = "erase"
        # 0.408 * 0.4 = 0.1632
        elif self.cons_scale_prob > 0 and random.random() < self.cons_scale_prob:
            args["aug_handler"]     = random_scale
            args["aug_type"]        = "scale"
        # 0.2448
        else:
            loss_consist        = 0
            mean_tidbit         = 0
            loss_distill2       = 0
            loss_consist_str    = '-'
            do_consist_loss     = False

        if do_consist_loss:
            loss_consist, loss_distill2, aug_desc = calculate_consist_loss(**args)
            loss_consist_str = f"{loss_consist:.3f}/{aug_desc}"

        only_calc_refined_loss = True
        stu_pred = refined_img_list[2]
        if mid_gt.shape[1] == 3:
            loss_stu = (self.lap(stu_pred, mid_gt)).mean()
            if not only_calc_refined_loss:
                for stu_crude_pred in crude_img_list[:3]:
                    # lap: laplacian pyramid loss.
                    loss_stu += (self.lap(stu_crude_pred, mid_gt)).mean()
                loss_stu = loss_stu / 4
            # loss_tea: laplacian pyramid loss between warped image by teacher's flow & the ground truth image
            loss_tea = (self.lap(merged_teacher, mid_gt)).mean()
        else:
            loss_stu = 0
            loss_tea = 0

        if self.esti_sofi:
            refined_img0        = refined_img_list[3]
            refined_img1        = refined_img_list[4]
            #crude_img0          = crude_img_list[3]
            #crude_img1          = crude_img_list[4]
            loss_refined_img0   = (self.lap(refined_img0, img0)).mean()
            loss_refined_img1   = (self.lap(refined_img1, img1)).mean()

            # loss on crude_img0/crude_img1 is highly inaccurate. So disable it.
            loss_sofi           = (loss_refined_img0 + loss_refined_img1) / 2
        else:
            loss_sofi = torch.tensor(0, device=imgs.device)

        loss_smooth = 0
        for flow in flow_list + [flow_teacher]:
            if flow is not None:
                # by default, use flow_smooth_delta
                if self.use_edge_aware_smooth_loss:
                    curr_smooth_loss = edge_aware_smoothness_order1(img0, img1, flow)
                else:
                    curr_smooth_loss = flow_smooth_delta(flow)
                loss_smooth += curr_smooth_loss

        if training:
            self.optimG.zero_grad()
            CONS_DISTILL_DISCOUNT = 2
            # loss_distill: L1 loss between the teacher's flow and the student's flow.
            # loss_distill2: the distillation loss when the input is randomly augmented. 
            # Discounted by 2, so the effective weight is 0.01.
            loss_G = loss_stu + loss_tea + (loss_distill + loss_distill2 / CONS_DISTILL_DISCOUNT) * self.distill_loss_weight \
                     + loss_consist * self.consist_loss_weight + loss_sofi + loss_smooth * self.smooth_loss_weight
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
                'loss_smooth':  loss_smooth,
                'loss_consist_str': loss_consist_str,
               }

class SOFI_Wrapper(nn.Module):
    def __init__(self, flownet, sofi_mode='dual'):
        super().__init__()
        self.flownet = flownet
        self.flownet.eval()
        self.sofi_mode = sofi_mode # 'LR', 'RL', 'dual', 'mid'

    def load_state_dict(self, checkpoint, strict=False):
        checkpoint2 = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                checkpoint2[k[7:]] = v
            else:
                checkpoint2[k] = v
        msg = self.flownet.load_state_dict(checkpoint2, strict=strict)
        return msg

    # Simulate the interface of CRAFT.
    def forward(self, image0, image1, iters=12, flow_init=None, upsample=True, test_mode=1):
        # Only support test_mode = 1.
        if test_mode != 1:
            breakpoint()

        scale_list = [4, 2, 1]        
        if self.sofi_mode == 'LR':
            imgs_LR = torch.cat([image0, image1], dim=1) / 255.0
            imgs    = imgs_LR
        elif self.sofi_mode == 'RL':
            imgs_RL = torch.cat([image1, image0], dim=1) / 255.0
            imgs    = imgs_RL
        elif self.sofi_mode == 'dual':
            imgs_LR = torch.cat([image0, image1], dim=1) / 255.0
            imgs_RL = torch.cat([image1, image0], dim=1) / 255.0
            # Put imgs_LR and imgs_RL in the same batch to avoid two different batches.
            imgs    = torch.cat([imgs_LR, imgs_RL], dim=0)

        # Provide an empty tensor as mid_gt, just to make the model happy.
        mid_gt  = imgs[:, :0]           
        flow_list, mask, crude_img_list, refined_img_list, teacher_dict, loss_distill \
                = self.flownet(imgs, mid_gt, scale_list)

        flow_sofi = flow_list[3]
        # flow_list[2] * 2 is the mid flow, which is around 0.1 AEPE worse than flow_sofi.
        #flow_sofi = flow_list[2] * 2
        flow_01   = flow_sofi[:, 2:4]
        flow_10   = flow_sofi[:, 0:2]
        if self.sofi_mode == 'LR':
            flow = flow_01
        elif self.sofi_mode == 'RL':
            flow = flow_10
        elif self.sofi_mode == 'dual':
            # flow_01[:2]: 0->1 flow of images_LR, 
            # flow_10[2:]: 1->0 flow of images_RL. Still 0->1 flow, but estimated with the reverse image order.
            # Take the average of the two directions.
            flow = (flow_01[0] + flow_10[1]) / 2

        return None, flow
