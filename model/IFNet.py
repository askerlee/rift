import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.setrans import SETransConfig, SelfAttVisPosTrans, print0

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, name, in_planes, c=64, img_chans=3, apply_trans=False):
        super(IFBlock, self).__init__()
        self.name = name
        self.apply_trans = apply_trans
        self.img_chans   = img_chans

        if not self.apply_trans:
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 2, 1),
                conv(c//2, c, 3, 2, 1),
                )

        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # lastconv outputs 5 channels: 4 flow and 1 mask
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

        if self.apply_trans:
            self.conv_img = nn.Sequential(
                conv(img_chans, c//2, 3, 2, 1),
                conv(c//2, c, 3, 2, 1),
                )
            self.nonimg_chans = c - 2 * img_chans
            self.conv_bridge = conv(2 * c + self.nonimg_chans, c, 3, 2, 1)

            self.trans_config = SETransConfig()
            self.trans_config.in_feat_dim = c
            self.trans_config.feat_dim  = c
            # f2trans(x) = attn_aggregate(v(x)) + x. Here attn_aggregate and v (first_linear) both have 4 modes.
            self.trans_config.has_input_skip = True
            # No FFN. f2trans simply aggregates similar features.
            self.trans_config.has_FFN = True
            # When doing feature aggregation, set attn_mask_radius > 0 to exclude points that are too far apart, to reduce noises.
            # E.g., 64 corresponds to 64*8=512 pixels in the image space.
            self.trans_config.attn_mask_radius = -1
            # Not tying QK performs slightly better.
            self.trans_config.tie_qk_scheme = None
            self.trans_config.qk_have_bias  = False
            self.trans_config.out_attn_probs_only    = False
            self.trans_config.attn_diag_cycles  = 1000
            self.trans_config.num_modes         = 4
            self.trans_config.pos_code_type     = 'bias'
            self.trans_config.pos_bias_radius   = 7
            self.trans_config.pos_code_weight   = 1.0
            self.trans = SelfAttVisPosTrans(self.trans_config, f"{self.name} transformer")
            print0("trans config:\n{}".format(self.trans_config.__dict__))

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)

        if not self.apply_trans:
            # conv0: 2 layers of conv, kernel size 3.
            x = self.conv0(x)
            # x: [1, 240, 16, 28] in 'block0'.
            #    [1, 150, 32, 56] in 'block1'.
            #    [1, 90, 64, 112] in 'block2'.

            # convblock: 8 layers of conv, kernel size 3.
            # if apply_trans, trans is a transformer layer with a skip connection: 
            # x' = w*trans(conv(x)) + (1-w) * conv(x) + x. w is a learnable weight.
            x = self.convblock(x) + x
        else:
            img0 = x[:, 0:self.img_chans]
            img1 = x[:, self.img_chans:self.img_chans*2]
            nonimg = x[:, self.img_chans*2:]
            x0 = self.conv_img(img0)
            x1 = self.trans(self.conv_img(img1))
            x  = self.conv_bridge(torch.cat((x0, x1, nonimg), 1))

        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        # flow has 4 channels. 2 for one direction, 2 for the other direction
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    
class IFNet(nn.Module):
    def __init__(self, trans_layer_idx=-1):
        super(IFNet, self).__init__()
        self.trans_layer_idx = trans_layer_idx
        self.block0 = IFBlock('block0', 6,          c=240, img_chans=3, 
                              apply_trans=(trans_layer_idx==0))
        self.block1 = IFBlock('block1', 13+4,       c=144, img_chans=6, 
                              apply_trans=(trans_layer_idx==1))
        self.block2 = IFBlock('block2', 13+4,       c=80,  img_chans=6, 
                              apply_trans=(trans_layer_idx==2))
        self.block_tea = IFBlock('block_tea', 16+4, c=80,  img_chans=6, 
                              apply_trans=(trans_layer_idx==2))
        self.contextnet = Contextnet()
        # unet: 17 channels of input, 3 channels of output. Output is between 0 and 1.
        self.unet = Unet()

    # scale_list: the scales to shrink the feature maps. scale_factor = 1. / scale_list[i]
    # For evaluation on benchmark datasets, as only the middle frame is compared,
    # we don't need to consider a flexible timestep here.
    def forward(self, x, scale_list=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        # During inference, gt is an empty tensor.
        gt = x[:, 6:] 
        flow_list = []
        to_merge = []
        merged = [None, None, None]
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            # scale_list[i]: 1/4, 1/2, 1, i.e., from coarse to fine grained.
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, warped_img0, img1, warped_img1, mask), 1), flow, scale=scale_list[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale_list[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            to_merge_student = (warped_img0, warped_img1)
            to_merge.append(to_merge_student)
        
        if gt.shape[1] == 3:
            # teacher only works at the last scale, i.e., the full image.
            # block_tea ~ block2, except that block_tea takes gt as extra input.
            # block_tea input: torch.cat: [1, 13, 256, 448], flow: [1, 4, 256, 448].
            # flow_d: flow difference between the teacher and the student. 
            # (or residual of the teacher)
            flow_d, mask_d = self.block_tea(torch.cat((img0, warped_img0, img1, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            # mask_list[i]: *soft* mask (weights) at the i-th scale.
            # merged[i]: average of 1->2 and 2->1 warped images.
            merged[i] = to_merge[i][0] * mask_list[i] + to_merge[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                # loss_mask indicates where the warped images according to student's prediction 
                # is worse than that of the teacher.
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                # If at some points, the warped image of the teacher is better than the student,
                # then regard the flow at these points are more accurate, and use them to teach the student.
                loss_distill += ((flow_teacher.detach() - flow_list[i]).abs() * loss_mask).mean()
                
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        # unet output is always within (0, 1). tmp*2-1: within (-1, 1).
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        # flow_list, mask_list: flow and mask in 3 different scales.
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
