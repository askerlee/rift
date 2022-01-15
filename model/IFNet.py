import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.setrans import SETransConfig, SelfAttVisPosTrans, print0
import os
import torch.distributed as dist

local_rank = int(os.environ.get('LOCAL_RANK', 0))

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

def debug():
    if local_rank == 0:
        breakpoint()
    else:
        dist.barrier()

class IFBlock(nn.Module):
    def __init__(self, name, in_planes, c=64, img_chans=3, apply_trans=False):
        super(IFBlock, self).__init__()
        self.name = name
        self.apply_trans = apply_trans
        self.img_chans   = img_chans

        # lastconv outputs 5 channels: 4 flow and 1 mask. upsample by 2x.
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

        if not self.apply_trans:
            # downsample by 4x.
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
        else:
            self.nonimg_chans = in_planes - 2 * self.img_chans
            self.conv_img = nn.Sequential(
                                conv(self.img_chans, c//2, 3, 2, 1),
                                conv(c//2, c//2, 3, 2, 1),
                                conv(c//2, c//2),
                                conv(c//2, c//2),
                                conv(c//2, c//2),                
                            )
            if self.nonimg_chans > 0:
                # nonimg: mask + flow computed in the previous scale (only available for block1 and block2)
                self.conv_nonimg = nn.Sequential(
                                        conv(self.nonimg_chans, c//2, 3, 2, 1),
                                        conv(c//2, c//2, 3, 2, 1)
                                   )
                self.conv_bridge = conv(3 * (c//2), c, 3, 1, 1)
            else:
                # No non-img channels. Just to bridge the channel number difference.
                self.conv_bridge = conv(c, c, 3, 1, 1)

            # Moved 3 conv layers from convblock to conv_img and conv_nonimg.
            self.convblock = nn.Sequential(
                                conv(c, c),
                                conv(c, c),
                                conv(c, c),
                                conv(c, c),
                                conv(c, c),
                             )

            self.trans_config = SETransConfig()
            self.trans_config.in_feat_dim = c//2
            self.trans_config.feat_dim    = c//2
            # f2trans(x) = attn_aggregate(v(x)) + x. Here attn_aggregate and v (first_linear) both have 4 modes.
            self.trans_config.has_input_skip = True
            # No FFN. f2trans simply aggregates similar features.
            # has_FFN reduces performance.
            self.trans_config.has_FFN = False
            # When doing feature aggregation, set attn_mask_radius > 0 to exclude points that are too far apart, to reduce noises.
            # E.g., 64 corresponds to 64*8=512 pixels in the image space.
            self.trans_config.attn_mask_radius = -1
            # Not tying QK performs slightly better.
            self.trans_config.tie_qk_scheme = 'none'
            self.trans_config.qk_have_bias  = False
            self.trans_config.out_attn_probs_only   = False
            self.trans_config.attn_diag_cycles  = 1000
            self.trans_config.num_modes         = 4
            self.trans_config.pos_code_type     = 'bias'
            self.trans_config.pos_bias_radius   = 7
            self.trans_config.pos_code_weight   = 1.0
            self.trans = SelfAttVisPosTrans(self.trans_config, f"{self.name} transformer")
            print0("trans config:\n{}".format(self.trans_config.__dict__))

    def forward(self, x, flow, scale):
        # resize x to input size / scale.
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            # the size and magnitudes of the flow is scaled to the size of this layer. 
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)

        if not self.apply_trans:
            # conv0: 2 layers of conv, kernel size 3.
            x = self.conv0(x)
            # x: [1, 240, 14, 14] in 'block0'.
            #    [1, 150, 28, 28] in 'block1'.
            #    [1, 90,  56, 56] in 'block2'.
            # That is, input size / scale / 4. 
        else:
            img0 = x[:, 0:self.img_chans]
            img1 = x[:, self.img_chans:self.img_chans*2]
            nonimg = x[:, self.img_chans*2:]
            x0 = self.conv_img(img0)
            # if apply_trans, trans is a transformer layer with a skip connection: 
            # x' = w*trans(conv_img(img1)) + (1-w) * conv_img(img1). w is a learnable weight.
            x1 = self.trans(self.conv_img(img1))
            if self.nonimg_chans > 0:
                x_nonimg = self.conv_nonimg(nonimg)
                x  = self.conv_bridge(torch.cat((x0, x1, x_nonimg), 1))
            else:
                x  = self.conv_bridge(torch.cat((x0, x1), 1))
                
        # convblock: 8 layers of conv, kernel size 3.
        x = self.convblock(x) + x
        # unscaled_output size = input size / scale / 2.
        unscaled_output = self.lastconv(x)
        unscaled_flow = unscaled_output[:, :4]
        unscaled_mask_score = unscaled_output[:, 4:5]
        
        scaled_output = F.interpolate(unscaled_output, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        # tmp/flow/mask_score size = input size.
        # flow has 4 channels. 2 for one direction, 2 for the other direction
        flow = scaled_output[:, :4] * scale * 2
        mask_score = scaled_output[:, 4:5]
        return flow, mask_score, unscaled_flow, unscaled_mask_score
    
class IFNet(nn.Module):
    def __init__(self, use_rife_settings=False, trans_layer_indices=()):
        super(IFNet, self).__init__()
        self.trans_layer_indices = trans_layer_indices
        self.use_rife_settings = use_rife_settings
        if self.use_rife_settings:
            block_widths = [240, 150, 90]
            self.mask_score_res_weight = 1
        else:
            block_widths = [240, 144, 80]
            self.mask_score_res_weight = 0

        self.block0 = IFBlock('block0', 6,          c=block_widths[0], img_chans=3, 
                              apply_trans=(0 in trans_layer_indices))
        self.block1 = IFBlock('block1', 13+4,       c=block_widths[1], img_chans=6, 
                              apply_trans=(1 in trans_layer_indices))
        self.block2 = IFBlock('block2', 13+4,       c=block_widths[2],  img_chans=6, 
                              apply_trans=(2 in trans_layer_indices))
        # block_tea takes gt (the middle frame) as extra input.
        self.block_tea = IFBlock('block_tea', 16+4, c=block_widths[2],  img_chans=6, 
                              apply_trans=(3 in trans_layer_indices))
        self.contextnet = Contextnet()
        # unet: 17 channels of input, 3 channels of output. Output is between 0 and 1.
        self.unet = Unet()
        self.distill_scheme = 'soft' # 'hard' or 'soft'
        # As the distll mask weight is obtained by sigmoid(), even if teacher is worse than student, i.e., 
        # (student - teacher) < 0, the distill mask weight could still be as high as ~0.5. 
        self.distill_soft_min_weight = 0.4  
        

    # scale_list: the scales to shrink the feature maps. scale_factor = 1. / scale_list[i]
    # For evaluation on benchmark datasets, as only the middle frame is compared,
    # we don't need to consider a flexible timestep here.
    def forward(self, x, scale_list=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        # During inference, gt is an empty tensor.
        gt = x[:, 6:] 
        # gt is provided, i.e., in the training stage.
        is_training = gt.shape[1] == 3

        flow_list = []
        warped_imgs_list = []
        merged_img_list = [None, None, None]
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu_blocks = [self.block0, self.block1, self.block2]
        for i in range(3):
            # scale_list[i]: 1/4, 1/2, 1, i.e., from coarse to fine grained, and from small to large images.
            if flow != None:
                if self.use_rife_settings:
                    block_input = torch.cat((img0, img1, warped_img0, warped_img1, mask_score), 1)
                else:
                    block_input = torch.cat((img0, warped_img0, img1, warped_img1, mask_score), 1)

                # flow_d, flow returned from an IFBlock is always of the size of the original image.
                flow_d, mask_score_d, unscaled_flow_d, unscaled_mask_score_d = \
                    stu_blocks[i](block_input, flow, scale=scale_list[i])
                flow = flow + flow_d
                mask_score = mask_score_d + mask_score * self.mask_score_res_weight
            else:
                flow,  mask_score, unscaled_flow_d, unscaled_mask_score_d = \
                    stu_blocks[i](torch.cat((img0, img1), 1), None, scale=scale_list[i])
            mask_list.append(torch.sigmoid(mask_score))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            warped_imgs = (warped_img0, warped_img1)
            warped_imgs_list.append(warped_imgs)
        
        if is_training:
            # teacher only works at the last scale, i.e., the full image.
            # block_tea ~ block2, except that block_tea takes gt (the middle frame) as extra input.
            # block_tea input: torch.cat: [1, 13, 256, 448], flow: [1, 4, 256, 448].
            # flow_d: flow difference between the teacher and the student. 
            # (or residual of the teacher). The teacher only predicts the residual.
            flow_d, mask_score_d, unscaled_flow_d, unscaled_mask_score_d = \
                self.block_tea(torch.cat((img0, warped_img0, img1, warped_img1, mask_score, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_score = mask_score_d + mask_score * self.mask_score_res_weight
            mask_teacher = torch.sigmoid(mask_score)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            # mask_list[i]: *soft* mask (weights) at the i-th scale.
            # merged_img_list[i]: average of 1->2 and 2->1 warped images.
            merged_img_list[i] = warped_imgs_list[i][0] * mask_list[i] + \
                                 warped_imgs_list[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                # distil_mask indicates where the warped images according to student's prediction 
                # is worse than that of the teacher.
                student_residual = (merged_img_list[i] - gt).abs().mean(1, True)
                teacher_residual = (merged_teacher - gt).abs().mean(1, True)
                if self.distill_scheme == 'hard':
                    distil_mask = (student_residual > teacher_residual + 0.01).float().detach()
                else:
                    distil_mask = (student_residual - teacher_residual).sigmoid().detach()
                    # / (1 - self.distill_soft_min_weight) to normalize distil_mask to be within (0, 1).
                    # * 2 to make the mean of distil_mask to be 1, same as the 'hard' scheme.
                    distil_mask = 2 * (distil_mask - self.distill_soft_min_weight).clamp(min=0) / (1 - self.distill_soft_min_weight)

                # If at some points, the warped image of the teacher is better than the student,
                # then regard the flow at these points are more accurate, and use them to teach the student.
                # loss_distill is the sum of the distillation losses at 3 different scales.
                loss_distill += ((flow_teacher.detach() - flow_list[i]).abs() * distil_mask).mean()
                
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask_score, flow, c0, c1)
        # unet output is always within (0, 1). tmp*2-1: within (-1, 1).
        img_residual = tmp[:, :3] * 2 - 1
        merged_img_list[2] = torch.clamp(merged_img_list[2] + img_residual, 0, 1)
        # flow_list, mask_list: flow and mask in 3 different scales.
        return flow_list, mask_list[2], merged_img_list, flow_teacher, merged_teacher, loss_distill
