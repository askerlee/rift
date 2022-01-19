import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warp import warp
from model.refine import *
from model.setrans import SETransConfig, SelfAttVisPosTrans, print0
import os
import torch.distributed as dist

local_rank = int(os.environ.get('LOCAL_RANK', 0))

def deconv_gen(do_BN=False):
    if do_BN:
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.Identity

    def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, 
                                            kernel_size=kernel_size, stride=stride, padding=padding),
                    norm_layer(out_planes),
                    nn.PReLU(out_planes)
                )

    return deconv

def conv_gen(do_BN=False):
    if do_BN:
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.Identity

    def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                    norm_layer(out_planes),
                    nn.PReLU(out_planes)
                )

    return conv

def debug():
    if local_rank == 0:
        breakpoint()
    else:
        dist.barrier()

# https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/2
class Clamp01(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# Warp images with multiple groups of flow, and combine them into one group with flow group attention.
def multiwarp(img0, img1, multiflow, multimask_score, M):
    img0_warped_list = []
    img1_warped_list = []
    multimask01_score_list = []
    multimask10_score_list = []
    # multiflow at dim=1: 
    # flow01_1, flow01_2, ..., flow01_M, flow10_1, flow10_2, ..., flow10_M
    # Each block has 2 channels.
    for i in range(M):
        img0_warped = warp(img0, multiflow[:, i*2 : i*2+2])
        img1_warped = warp(img1, multiflow[:, i*2+2*M : i*2+2*M+2])
        img0_warped_list.append(img0_warped)
        img1_warped_list.append(img1_warped)
        # Warp the mask scores. The scores are generated mostly based on
        # unwarped images, and there's misalignment between warped images and unwarped 
        # scores. Therefore, we need to warp the mask scores as well.
        # But doing so only leads to very slight improvement (~0.02 psnr).
        mask01_score_warped = warp(multimask_score[:, [i]], multiflow[:, i*2 : i*2+2])
        mask10_score_warped = warp(multimask_score[:, [i+M]], multiflow[:, i*2+2*M : i*2+2*M+2])
        multimask01_score_list.append(mask01_score_warped)
        multimask10_score_list.append(mask10_score_warped)
    if M == 1:
        return img0_warped_list[0], img1_warped_list[0]

    # img0_warped_list, img1_warped_list are two lists, each of length M.
    # => [16, M, 3, 224, 224]
    warped_img0s = torch.stack(img0_warped_list, dim=1)
    warped_img1s = torch.stack(img1_warped_list, dim=1)
    # multimask_score: 2*M+1 channels. 2*M for M groups of (0->0.5, 1->0.5) flow attention scores, 
    # 1: mask, for the warp0-warp1 combination weight.
    # warp0_attn: [16, M, 1, 224, 224]
    assert multimask_score.shape[1] == 2*M+1
    multimask01_score = torch.stack(multimask01_score_list, dim=1)
    multimask10_score = torch.stack(multimask10_score_list, dim=1)
    warp0_attn = torch.softmax(multimask01_score, dim=1)
    warp1_attn = torch.softmax(multimask10_score, dim=1)
    img0_warped = (warp0_attn * warped_img0s).sum(dim=1)
    img1_warped = (warp1_attn * warped_img1s).sum(dim=1)

    return img0_warped, img1_warped

# Use flow group attention to combine multiple flow groups into one.
def multimerge_flow(multiflow, multimask_score, M):
    if M == 1:
        flow01, flow10 = multiflow[:, :2], multiflow[:, 2:4]
        flow = multiflow
    else:
        # multiflow: [16, 4*M, 224, 224]
        newshape = list(multiflow.shape)
        newshape[1:2] = [M, 2]
        # multiflow01, multiflow10: [16, M, 2, 224, 224]
        multiflow01 = multiflow[:, :M*2].reshape(newshape)
        multiflow10 = multiflow[:, M*2:].reshape(newshape)
        # warp0_attn, warp1_attn: [16, M, 1, 224, 224]
        # multiflow is unwarped, so we don't need to warp the mask scores.
        warp0_attn = torch.softmax(multimask_score[:, :M], dim=1).unsqueeze(dim=2)
        warp1_attn = torch.softmax(multimask_score[:, M:2*M], dim=1).unsqueeze(dim=2)
        # flow01, flow10: [16, 2, 224, 224]
        flow01 = (warp0_attn * multiflow01).sum(dim=1)
        flow10 = (warp1_attn * multiflow10).sum(dim=1)
        flow = torch.cat([flow01, flow10], dim=1)
    return flow, flow01, flow10

class IFBlock(nn.Module):
    def __init__(self, name, in_planes, c=64, img_chans=3, multi=1, do_BN=False, apply_trans=False):
        super(IFBlock, self).__init__()
        self.name = name
        self.apply_trans = apply_trans
        self.img_chans   = img_chans
        # M, multi: How many copies of flow/mask are generated?
        self.M = multi
        if self.M == 1:
            # originally, lastconv outputs 5 channels: 4 flow and 1 mask. upsample by 2x.
            out_chan_num = 5
        else:
            # when outputting multiple flows, 4*M are flow channels, 
            # 2*M flow group attention, 1 mask weight to combine warp0 and warp1.
            out_chan_num = 6 * self.M + 1
        self.lastconv = nn.ConvTranspose2d(c, out_chan_num, 4, 2, 1)

        conv = conv_gen(do_BN=do_BN)

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

        scaled_output = F.interpolate(unscaled_output, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        # tmp/flow/mask_score size = input size.
        # flow has 4 channels. 2 for one direction, 2 for the other direction
        multiflow = scaled_output[:, : 4*self.M] * scale * 2
        # multimask_score: 
        # if M == 1, multimask_score has one channel, just as the original scheme.
        # if M > 1, 2*M+1 channels. 2*M for M groups of (0->0.5, 1->0.5) flow group attention scores, 
        # 1 for the warp0-warp1 combination weight.
        # If M==1, the first two channels are redundant and never used or involved into training.
        multimask_score = scaled_output[:, 4*self.M : ]
        return multiflow, multimask_score
    
class IFNet(nn.Module):
    def __init__(self, use_rife_settings=False, mask_score_res_weight=-1,
                 multi=(1,1,1), do_BN=False, trans_layer_indices=()):
        super(IFNet, self).__init__()
        self.trans_layer_indices = trans_layer_indices
        self.use_rife_settings = use_rife_settings
        if self.use_rife_settings:
            block_widths = [240, 150, 90]
            self.mask_score_res_weight = 1
        else:
            block_widths = [240, 144, 80]
            if mask_score_res_weight >= 0:
                self.mask_score_res_weight = mask_score_res_weight
            else:
                self.mask_score_res_weight = 0

        self.Ms = multi
        self.block0 =    IFBlock('block0',    6,    c=block_widths[0], img_chans=3, 
                                 multi=self.Ms[0],  do_BN=do_BN)
        self.block1 =    IFBlock('block1',    13+4, c=block_widths[1], img_chans=6,  
                                 multi=self.Ms[1],  do_BN=do_BN)
        self.block2 =    IFBlock('block2',    13+4, c=block_widths[2], img_chans=6, 
                                 multi=self.Ms[2],  do_BN=do_BN)
        # block_tea takes gt (the middle frame) as extra input. 
        # block_tea only outputs one group of flow, as it takes extra info and the single group of 
        # output flow is already quite accurate.
        self.block_tea = IFBlock('block_tea', 16+4, c=block_widths[2],  img_chans=6, 
                                 multi=self.Ms[2], do_BN=do_BN)
        self.contextnet = Contextnet()
        # unet: 17 channels of input, 3 channels of output. Output is between 0 and 1.
        self.unet = Unet()
        self.distill_scheme = 'hard' # 'hard' or 'soft'
        # As the distll mask weight is obtained by sigmoid(), even if teacher is worse than student, i.e., 
        # (student - teacher) < 0, the distill mask weight could still be as high as ~0.5. 
        self.distill_soft_min_weight = 0.4  

        # Clamp with gradient works worse. Maybe when a value is clamped, that means it's an outlier?
        self.use_grad_clamp = False
        if self.use_grad_clamp:
            clamp01_inst = Clamp01()
            self.clamp01 = clamp01_inst.apply

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

        multiflow_list = []
        flow_list = []
        warped_imgs_list = []
        merged_img_list  = [None, None, None]
        mask_list = []
        img0_warped = img0
        img1_warped = img1
        multiflow = None 
        loss_distill = 0
        stu_blocks = [self.block0, self.block1, self.block2]
        for i in range(3):
            # scale_list[i]: 1/4, 1/2, 1, i.e., from coarse to fine grained, and from smaller to larger images.
            if multiflow != None:
                if self.use_rife_settings:
                    stu_input = torch.cat((img0, img1, img0_warped, img1_warped, mask_score), 1)
                else:
                    stu_input = torch.cat((img0, img0_warped, img1, img1_warped, mask_score), 1)

                # flow, flow_d: [16, 4*M, 224, 224]
                # mask_score, mask_score_d:   [16, 1*M, 224, 224]
                # flow_d, flow returned from an IFBlock is always of the size of the original image.
                multiflow_d, multimask_score_d = stu_blocks[i](stu_input, flow, scale=scale_list[i])
                if self.Ms[i-1] == self.Ms[i]:
                    multiflow_res       = multiflow
                    multimask_score_res = multimask_score
                else:
                    # If multiflow from the previous iteration has more channels than the current iteration,
                    # then only take the first Ms[i] channels (of each direction) as the residual.
                    # Mp: M of the previous iteration. Mc: M of the current iteration.
                    Mp, Mc = self.Ms[i-1], self.Ms[i]
                    multiflow_res       = torch.cat([ multiflow[:, :2*Mc], multiflow[:, 2*Mp:2*Mp+2*Mc] ], 1)
                    multimask_score_res = torch.cat([ multimask_score[:, :Mc], multimask_score[:, Mp:Mp+Mc], 
                                                      multimask_score[:, [-1]] ], 1)
                multiflow = multiflow_res + multiflow_d
                multimask_score = multimask_score_d + multimask_score_res * self.mask_score_res_weight
            else:
                stu_input = torch.cat((img0, img1), 1)
                multiflow,   multimask_score   = stu_blocks[i](stu_input, None, scale=scale_list[i])

            mask_score = multimask_score[:, [-1]]
            mask_list.append(torch.sigmoid(mask_score))
            multiflow_list.append(multiflow)
            flow, flow01, flow10 = multimerge_flow(multiflow, multimask_score, self.Ms[i])
            flow_list.append(flow)
            img0_warped, img1_warped = \
                multiwarp(img0, img1, multiflow, multimask_score, self.Ms[i])
            warped_imgs = (img0_warped, img1_warped)
            warped_imgs_list.append(warped_imgs)

        if is_training:
            # teacher only works at the last scale, i.e., the full image.
            # block_tea ~ block2, except that block_tea takes gt (the middle frame) as extra input.
            # block_tea input: torch.cat: [1, 13, 256, 448], flow: [1, 4, 256, 448].
            # flow_d: flow difference between the teacher and the student. 
            # (or residual of the teacher). The teacher only predicts the residual.
            if self.use_rife_settings:
                tea_input = torch.cat((img0, img1, img0_warped, img1_warped, mask_score, gt), 1)
            else:
                tea_input = torch.cat((img0, img0_warped, img1, img1_warped, mask_score, gt), 1)    

            multiflow_d, multimask_score_d = self.block_tea(tea_input, flow, scale=1)

            # multiflow and multimask_score are from block2, 
            # which always have the same M as the teacher.
            multiflow_res       = multiflow
            multimask_score_res = multimask_score                      
            multiflow_tea = multiflow_res + multiflow_d
            multimask_score_tea = multimask_score_d + multimask_score_res * self.mask_score_res_weight
            warped_img0_tea, warped_img1_tea = \
                multiwarp(img0, img1, multiflow_tea, multimask_score_tea, self.Ms[2])
            mask_score_tea = multimask_score_tea[:, [-1]]
            mask_tea = torch.sigmoid(mask_score_tea)
            merged_tea = warped_img0_tea * mask_tea + warped_img1_tea * (1 - mask_tea)
            flow_tea, _, _ = multimerge_flow(multiflow_tea, multimask_score_tea, self.Ms[2])
        else:
            flow_tea = None
            merged_tea = None

        for i in range(3):
            # mask_list[i]: *soft* mask (weights) at the i-th scale.
            # merged_img_list[i]: average of 1->2 and 2->1 warped images.
            merged_img_list[i] = warped_imgs_list[i][0] * mask_list[i] + \
                                 warped_imgs_list[i][1] * (1 - mask_list[i])
                                 
            if gt.shape[1] == 3:
                # distil_mask indicates where the warped images according to student's prediction 
                # is worse than that of the teacher.
                student_residual = (merged_img_list[i] - gt).abs().mean(1, True)
                teacher_residual = (merged_tea - gt).abs().mean(1, True)
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
                loss_distill += ((flow_tea.detach() - flow_list[i]).abs() * distil_mask).mean()

        c0 = self.contextnet(img0, flow01)
        c1 = self.contextnet(img1, flow10)
        tmp = self.unet(img0, img1, img0_warped, img1_warped, mask_score, flow, c0, c1)
        # unet output is always within (0, 1). tmp*2-1: within (-1, 1).
        img_residual = tmp[:, :3] * 2 - 1

        if self.use_grad_clamp:
            merged_img_list[2] = self.clamp01(merged_img_list[2] + img_residual)
        else:
            merged_img_list[2] = torch.clamp(merged_img_list[2] + img_residual, 0, 1)

        # flow_list, mask_list: flow and mask in 3 different scales.
        return flow_list, mask_list[2], merged_img_list, flow_tea, merged_tea, loss_distill
