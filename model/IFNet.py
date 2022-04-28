import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warp import multiwarp, multimerge_flow
from model.refine import *
from model.setrans import SETransConfig, SelfAttVisPosTrans, print0
import os
import torch.distributed as dist
from model.laplacian import LapLoss
import functools

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

# Dual teaching helps slightly.
def dual_teaching_loss(mid_gt, img_stu, flow_stu, img_tea, flow_tea):
    loss_distill = 0
    # Ws[0]: weight of teacher -> student.
    # Ws[1]: weight of student -> teacher.
    # Two directions could take different weights.
    # Set Ws[1] to 0 to disable student -> teacher.
    Ws = [1, 0.5]
    use_lap_loss = False
    # Laplacian loss performs better in earlier epochs, but worse in later epochs.
    # Moreover, Laplacian loss is significantly slower.
    if use_lap_loss:
        loss_fun = LapLoss(max_levels=3, reduction='none')
    else:
        loss_fun = nn.L1Loss(reduction='none')

    for i in range(2):
        student_error = loss_fun(img_stu, mid_gt).mean(1, True)
        teacher_error = loss_fun(img_tea, mid_gt).mean(1, True)
        # distill_mask indicates where the warped images according to student's prediction 
        # is worse than that of the teacher.
        # If at some points, the warped image of the teacher is better than the student,
        # then regard the flow at these points are more accurate, and use them to teach the student.
        distill_mask = (student_error > teacher_error + 0.01).float().detach()

        # loss_distill is the sum of the distillation losses at 2 directions.
        loss_distill += Ws[i] * ((flow_tea.detach() - flow_stu).abs() * distill_mask).mean()

        # Swap student and teacher, and calculate the distillation loss again.
        img_stu, flow_stu, img_tea, flow_tea = \
            img_tea, flow_tea, img_stu, flow_stu
        # The distillation loss from the student to the teacher is given a smaller weight.

    # loss_distill = loss_distill / 2    
    return loss_distill

# https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/2
class Clamp01(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class IFBlock(nn.Module):
    # If do_BN=True, batchnorm is inserted into each conv layer. But it reduces performance. So disabled.
    def __init__(self, name, c, img_chans, nonimg_chans, multi):
        super(IFBlock, self).__init__()
        self.name = name
        # Each image: concat of (original image, warped image).
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

        conv = conv_gen(do_BN=False)

        # At first scale, flow is absent. So nonimg_chans = 1.
        # At other scales, nonimg_chans = 5: 1 channel of global_mask & 4 channels of flow.
        self.nonimg_chans = nonimg_chans   

        # conv_img downscales input image to 1/4 size.
        self.conv_img = nn.Sequential(
                            conv(self.img_chans, c//2, 3, 2, 1),
                            conv(c//2, c//2, 3, 2, 1),
                            conv(c//2, c//2),
                            conv(c//2, c//2),
                            conv(c//2, c//2),                
                        )

        # nonimg: mask + flow computed in the previous scale.
        # In scale 1, they are initialized as 0.
        self.conv_nonimg = nn.Sequential(
                                conv(self.nonimg_chans, c//2, 3, 2, 1),
                                conv(c//2, c//2, 3, 2, 1)
                            )

        # No non-img channels. Just to bridge the channel number difference.
        self.conv_bridge = conv(3 * (c//2), c, 3, 1, 1)

        # Moved 3 conv layers from mixconv in RIFE to conv_img.
        self.mixconv = nn.Sequential(
                            conv(c, c),
                            conv(c, c),
                            conv(c, c),
                            conv(c, c),
                            conv(c, c),
                        )

    def forward(self, imgs, nonimg, flow, scale):
        # Downscale img0/img1 by scale.
        imgs   = F.interpolate(imgs,   scale_factor = 1. / scale, recompute_scale_factor=False, 
                               mode="bilinear", align_corners=False)
        nonimg = F.interpolate(nonimg, scale_factor = 1. / scale, recompute_scale_factor=False, 
                               mode="bilinear", align_corners=False)
        if flow is not None:
            # the size and magnitudes of the flow is scaled to the size of this layer. 
            # Values in flow needs to be scaled as well. So flow and nonimg are treated separately.
            flow   = F.interpolate(flow,   scale_factor = 1. / scale, recompute_scale_factor=False, 
                                   mode="bilinear", align_corners=False) * 1. / scale
            nonimg = torch.cat([nonimg, flow], dim=1)

        # Pack the channels of the two images into the batch dimension,
        # so that the channel number is the same as one image.
        # This makes the feature extraction of the two images slightly faster (hopefully).
        imgs_bpack_shape = list(imgs.shape)
        imgs_bpack_shape[0:2] = [ -1, self.img_chans ]
        imgs_bpack = imgs.reshape(imgs_bpack_shape)
        xs_feat = self.conv_img(imgs_bpack)
        # Unpack the two images in the batch dimension into the channel dimension.
        xs_bunpack_shape = list(xs_feat.shape)
        xs_bunpack_shape[0:2] = [ imgs.shape[0], -1 ]
        xs_feat = xs_feat.reshape(xs_bunpack_shape)

        nonimg_feat = self.conv_nonimg(nonimg)
        x  = self.conv_bridge(torch.cat((xs_feat, nonimg_feat), 1))

        # x: [1, 240, 14, 14] in 'block0'.
        #    [1, 150, 28, 28] in 'block1'.
        #    [1, 90,  56, 56] in 'block2'.
        # That is, input size / scale / 4. 
        # mixconv: 5 layers of conv, kernel size 3. 
        # mixconv mixes the features of images, nonimg, and flow.
        x = self.mixconv(x) + x
        # unscaled_output size = input size / scale / 2.
        unscaled_output = self.lastconv(x)

        scaled_output = F.interpolate(unscaled_output, scale_factor = scale * 2, recompute_scale_factor=False, 
                                      mode="bilinear", align_corners=False)
        # multiflow/multimask_score: same size as original images.
        # each group of flow has 4 channels. 2 for one direction, 2 for the other direction
        # multiflow has 4*M channels.
        multiflow = scaled_output[:, : 4*self.M] * scale * 2
        # multimask_score: 
        # if M == 1, multimask_score has one channel, just as the original scheme.
        # if M > 1, 2*M+1 channels. 2*M for M groups of (0.5->0, 0.5->1) flow group attention scores, 
        # 1 for the warp0-warp1 combination weight.
        # If M==1, the first two channels are redundant and never used or involved into training.
        multimask_score = scaled_output[:, 4*self.M : ]

        return multiflow, multimask_score
    
# Incorporate SOFI into RIFT.
# SOFI: Self-supervised optical flow through video frame interpolation.    
class IFNet(nn.Module):
    def __init__(self, multi=(8,8,4), is_big_model=False, esti_sofi=False, num_sofi_loops=2):
        super(IFNet, self).__init__()

        if is_big_model:
            block_widths = [240, 160, 120]
        else:
            block_widths = [240, 144, 80]
            
        self.Ms = multi
        self.block0     =   IFBlock('block0',     c=block_widths[0], img_chans=3, nonimg_chans=5, 
                                    multi=self.Ms[0])
        self.block1     =   IFBlock('block1',     c=block_widths[1], img_chans=6, nonimg_chans=5, 
                                    multi=self.Ms[1])
        self.block2     =   IFBlock('block2',     c=block_widths[2], img_chans=6, nonimg_chans=5,
                                    multi=self.Ms[2])
        # block_tea takes mid_gt (the middle frame) as extra input. 
        self.block_tea  =   IFBlock('block_tea',  c=block_widths[2], img_chans=6, nonimg_chans=8,
                                    multi=self.Ms[2])
        
        self.contextnet = Contextnet()
        # unet: 17 channels of input, 3 channels of output. Output is between 0 and 1.
        self.unet = Unet()
        self.esti_sofi = esti_sofi
        if self.esti_sofi:
            self.block_sofi = IFBlock('block_sofi', c=block_widths[2], img_chans=6, nonimg_chans=5,
                                      multi=self.Ms[2])
            self.sofi_unet0 = SOFI_Unet()
            self.sofi_unet1 = SOFI_Unet()
            self.stopgrad_prob = 0
            self.num_sofi_loops = num_sofi_loops
        else:
            self.num_sofi_loops = 0

        # Clamp with gradient works worse. Maybe when a value is clamped, that means it's an outlier?
        self.use_clamp_with_grad = False
        if self.use_clamp_with_grad:
            clamp01_inst = Clamp01()
            self.clamp = clamp01_inst.apply
        else:
            self.clamp = functools.partial(torch.clamp, min=0, max=1)
            
    # scale_list: the scales to shrink the feature maps. scale_factor = 1. / scale_list[i]
    # For evaluation on benchmark datasets, as only the middle frame is compared,
    # we don't need to consider a flexible timestep here.
    def forward(self, imgs, mid_gt, scale_list=[4,2,1], timestep=0.5):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:6]
        # During inference, mid_gt is an empty tensor.
        # If mid_gt is provided (in the training stage), then do distillation.
        do_distillation = (mid_gt.shape[1] == 3)

        img0_warped, img1_warped = None, None

        stu_blocks = [self.block0, self.block1, self.block2]
        loss_distill = 0

        # 3 scales of interpolation flow. 
        flow_list = [None, None, None]
        # 2 loops of sofi flow.
        sofi_flow_list = [ None for i in range(self.num_sofi_loops) ]
        # 3 scales of backwarped middle frame (each scale has two images of two directions).
        warped_imgs_list = []
        # 3 scales of crude middle frame (two directions are merged to one image) + warped img0 + warped img1.
        crude_img_list   = [None, None, None, None, None]
        # 3 scales of estimated middle frame + reconstructed img0 + reconstructed img1
        refined_img_list = [None, None, None, None, None]
        mask_list = []

        for i in range(3):
            if i == 0:
                imgs = torch.cat((img0, img1), 1)
                global_mask_score = torch.zeros_like(img0[:, [0]])
                flow_shape = list(img0.shape)
                flow_shape[1] = 4
                flow = torch.zeros(flow_shape, device=img0.device)
                # flow = None
            else:
                # scale_list[i]: 1/4, 1/2, 1, i.e., from coarse to fine grained, and from smaller to larger images.
                imgs = torch.cat((img0, img0_warped, img1, img1_warped), 1)
            # flow: merged flow from multiflow of the previous iteration.
            # multiflow_d: delta multiflow. 
            # multimask_score_d: score for delta multiflow (not delta score; score has no skip connection.)
            # multiflow, multiflow_d: [16, 4*M, 224, 224]
            # multimask_score, multimask_score_d:   [16, 2*M+1, 224, 224]
            # multiflow, multimask_score returned from an IFBlock is always of the size of the original image.
            multiflow_d, multimask_score_d = stu_blocks[i](imgs, global_mask_score, flow, scale=scale_list[i])
            
            if i == 0:
                multiflow_skip = 0
            else:
                Mp = self.Ms[i-1]
                Mc = self.Ms[i]
                if Mp == Mc:
                    multiflow_skip  = multiflow
                elif Mp > Mc:
                    # If multiflow from the previous iteration has more channels than the current iteration.
                    # Mp: M of the previous iteration. Mc: M of the current iteration.
                    # Only take the first Ms[i] channels (of each direction) as the residual.
                    multiflow_skip  = torch.cat([ multiflow[:, :2*Mc], multiflow[:, 2*Mp:2*Mp+2*Mc] ], 1)
                # Mp < Mc. Shouldn't happen.
                else:
                    debug()

            multiflow       = multiflow_skip + multiflow_d
            # multimask_score of different layers have little correlations. 
            # No need to have residual connections.
            multimask_score = multimask_score_d
            global_mask_score = multimask_score[:, [-1]]
            mask_list.append(torch.sigmoid(global_mask_score))

            # flow: single bi-flow merged from multiflow.
            flow = multimerge_flow(multiflow, multimask_score, self.Ms[i])
            flow_list[i] = flow
            img0_warped, img1_warped = \
                multiwarp(img0, img1, multiflow, multimask_score, self.Ms[i])
            warped_imgs = (img0_warped, img1_warped)
            warped_imgs_list.append(warped_imgs)

        if do_distillation:
            # multiflow and multimask_score are from block2, 
            # which always have the same M as the teacher.
            multiflow_skip       = multiflow

            # teacher only works at the last scale, i.e., the full image.
            # block_tea ~ block2, except that block_tea takes mid_gt (the middle frame) as extra input.
            # block_tea input: torch.cat: [1, 13, 256, 448], flow: [1, 4, 256, 448].
            # multiflow_d / multimask_score_d: flow / mask score difference 
            # between the teacher and the student (or residual of the teacher). 
            # The teacher only predicts the residual.
            imgs   = torch.cat((img0, img0_warped, img1, img1_warped), 1)
            nonimg = torch.cat((global_mask_score, mid_gt), 1)
            multiflow_tea_d, multimask_score_tea = self.block_tea(imgs, nonimg, flow, scale=1)

            # Removing this residual connection makes the teacher perform much worse.                      
            multiflow_tea = multiflow_skip + multiflow_tea_d
            img0_warped_tea, img1_warped_tea = \
                multiwarp(img0, img1, multiflow_tea, multimask_score_tea, self.Ms[2])
            global_mask_score_tea = multimask_score_tea[:, [-1]]
            mask_tea = torch.sigmoid(global_mask_score_tea)
            merged_tea = img0_warped_tea * mask_tea + img1_warped_tea * (1 - mask_tea)
            flow_tea = multimerge_flow(multiflow_tea, multimask_score_tea, self.Ms[2])
        else:
            flow_tea   = None
            merged_tea = None

        for i in range(3):
            # mask_list[i]: *soft* mask (weights) at the i-th scale.
            # crude_img_list[i]: average of 0.5->0 and 0.5->1 warped images.
            crude_img_list[i] = warped_imgs_list[i][0] * mask_list[i] + \
                                warped_imgs_list[i][1] * (1 - mask_list[i])
                                
            if do_distillation:
                # dual_teaching_loss: the student can also teach the teacher, 
                # when the student is more accurate.
                # Distilling both merged flow and global mask score leads to slightly worse performance.
                loss_distill += dual_teaching_loss(mid_gt, 
                                                   crude_img_list[i], flow_list[i], 
                                                   merged_tea,        flow_tea,     
                                                  )

        M = self.Ms[-1]
        # multiflow0, multiflow1: first/second half of multiflow.
        # multimask_score0, multimask_score1: first/second half of multimask_score (except the global score).
        multiflow0, multiflow1              = multiflow[:, :2*M],     multiflow[:, 2*M:4*M]
        multimask_score0, multimask_score1  = multimask_score[:, :M], multimask_score[:, M:2*M]

        if self.esti_sofi:
            # Ms: multi, i.e., different scales in each layer.
            # First use 2*(middle->0, middle->1) to approximate the flow (1->0, 0->1).
            # db: double (flow)
            multiflow_sofi = multiflow * 2
            flow_sofi      = flow *2
            global_mask_score_sofi = global_mask_score
            img0_warped_sofi, img1_warped_sofi = \
                multiwarp(img0, img1, multiflow_sofi, multimask_score, self.Ms[-1])
            for k in range(self.num_sofi_loops):
                imgs = torch.cat((img0, img0_warped_sofi, img1, img1_warped_sofi), 1)
                # multiflow_sofi_d: flow delta between multiflow_sofi and 2*(middle flow).
                # the last channel of multimask_score_sofi is global mask weight to blend two directions, 
                # which is not used in SOFI.
                multiflow_sofi_d, multimask_score_sofi = self.block_sofi(imgs, global_mask_score_sofi, flow_sofi, scale=scale_list[0])
                # multiflow_sofi: refined flow (1->0, 0->1).
                # stopgrad helps during early stages, but hurts during later stages. 
                # Therefore make it stochastic with a small prob (default 0.3).
                if k == 0 and (self.stopgrad_prob > 0 and torch.rand(1) < self.stopgrad_prob) \
                  or k > 0:
                    multiflow_sofi = multiflow_sofi_d + multiflow_sofi.data
                else:
                    multiflow_sofi = multiflow_sofi_d + multiflow_sofi
                flow_sofi = multimerge_flow(multiflow_sofi, multimask_score_sofi, M)     
                global_mask_score_sofi = multimask_score_sofi[:, [-1]]
                img0_warped_sofi, img1_warped_sofi = multiwarp(img0, img1, multiflow_sofi, multimask_score_sofi, self.Ms[-1])
                sofi_flow_list.append(flow_sofi)

            multiflow10_sofi,       multiflow01_sofi        = multiflow_sofi[:, :2*M],      multiflow_sofi[:, 2*M:4*M]
            multimask_score10_sofi, multimask_score01_sofi  = multimask_score_sofi[:, :M],  multimask_score_sofi[:, M:2*M]
            # flow_sofi: single bi-flow merged from multiflow_sofi.
        else:
            multiflow_sofi,       multiflow10_sofi,       multiflow01_sofi        = None, None, None
            multimask_score_sofi, multimask_score10_sofi, multimask_score01_sofi  = None, None, None

        # contextnet generates warped features of the input image. 
        # multimask_score* is used in multiwarp, i.e., first warp features according to multiflow*, 
        # then combine with multimask_score*.
        # ctx0, ctx1: four level conv features of img0 and img1, gradually scaled down. 
        # If esti_sofi, ctx0_sofi, ctx1_sofi are contextual features warped 
        # by multiflow10_sofi and multiflow01_sofi, respectively.
        # Otherwise, multiflow10_sofi, multimask_score10_sofi are None,
        # and accordingly, ctx0_sofi, ctx1_sofi are None.
        ctx0, ctx0_sofi = self.contextnet(img0, M, multiflow0, multimask_score0, 
                                          multiflow10_sofi, multimask_score10_sofi)
        ctx1, ctx1_sofi = self.contextnet(img1, M, multiflow1, multimask_score1, 
                                          multiflow01_sofi, multimask_score01_sofi)

        # unet is to refine the crude image crude_img_list[2] with its output img_residual.
        # flow: merged flow (of two directions) from multiflow computed in the last iteration.
        img_residual = self.unet(img0, img1, img0_warped, img1_warped, global_mask_score, flow, ctx0, ctx1)
        # unet activation function changes from softmax to tanh. No need to scale anymore.
        refined_img = self.clamp(crude_img_list[2] + img_residual)
        # refined_img_list[0~1] are always None, to make the indices consistent with crude_img_list.
        refined_img_list[2] = refined_img

        if self.esti_sofi:        
            # img0_warped_sofi is a crude version of img1, and is refined with img1_residual.
            # img1_warped_sofi is a crude version of img0, and is refined with img0_residual.
            flow10, flow01 = flow_sofi.split(2, dim=1)
            # flow01_warp: flow01 aligned to image1.
            flow01_align1 = warp(flow01, flow10)
            # flow10_warp: flow10 aligned to image0.
            flow10_align0 = warp(flow10, flow01)
            # flow_sofi extended with flow01 aligned to image1.
            flow_sofi_01a1 = torch.cat((flow_sofi, flow01_align1), dim=1)
            # flow_sofi extended with flow10 aligned to image0.
            flow_sofi_10a0 = torch.cat((flow_sofi, flow10_align0), dim=1)
            flow_sofi_warp = torch.cat((flow_sofi, flow10_align0, flow01_align1), dim=1)

            img0_residual = self.sofi_unet0(img1, img1_warped_sofi, flow_sofi_warp, ctx1_sofi)
            img1_residual = self.sofi_unet1(img0, img0_warped_sofi, flow_sofi_warp, ctx0_sofi)

            refined_img0  = self.clamp(img1_warped_sofi + img0_residual)
            refined_img1  = self.clamp(img0_warped_sofi + img1_residual)
            refined_img_list[3] = refined_img0
            refined_img_list[4] = refined_img1

        teacher_dict = { 'flow_teacher': flow_tea, 'merged_teacher': merged_tea }
        # flow_list, mask_list: flow and mask in 3 different scales.
        # If mid_gt is None, loss_distill = 0.
        return flow_list, sofi_flow_list, mask_list[2], crude_img_list, refined_img_list, teacher_dict, loss_distill
