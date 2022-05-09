import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warp import backwarp, multiwarp, multimerge_flow
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        # PReLU: Parametric ReLU, making the slope in LeakyReLU learnable.
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    # Usually out_planes > in_planes, say out_planes = 2 * in_planes. So conv1 does channel expansion.
    # By default stride=2, i.e., the output feature map is half of the input in size.
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Contextnet_rife(nn.Module):
    def __init__(self, c=16):
        super(Contextnet_rife, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = backwarp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = backwarp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = backwarp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = backwarp(x, flow)
        return [f1, f2, f3, f4]
        
# Contextnet generates warped features of the input image. 
# flow is not used as input to generate the features, but to warp the features.
class Contextnet(nn.Module):
    def __init__(self, c=16):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    # Can take either one or two sets of multiflow/multimask_score.
    def forward(self, x, M, multiflow, multimask_score, multiflow2=None, multimask_score2=None):
        x = self.conv1(x)
        multiflow = F.interpolate(multiflow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1, _ = multiwarp(x, None, multiflow, multimask_score, M)
        if multiflow2 is not None:
            multiflow2 = F.interpolate(multiflow2, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            g1, _ = multiwarp(x, None, multiflow2, multimask_score2, M)

        x = self.conv2(x)
        multiflow = F.interpolate(multiflow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2, _ = multiwarp(x, None, multiflow, multimask_score, M)
        if multiflow2 is not None:
            multiflow2 = F.interpolate(multiflow2, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            g2, _ = multiwarp(x, None, multiflow2, multimask_score2, M)

        x = self.conv3(x)
        multiflow = F.interpolate(multiflow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3, _ = multiwarp(x, None, multiflow, multimask_score, M)
        if multiflow2 is not None:
            multiflow2 = F.interpolate(multiflow2, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            g3, _ = multiwarp(x, None, multiflow2, multimask_score2, M)

        x = self.conv4(x)
        multiflow = F.interpolate(multiflow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4, _ = multiwarp(x, None, multiflow, multimask_score, M)
        if multiflow2 is not None:
            multiflow2 = F.interpolate(multiflow2, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            g4, _ = multiwarp(x, None, multiflow2, multimask_score2, M)

        fs = [f1, f2, f3, f4]
        if multiflow2 is not None:
            gs = [g1, g2, g3, g4]
        else:
            gs = None

        # f1, f2, f3, f4 are gradually scaled down. f1: 1/2, f2: 1/4, f3: 1/8, f4: 1/16 of the input x.
        # f1, f2, f3, f4 are warped by flow.
        # The feature maps in every scale are warped only after the last conv of the corresponding scale, 
        # not in the middle. I.e., here no conv will be applied to warped features.
        return fs, gs

# Unet: 17 channels of input, 3 channels of output.
class Unet(nn.Module):
    def __init__(self, c=16):
        super(Unet, self).__init__()
        # 17: 4 images (4*3) + mask (1) + flow (4)
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0   = deconv(32*c, 8*c)
        self.up1   = deconv(16*c, 4*c)
        self.up2   = deconv(8*c, 2*c)
        self.up3   = deconv(4*c, c)
        self.conv  = nn.Conv2d(c, 3, 3, 1, 1)

    # context0: 4 conv features of img0 extracted with contextnet. channels: c, 2c, 4c, 8c.
    # context1: 4 conv features of img1 extracted with contextnet. channels: c, 2c, 4c, 8c.
    # context0, context1 are two lists of 4 feature maps in 4 scales.
    # Unet takes original images, warped images, mask and flow as input, much richer than contextnet.
    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, context0, context1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, context0[0], context1[0]), 1))
        s2 = self.down2(torch.cat((s1, context0[1], context1[1]), 1))
        s3 = self.down3(torch.cat((s2, context0[2], context1[2]), 1))
        x  = self.up0(  torch.cat((s3, context0[3], context1[3]), 1))
        x  = self.up1(torch.cat((x, s2), 1)) 
        x  = self.up2(torch.cat((x, s1), 1)) 
        x  = self.up3(torch.cat((x, s0), 1)) 
        x  = self.conv(x)
        # 0 < x < 1 due to sigmoid.
        # the returned tensor is scaled to [-1, 1], and used as image residual.
        # return torch.sigmoid(x)
        return torch.tanh(x)

# SOFI_Unet: 10 channels of input, 3 channels of output.
class SOFI_Unet(nn.Module):
    def __init__(self, c=16):
        super(SOFI_Unet, self).__init__()
        ## 16: 2 images (2*3) + global_mask_score_sofi (2 directions) + flow (4 normal + 4 warped)
        # 11: 2 images (2*3) + global_mask_score_sofi (1 direction) + flow (2 normal + 2 warped)
        self.down0 = Conv2(11, 2*c)
        self.down1 = Conv2(3*c, 4*c)
        self.down2 = Conv2(6*c, 8*c)
        self.down3 = Conv2(12*c, 16*c)
        self.up0   = deconv(24*c, 8*c)
        self.up1   = deconv(16*c, 4*c)
        self.up2   = deconv(8*c, 2*c)
        self.up3   = deconv(4*c, c)
        self.conv  = nn.Conv2d(c, 3, 3, 1, 1)

    # context0: 4 conv features of img0 extracted with contextnet. channels: c, 2c, 4c, 8c. 
    # context1: 4 conv features of img1 extracted with contextnet. channels: c, 2c, 4c, 8c.
    # context0, context1 are two lists of 4 feature maps in 4 scales.
    # Unet takes original images, warped images, mask and flow as input, much richer than contextnet.
    def forward(self, img0, warped_img0, mask, flow, context0):
        s0 = self.down0(torch.cat((img0, warped_img0, mask, flow), 1))  # 15 -> 2c
        s1 = self.down1(torch.cat((s0, context0[0]), 1))                # 3c -> 4c
        s2 = self.down2(torch.cat((s1, context0[1]), 1))                # 6c -> 8c
        s3 = self.down3(torch.cat((s2, context0[2]), 1))                # 12c -> 16c
        x  = self.up0(  torch.cat((s3, context0[3]), 1))                # 24c -> 8c
        x  = self.up1(torch.cat((x, s2), 1))                            # 16c -> 4c
        x  = self.up2(torch.cat((x, s1), 1))                            # 8c -> 2c
        x  = self.up3(torch.cat((x, s0), 1))                            # 4c -> c
        x  = self.conv(x)
        # 0 < x < 1 due to sigmoid.
        # the returned tensor is scaled to [-1, 1], and used as image residual.
        # return torch.sigmoid(x)
        return torch.tanh(x)
