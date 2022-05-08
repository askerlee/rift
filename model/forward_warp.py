# Adapted from
# https://github.com/lliuz/ARFlow/blob/master/utils/warp_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

def mesh_grid(B, H, W):
    # mesh grid consisting of (x, y) coordinates.
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def forward_warp(image, flow12):
    B, _, H, W  = flow12.size()
    C           = image.shape[1]
    # base_grid: [B, 2, H, W], mesh grid consisting of (x, y) coordinates.
    base_grid   = mesh_grid(B, H, W).type_as(flow12)  
    # base_grid + flow12: [B, 2, H, W], coordinates mapped onto image2.
    coord_map   = base_grid + flow12

    # x, y: flattend x, y coordinates.
    x = coord_map[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = coord_map[:, 1, :, :].view(B, -1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    # x_left,  y_top:    left/top     coordinates of the pixel, capped at image boundary.
    # x_right, y_bottom: right/bottom coordinates of the pixel, capped at image boundary.
    x_left      = x0.clamp(0, W - 1)
    y_top       = y0.clamp(0, H - 1)
    x_right     = x1.clamp(0, W - 1)
    y_bottom    = y1.clamp(0, H - 1)

    x_right_is_out  = (x1 != x_right)
    y_bottom_is_out = (y1 != y_bottom)
    x_left_is_out   = (x0 != x_left)
    y_top_is_out    = (y0 != y_top)
    # invalid: invalid coordinates (going out of the image).
    invalid = torch.cat([x_right_is_out | y_bottom_is_out,
                         x_right_is_out | y_top_is_out,
                         x_left_is_out  | y_bottom_is_out,
                         x_left_is_out  | y_top_is_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    # bg: batch, geometric (B, H*W)
    indeg_bg        = torch.zeros(B, H*W).type_as(image)
    image_warped_bgc  = torch.zeros(B, H*W, C).type_as(image)
    
    # xy_addr: [B, 4*H*W]. 
    # Each block is the (flattened) coordinates that index one corner of the 
    # flow destination (enclosed in 4 corner pixels). Four corners are:
    # bottom-right, top-right, bottom-left, top-left.
    # 4---2
    # |   |
    # 3---1
    xy_addr = torch.cat([x_right + W * y_bottom,
                         x_right + W * y_top,
                         x_left  + W * y_bottom,
                         x_left  + W * y_top], 1).long()  
    # weights_bg: [B, 4*H*W].
    # Four weight blocks are the interpolation weights for four corner pixels.
    # For example, weight_br[0,0] is how much of the pixel value of src[0, 0] 
    # will be distributed to the bottom-right corner (address: xy_addr_br[0, 0])
    # of the flow destination (enclosed in 4 corner pixels).
    weights_bg = torch.cat([ (1 - torch.abs(x - x_right)) * (1 - torch.abs(y - y_bottom)),
                             (1 - torch.abs(x - x_right)) * (1 - torch.abs(y - y_top)),
                             (1 - torch.abs(x - x_left))  * (1 - torch.abs(y - y_bottom)),
                             (1 - torch.abs(x - x_left))  * (1 - torch.abs(y - y_top)) ],
                           1)
    weights_bg[invalid] = 0

    # image_bgc: [B, H*W, C]. g: flattened geometrical dimensions (H, W).
    image_bgc   = image.reshape(B, C, -1).permute(0, 2, 1)
    # image_b4gc: [B, 4*H*W, C].
    image_b4gc  = image_bgc.repeat(1, 4, 1)
    indeg_bg.scatter_add_(1, xy_addr, weights_bg)
    xy_addr_c   = xy_addr.unsqueeze(-1).repeat(1, 1, C)
    image_warped_bgc.scatter_add_(1, xy_addr_c, weights_bg.unsqueeze(-1) * image_b4gc)
    indeg_bg[indeg_bg < 0.5] = 1
    image_warped_bgc  = image_warped_bgc / indeg_bg.unsqueeze(-1)
    image_warped      = image_warped_bgc.permute(0, 2, 1).reshape(B, C, H, W)

    return image_warped

def fwarp_blob(mid_flow, mid_multiflow, multimask_score, M, 
               fwarp_do_normalize=False):
    flow_m0,            flow_m1             = mid_flow[:, :2],        mid_flow[:, 2:]
    multiflow_m0,       multiflow_m1        = mid_multiflow[:, :2*M], mid_multiflow[:, 2*M:4*M]
    multimask_score_m0, multimask_score_m1  = multimask_score[:, :M], multimask_score[:, M:2*M]
    global_mask_score                       = multimask_score[:, [-1]]

    # First use 2*(middle->0, middle->1) flow to approximate the flow (1->0, 0->1).
    # But m0, m1 flow is aligned to the middle frame. Has to warp to align with img0/img1.
    # forward_warp is slow. To speed up, we pack them up, warp, and then unpack.
    if fwarp_do_normalize:
        # ones is all-one pseudo images used to count the in-degree of each target pixel 
        # in image 0 and image 1. The counts are fractional.
        ones = torch.ones_like(mid_multiflow[:, [0]])
    else:
        # ones is of zero-sized tensors, to act as placeholders in the concatenated array.
        ones = torch.ones_like(mid_multiflow[:, []])

    # m->0, m->1 should be around half of 1->0, 0->1. 
    # So multiflow_m1 * 2 approximates multiflow01, and multiflow_m0 * 2 approximates multiflow10.
    blob1 = torch.cat([multiflow_m1 * 2, flow_m1 * 2, multimask_score_m1, global_mask_score, ones], 1)
    blob0 = torch.cat([multiflow_m0 * 2, flow_m0 * 2, multimask_score_m0, global_mask_score, ones], 1)
    # forward_warp m1 flow (and scores) by m0 flow, so that coordiates of the middle frame 
    # are aligned with coordinates of img0.
    blob1_fw0 = forward_warp(blob1, flow_m0)
    # forward_warp m0 flow (and scores) by m1 flow, so that coordiates of the middle frame
    # are aligned with coordinates of img1.
    blob0_fw1 = forward_warp(blob0, flow_m1)
    # multiflow01_sofi:         2*M channels
    # flow01:                   2 channels
    # multimask_score01_sofi:   M channels
    # global_mask_score01_sofi: 1 channel
    # img1_fw0:                 3 channels
    # indeg_m0:                 1 or 0 channels (= fwarp_do_normalize)
    assert blob1_fw0.shape[1] == 3 * M + 3 + fwarp_do_normalize
    assert blob0_fw1.shape[1] == 3 * M + 3 + fwarp_do_normalize
    multiflow01_sofi, flow01, multimask_score01_sofi, global_mask_score01_sofi, indeg_m0 = \
        blob1_fw0[:, :2*M], blob1_fw0[:, 2*M:2*M+2], blob1_fw0[:, 2*M+2:3*M+2], \
        blob1_fw0[:, 3*M+2:3*M+3], blob1_fw0[:, 3*M+3:]
    multiflow10_sofi, flow10, multimask_score10_sofi, global_mask_score10_sofi, indeg_m1 = \
        blob0_fw1[:, :2*M], blob0_fw1[:, 2*M:2*M+2], blob0_fw1[:, 2*M+2:3*M+2], \
        blob0_fw1[:, 3*M+2:3*M+3], blob0_fw1[:, 3*M+3:]

    if fwarp_do_normalize:
        # indeg_m0: 1-channel fractional counts of pixels in middle frame mapped to each pixel in img0.
        # indeg_m1: 1-channel fractional counts of pixels in middle frame mapped to each pixel in img1.
        # Only normalize the pixels whose in-degree >= 0.5.
        # Flow values at pixels with zero or small fractional in-degree are kept unchanged; probably only 
        # (a fraction of) one source pixel is mapped to this pixel.
        indeg_m0[ indeg_m0 < 0.5 ] = 1
        indeg_m1[ indeg_m1 < 0.5 ] = 1
        # multiflow01_sofi is forward_warped from multiflow_m1 using flow_m0. So divided by indeg_m0.
        # multiflow10_sofi is forward_warped from multiflow_m0 using flow_m1. So divided by indeg_m1.
        # Selectively backprop through indeg_m0 and indeg_m1.
        # Backpropping from multiflow01_sofi/multiflow10_sofi through indeg_m0/indeg_m1 
        # leads to slightly better performance.
        multiflow01_sofi         = multiflow01_sofi / indeg_m0
        multiflow10_sofi         = multiflow10_sofi / indeg_m1
        # flow01 is forward_warped from flow_m1 using flow_m0. So divided by indeg_m0.
        # flow10 is forward_warped from flow_m0 using flow_m1. So divided by indeg_m1.
        # !!! Backpropping from flow01/flow10 through indeg_m0/indeg_m1 leads to divergence !!!
        # The reason is unknown.
        flow01                   = flow01 / indeg_m0.detach()
        flow10                   = flow10 / indeg_m1.detach()
        # Normalizing the multi-mask scores have less impact, as they are pixel-wise transformed by softmax.
        # The relative orders of the mask scores at each pixel don't change, 
        # but the softmax weights do change a little bit after normalization.
        # No backpropping through indeg_m0/indeg_m1 leads to slightly better performance.
        # Maybe because in multiwarp(), multiflow01_sofi * softmax(multimask_score01_sofi),
        # and if both are bprop-ed, the two types of gradients interfere and cause some bad effect.
        multimask_score01_sofi   = multimask_score01_sofi / indeg_m0.detach()
        multimask_score10_sofi   = multimask_score10_sofi / indeg_m1.detach()
        # The effect of normalizing the global mask score is unknown. 
        # But doing normalization shouldn't make it worse.
        # Backpropping through indeg_m0/indeg_m1 leads to slightly better performance.
        # Maybe because this bp path is independent of multiwarp().
        global_mask_score01_sofi = global_mask_score01_sofi / indeg_m0
        global_mask_score10_sofi = global_mask_score10_sofi / indeg_m1

    return multiflow01_sofi, flow01, multimask_score01_sofi, global_mask_score01_sofi, \
           multiflow10_sofi, flow10, multimask_score10_sofi, global_mask_score10_sofi

def fwarp_imgs(img0, img1, flow_sofi, fwarp_do_normalize=False):
    flow_10, flow_01 = flow_sofi[:, :2], flow_sofi[:, 2:]

    if fwarp_do_normalize:
        # ones0, ones1 are all-one pseudo images used to count the in-degree of each target pixel 
        # in image 0 and image 1. The counts are fractional.
        ones0 = torch.ones_like(img0[:, [0]])
        ones1 = torch.ones_like(img0[:, [0]])
    else:
        # ones0, ones1 are of zero-sized tensors, to act as placeholders in the concatenated array.
        ones0 = torch.ones_like(img0[:, []])
        ones1 = torch.ones_like(img0[:, []])

    blob1 = torch.cat([img1, ones1], 1)
    blob0 = torch.cat([img0, ones0], 1)
    # forward_warp m1 flow (and scores) by m0 flow, so that coordiates of the middle frame 
    # are aligned with coordinates of img0.
    blob1_fw0 = forward_warp(blob1, flow_10)
    # forward_warp m0 flow (and scores) by m1 flow, so that coordiates of the middle frame
    # are aligned with coordinates of img1.
    blob0_fw1 = forward_warp(blob0, flow_01)
    # img1_fw0:                 3 channels
    # indeg_m0:                 1 or 0 channels (= fwarp_do_normalize)
    assert blob1_fw0.shape[1] == 3 + fwarp_do_normalize
    assert blob0_fw1.shape[1] == 3 + fwarp_do_normalize
    img1_fw0, indeg_10 = blob1_fw0[:, :3], blob1_fw0[:, 3:]
    img0_fw1, indeg_01 = blob0_fw1[:, :3], blob0_fw1[:, 3:]

    if fwarp_do_normalize:
        # indeg_10: 1-channel fractional counts of pixels in middle frame mapped to each pixel in img0.
        # indeg_01: 1-channel fractional counts of pixels in middle frame mapped to each pixel in img1.
        # Only normalize the pixels whose in-degree >= 0.5.
        # Flow values at pixels with zero or small fractional in-degree are kept unchanged; probably only 
        # (a fraction of) one source pixel is mapped to this pixel.
        indeg_10[ indeg_10 < 0.5 ] = 1
        indeg_01[ indeg_01 < 0.5 ] = 1
        # img1_fw0 is forward_warped from img1 using flow_10. So divided by indeg_10.
        # img0_fw1 is forward_warped from img0 using flow_01. So divided by indeg_01.
        img1_fw0 = img1_fw0 / indeg_10.detach()
        img0_fw1 = img0_fw1 / indeg_01.detach()

    img0_fw1 = torch.clamp(img0_fw1, 0, 1)
    img1_fw0 = torch.clamp(img1_fw0, 0, 1)
    return img0_fw1, img1_fw0

if __name__ == '__main__':	
    from model.visgraph import make_dot
    from model.warp import backwarp

    torch.set_printoptions(sci_mode=False)
    flow12  = torch.full((1, 2, 5, 5),  1., requires_grad=True)
    # flow12 and flow21 are reverse flows.
    # flow21  = torch.full((1, 2, 5, 5), -1., requires_grad=True)
    x       = torch.randn(1, 2, 5, 5, requires_grad=True)
    x_warp = forward_warp(x, flow12)
    print(x)
    print(x_warp)
    #x_warp.sum().backward()
    #print(flow12)
    #print(flow12.grad)
    x2 = backwarp(x_warp, flow12)
    print(x2)
    #g = make_dot(x_warp)
    #g.view()
