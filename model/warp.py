import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Precomputed coordinate grids dictionary, with (tensor device, tensor size) as the keys.
backwarp_tenGrid = {}

# backwarp and multiwarp are doing backward warping using the forward flow.
# backwarp feature maps according to flow. ten: tensor?
def backwarp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)
    
    # channel 0: x (horizontal), channel 1: y (vertical).
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


# Warp images with multiple groups of flow, and combine them into one group with flow group attention.
# If M==1, multiwarp falls back to backwarp.
def multiwarp(img0, img1, multiflow, multimask_score, M):
    img0_warped_list = []
    img1_warped_list = []
    multimaskm0_score_list = []
    multimaskm1_score_list = []
    # multiflow at dim=1: 
    # flowm0_1, flowm0_2, ..., flowm0_M, flowm1_1, flowm1_2, ..., flowm1_M
    # m0 means flow from middle to img0, m1 means flow from middle to img1.
    # Each block has 2 channels.
    for i in range(M):
        # mid -> 0 flow to warp img0, which approximates mid.
        img0_warped = backwarp(img0, multiflow[:, i*2 : i*2+2])
        img0_warped_list.append(img0_warped)
        # Warp the mask scores. The scores are generated mostly based on
        # unwarped images, and there's misalignment between warped images and unwarped 
        # scores. Therefore, we need to warp the mask scores as well.
        # But doing so only leads to very slight improvement (~0.02 psnr).
        maskm0_score_warped = backwarp(multimask_score[:, [i]], multiflow[:, i*2 : i*2+2])
        multimaskm0_score_list.append(maskm0_score_warped)

        if img1 is not None:
            # mid -> 1 flow to warp img1, which approximates mid.
            img1_warped = backwarp(img1, multiflow[:, i*2+2*M : i*2+2*M+2])
            img1_warped_list.append(img1_warped)
            maskm1_score_warped = backwarp(multimask_score[:, [i+M]], multiflow[:, i*2+2*M : i*2+2*M+2])
            multimaskm1_score_list.append(maskm1_score_warped)
        else:
            # placeholder.
            img1_warped_list.append(None)

    if M == 1:
        return img0_warped_list[0], img1_warped_list[0]

    # multimask_score: 2*M+1 channels. 2*M for M groups of ML/MR flow attention scores, 
    # ML: 0.5 -> 0, MR: 0.5 -> 1.
    # ML_0, ML_1, ..., ML_M, MR_0, ..., MR_M, ML~MR weight
    # 1: mask, for the warp0-warp1 combination weight.
    # For sofi, the global mask scores may be bidirectional. In that case, there are totally 2*M+2 channels.
    if img1 is not None:
        assert multimask_score.shape[1] == 2*M+1 or multimask_score.shape[1] == 2*M+2

    # img0_warped_list, img1_warped_list are two lists, each of length M.
    # => [16, M, 3, 224, 224]
    warped_img0s        = torch.stack(img0_warped_list, dim=1)
    multimaskm0_score   = torch.stack(multimaskm0_score_list, dim=1)
    # warp0_attn: [16, M, 1, 224, 224]
    warp0_attn  = torch.softmax(multimaskm0_score, dim=1)
    img0_warped = (warp0_attn * warped_img0s).sum(dim=1)

    if img1 is not None:
        warped_img1s        = torch.stack(img1_warped_list, dim=1)
        multimaskm1_score   = torch.stack(multimaskm1_score_list, dim=1)
        warp1_attn  = torch.softmax(multimaskm1_score, dim=1)
        img1_warped = (warp1_attn * warped_img1s).sum(dim=1)
    else:
        img1_warped = None

    return img0_warped, img1_warped

# Use flow group attention to combine multiple flow groups into one.
def multimerge_flow(multiflow, multimask_score, M):
    if M == 1:
        multiflowm0, multiflowm1 = multiflow[:, :2], multiflow[:, 2:4]
        flow = multiflow
    else:
        multiflowm0 = multiflow[:, :2*M]
        multiflowm1 = multiflow[:, 2*M:]
        # multiflow: [16, 4*M, 224, 224]
        mf_unpack_shape = list(multiflow.shape)
        mf_unpack_shape[1:2] = [M, 2]
        # multiflowm0, multiflowm1: [16, M, 2, 224, 224]
        multiflowm0_unpack = multiflowm0.reshape(mf_unpack_shape)
        multiflowm1_unpack = multiflowm1.reshape(mf_unpack_shape)
        # warp0_attn, warp1_attn: [16, M, 1, 224, 224]
        # multiflow is unwarped, so we don't need to warp the mask scores.
        warp0_attn = torch.softmax(multimask_score[:, :M], dim=1).unsqueeze(dim=2)
        warp1_attn = torch.softmax(multimask_score[:, M:2*M], dim=1).unsqueeze(dim=2)
        # flowm0, flowm1: [16, 2, 224, 224]
        flowm0 = (warp0_attn * multiflowm0_unpack).sum(dim=1)
        flowm1 = (warp1_attn * multiflowm1_unpack).sum(dim=1)
        flow = torch.cat([flowm0, flowm1], dim=1)
    return flow

def fwarp_blob(fwarp, mid_flow, mid_multiflow, multimask_score, M, 
               fwarp_do_normalize=True):
    flow_m0, flow_m1                        = mid_flow[:, :2],        mid_flow[:, 2:]
    multiflow_m0, multiflow_m1              = mid_multiflow[:, :2*M], mid_multiflow[:, 2*M:4*M]
    multimask_score_m0, multimask_score_m1  = multimask_score[:, :M], multimask_score[:, M:2*M]
    global_mask_score                       = multimask_score[:, [-1]]

    # forward_flow() accepts flow in the shape of [B, H, W, 2]
    flow_m0_bhwc = flow_m0.permute(0, 2, 3, 1)
    flow_m1_bhwc = flow_m1.permute(0, 2, 3, 1)
    # First use 2*(middle->0, middle->1) flow to approximate the flow (1->0, 0->1).
    # But m0, m1 flow is aligned to the middle frame. Has to warp to align with img0/img1.
    # forward_warp is slow. To speed up, we pack them up, warp, and then unpack.
    if fwarp_do_normalize:
        # ones0, ones1 are all-one pseudo images used to count the in-degree of each target pixel 
        # in image 0 and image 1. The counts are fractional.
        ones0 = torch.ones_like(mid_multiflow[:, [0]])
        ones1 = torch.ones_like(mid_multiflow[:, [0]])
    else:
        # ones0, ones1 are of zero-sized tensors, to act as placeholders in the concatenated array.
        ones0 = torch.ones_like(mid_multiflow[:, []])
        ones1 = torch.ones_like(mid_multiflow[:, []])

    # m->0, m->1 should be around half of 1->0, 0->1. 
    # So multiflow_m1 * 2 approximates multiflow01, and multiflow_m0 * 2 approximates multiflow10.
    blob1 = torch.cat([multiflow_m1 * 2, flow_m1 * 2, multimask_score_m1, global_mask_score, ones1], 1)
    blob0 = torch.cat([multiflow_m0 * 2, flow_m0 * 2, multimask_score_m0, global_mask_score, ones0], 1)
    # fwarp m1 flow (and scores) by m0 flow, so that coordiates of the middle frame 
    # are aligned with coordinates of img0.
    blob1_fw0 = fwarp(blob1, flow_m0_bhwc)
    # fwarp m0 flow (and scores) by m1 flow, so that coordiates of the middle frame
    # are aligned with coordinates of img1.
    blob0_fw1 = fwarp(blob0, flow_m1_bhwc)
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
        # multiflow01_sofi is fwarped from multiflow_m1 using flow_m0. So divided by indeg_m0.
        # multiflow10_sofi is fwarped from multiflow_m0 using flow_m1. So divided by indeg_m1.
        # Selectively backprop through indeg_m0 and indeg_m1.
        # Backpropping from multiflow01_sofi/multiflow10_sofi through indeg_m0/indeg_m1 
        # leads to slightly better performance.
        multiflow01_sofi         = multiflow01_sofi / indeg_m0
        multiflow10_sofi         = multiflow10_sofi / indeg_m1
        # flow01 is fwarped from flow_m1 using flow_m0. So divided by indeg_m0.
        # flow10 is fwarped from flow_m0 using flow_m1. So divided by indeg_m1.
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

def fwarp_imgs(fwarp, img0, img1, flow_sofi, fwarp_do_normalize=True):
    flow_10, flow_01 = flow_sofi[:, :2], flow_sofi[:, 2:]

    # forward_flow() accepts flow in the shape of [B, H, W, 2]
    flow_10_bhwc = flow_10.permute(0, 2, 3, 1)
    flow_01_bhwc = flow_01.permute(0, 2, 3, 1)
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
    # fwarp m1 flow (and scores) by m0 flow, so that coordiates of the middle frame 
    # are aligned with coordinates of img0.
    blob1_fw0 = fwarp(blob1, flow_10_bhwc)
    # fwarp m0 flow (and scores) by m1 flow, so that coordiates of the middle frame
    # are aligned with coordinates of img1.
    blob0_fw1 = fwarp(blob0, flow_01_bhwc)
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
        # img1_fw0 is fwarped from img1 using flow_10. So divided by indeg_10.
        # img0_fw1 is fwarped from img0 using flow_01. So divided by indeg_01.
        img1_fw0 = img1_fw0 / indeg_10.detach()
        img0_fw1 = img0_fw1 / indeg_01.detach()

    img0_fw1 = torch.clamp(img0_fw1, 0, 1)
    img1_fw0 = torch.clamp(img1_fw0, 0, 1)
    return img0_fw1, img1_fw0
