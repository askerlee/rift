import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Precomputed coordinate grids dictionary, with (tensor device, tensor size) as the keys.
backwarp_tenGrid = {}

# warp feature maps according to flow. ten: tensor?
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


# Warp images with multiple groups of flow, and combine them into one group with flow group attention.
# If M==1, multiwarp falls back to warp.
def multiwarp(img0, img1, multiflow, multimask_score, M):
    img0_warped_list = []
    img1_warped_list = []
    multimask01_score_list = []
    multimask10_score_list = []
    # multiflow at dim=1: 
    # flow01_1, flow01_2, ..., flow01_M, flow10_1, flow10_2, ..., flow10_M
    # 01 means flow from img0 to middle, 10 means flow from img1 to middle.
    # Each block has 2 channels.
    for i in range(M):
        img0_warped = warp(img0, multiflow[:, i*2 : i*2+2])
        img0_warped_list.append(img0_warped)
        # Warp the mask scores. The scores are generated mostly based on
        # unwarped images, and there's misalignment between warped images and unwarped 
        # scores. Therefore, we need to warp the mask scores as well.
        # But doing so only leads to very slight improvement (~0.02 psnr).
        mask01_score_warped = warp(multimask_score[:, [i]], multiflow[:, i*2 : i*2+2])
        multimask01_score_list.append(mask01_score_warped)

        if img1 is not None:
            img1_warped = warp(img1, multiflow[:, i*2+2*M : i*2+2*M+2])
            img1_warped_list.append(img1_warped)
            mask10_score_warped = warp(multimask_score[:, [i+M]], multiflow[:, i*2+2*M : i*2+2*M+2])
            multimask10_score_list.append(mask10_score_warped)
        else:
            # placeholder.
            img1_warped_list.append(None)

    if M == 1:
        return img0_warped_list[0], img1_warped_list[0]

    # multimask_score: 2*M+1 channels. 2*M for M groups of L/R flow attention scores, 
    # L: 0 -> 0.5, R: 1 -> 0.5.
    # LR_0, LR_1, ..., LR_M, RL_0, ..., RL_M, LR~RL weight
    # 1: mask, for the warp0-warp1 combination weight.
    assert multimask_score.shape[1] == 2*M+1

    # img0_warped_list, img1_warped_list are two lists, each of length M.
    # => [16, M, 3, 224, 224]
    warped_img0s        = torch.stack(img0_warped_list, dim=1)
    multimask01_score   = torch.stack(multimask01_score_list, dim=1)
    # warp0_attn: [16, M, 1, 224, 224]
    warp0_attn  = torch.softmax(multimask01_score, dim=1)
    img0_warped = (warp0_attn * warped_img0s).sum(dim=1)

    if img1 is not None:
        warped_img1s        = torch.stack(img1_warped_list, dim=1)
        multimask10_score   = torch.stack(multimask10_score_list, dim=1)
        warp1_attn  = torch.softmax(multimask10_score, dim=1)
        img1_warped = (warp1_attn * warped_img1s).sum(dim=1)
    else:
        img1_warped = None

    return img0_warped, img1_warped

# Use flow group attention to combine multiple flow groups into one.
def multimerge_flow(multiflow, multimask_score, M):
    if M == 1:
        multiflow01, multiflow10 = multiflow[:, :2], multiflow[:, 2:4]
        flow = multiflow
    else:
        multiflow01 = multiflow[:, :M*2]
        multiflow10 = multiflow[:, M*2:]
        # multiflow: [16, 4*M, 224, 224]
        mf_unpack_shape = list(multiflow.shape)
        mf_unpack_shape[1:2] = [M, 2]
        # multiflow01, multiflow10: [16, M, 2, 224, 224]
        multiflow01_unpack = multiflow01.reshape(mf_unpack_shape)
        multiflow10_unpack = multiflow10.reshape(mf_unpack_shape)
        # warp0_attn, warp1_attn: [16, M, 1, 224, 224]
        # multiflow is unwarped, so we don't need to warp the mask scores.
        warp0_attn = torch.softmax(multimask_score[:, :M], dim=1).unsqueeze(dim=2)
        warp1_attn = torch.softmax(multimask_score[:, M:2*M], dim=1).unsqueeze(dim=2)
        # flow01, flow10: [16, 2, 224, 224]
        flow01 = (warp0_attn * multiflow01_unpack).sum(dim=1)
        flow10 = (warp1_attn * multiflow10_unpack).sum(dim=1)
        flow = torch.cat([flow01, flow10], dim=1)
    # Returned multiflow01, multiflow10 are not combined with attention. 
    # They will be used in contextnet.
    return flow, multiflow01, multiflow10, flow01, flow10
