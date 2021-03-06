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
            # m->1 flow to warp img1, which approximates mid.
            # m->1 flow starts from the 2*M-th channel.
            img1_warped = backwarp(img1, multiflow[:, 2*M+i*2 : 2*M+i*2+2])
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
