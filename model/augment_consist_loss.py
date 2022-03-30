import torch
import cv2
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate


def visualize_flow(flow, save_name):
    # https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    # flow: expected shape (H, W, 2)
    H, W, C = flow.shape
    flow_clone = flow.clone().detach().cpu().numpy()
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow_clone[..., 0], flow_clone[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(save_name, bgr)


# img0, img1, gt are 4D tensors of (B, 3, 256, 448). gt are the middle frames.
def random_shift(img0, img1, gt, flow, flow_teacher, shift_sigmas=(16, 10)):
    B, C, H, W = img0.shape
    u_shift_sigma, v_shift_sigma = shift_sigmas
    # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    # Make sure at most one of dx, dy is large. Otherwise the shift is too difficult.
    if random.random() > 0.5:
        dx = np.random.laplace(0, u_shift_sigma / 4)
        dy = np.random.laplace(0, v_shift_sigma)
    else:
        dx = np.random.laplace(0, u_shift_sigma)
        dy = np.random.laplace(0, v_shift_sigma / 4)

    # Make sure dx and dy are even numbers.
    dx = (int(dx) // 2) * 2
    dy = (int(dy) // 2) * 2
    dx2 = dx // 2
    dy2 = dy // 2
    
    # If flow=0, pixels at (dy, dx)_0a <-> (0, 0)_1a.
    if dx >= 0 and dy >= 0:
        # img0 is cropped at the bottom-right corner.               img0[:-dy, :-dx]
        img0_bound = (0,  H - dy,  0,  W - dx)
        # img1 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
        #                                                           img1[dy:,  dx:]
        img1_bound = (dy, H,       dx, W)
    if dx >= 0 and dy < 0:
        # img0 is cropped at the right side, and shifted to the up. img0[-dy:, :-dx]
        img0_bound = (-dy, H,      0,  W - dx)
        # img1 is shifted to the left and cropped at the bottom.    img1[:dy,  dx:]
        img1_bound = (0,   H + dy, dx, W)
        # (dx, 0)_0 => (dx, dy)_0a, (dx, 0)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.          
    if dx < 0 and dy >= 0:
        # img0 is shifted to the left, and cropped at the bottom.   img0[:-dy, -dx:]
        img0_bound = (0,   H - dy, -dx, W)
        # img1 is cropped at the right side, and shifted to the up. img1[dy:,  :dx]
        img1_bound = (dy,  H,      0,   W + dx)
        # (0, dy)_0 => (dx, dy)_0a, (0, dy)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.         
    if dx < 0 and dy < 0:
        # img0 is shifted by (-dx, -dy) to the left and up. img0[-dy:, -dx:]
        img0_bound = (-dy, H,      -dx, W)
        # img1 is cropped at the bottom-right corner.       img1[:dy,  :dx]
        img1_bound = (0,   H + dy, 0,   W + dx)

    # Swapping the shifts to img0 and img1, to increase diversity.
    reversed_01 = random.random() > 0.5
    # Make the shifted img0, img1, gt shifted copies of the same image. Performs slightly worse.
    do_identity_shift = False
    
    # dxy is the motion of the middle frame. It's always half of the relative motion between frames 0 and 1.
    if reversed_01:
        img0_bound, img1_bound = img1_bound, img0_bound
        if do_identity_shift:
            img0, img1, gt = img1, img1, img1
        # Shifting to img0 & img1 are swapped.
        # dxy: offsets (from old to new flow) for two directions.
        # Note the middle frame is shifted by *half* of dx, dy.
        # Note the flows are for backward warping (from middle to 0/1).
        # From 0.5 -> 0: negative delta (from the old flow). old 0.5->0 flow - (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: positive delta (from the old flow). old 0.5->1 flow + (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([-dx2, -dy2,  dx2,  dy2], dtype=float, device=img0.device)
    else:
        if do_identity_shift:
            img0, img1, gt = img0, img0, img0        
        # Note the middle frame is shifted by *half* of dx, dy.
        # From 0.5 -> 0: positive delta (from the old flow). old 0.5->0 flow + (dx, dy) = new 0.5->0 flow.
        # From 0.5 -> 1: negative delta (from the old flow). old 0.5->1 flow - (dx, dy) = new 0.5->1 flow.
        dxy = torch.tensor([ dx2,  dy2, -dx2, -dy2], dtype=float, device=img0.device)

    # T, B, L, R: top, bottom, left, right boundary.
    T1, B1, L1, R1 = img0_bound
    T2, B2, L2, R2 = img1_bound
    # For the middle frame, the numbers of cropped pixels at the left and right, or the up and the bottom are equal.
    # Therefore, after padding, the middle frame doesn't shift. It's just cropped at the center and 
    # zero-padded at the four sides.
    # This property makes it easy to compare the flow before and after shifting.
    dx2, dy2 = abs(dx2), abs(dy2)
    # TM, BM, LM, RM: new boundary of the middle frame.
    TM, BM, LM, RM = dy2, H - dy2, dx2, W - dx2
    img0a = img0[:, :, T1:B1, L1:R1]
    img1a = img1[:, :, T2:B2, L2:R2]
    gta   = gt[:, :, TM:BM, LM:RM]

    # Pad img0a, img1a, gta by half of (dy, dx), to the original size.
    # Note the pads are ordered as (x1, x2, y1, y2) instead of (y1, y2, x1, x2). 
    # The order is different from np.pad().
    img0a = F.pad(img0a, (dx2, dx2, dy2, dy2))
    img1a = F.pad(img1a, (dx2, dx2, dy2, dy2))
    gta   = F.pad(gta,   (dx2, dx2, dy2, dy2))

    dxy = dxy.view(1, 4, 1, 1)

    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    # mask for the middle frame. Both directions have the same mask.
    mask = torch.zeros(mask_shape, device=img0.device, dtype=bool)
    mask[:, :, TM:BM, LM:RM] = True

    # merge flow_handler with aug_handler
    # s enumerates all scales.
    flow_a = []
    for s in np.arange(len(flow)):
        flow_a.append(flow[s] + dxy)
    flow_teacher_a = flow_teacher + dxy
    return img0a, img1a, gta, flow_a, flow_teacher_a, mask, dxy


def _hflip(img0, img1, gt, flow, flow_teacher):
    # B, C, H, W
    img0a = hflip(img0.clone())
    img1a = hflip(img1.clone())
    gta = hflip(gt.clone())
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    sxy = torch.tensor([ -1,  1, -1, 1], dtype=float, device=img0.device)
    sxy = sxy.view(1, 4, 1, 1)
    # merge flow_handler with aug_handler
    # s enumerates all scales.
    flow_a = []
    for s in np.arange(len(flow)):
        temp = hflip(flow[s])
        flow_a.append(temp * sxy)
    temp = hflip(flow_teacher)
    flow_teacher_a = temp * sxy
    return img0a, img1a, gta, flow_a, flow_teacher_a, mask


def _vflip(img0, img1, gt, flow, flow_teacher):
    # B, C, H, W
    img0a = vflip(img0.clone())
    img1a = vflip(img1.clone())
    gta = vflip(gt.clone())
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    sxy = torch.tensor([ 1, -1, 1, -1], dtype=float, device=img0.device)
    sxy = sxy.view(1, 4, 1, 1)
    # merge flow_handler with aug_handler
    # s enumerates all scales.
    flow_a = []
    for s in np.arange(len(flow)):
        temp = vflip(flow[s])
        flow_a.append(temp * sxy)
    temp = vflip(flow_teacher)
    flow_teacher_a = temp * sxy
    return img0a, img1a, gta, flow_a, flow_teacher_a, mask

def random_flip(img0, img1, gt, flow, flow_teacher, shift_sigmas=None):
    if random.random() > 0.5:
        img0a, img1a, gta, flow_a, flow_teacher_a, smask = _hflip(img0, img1, gt, flow, flow_teacher)
    else:
        img0a, img1a, gta, flow_a, flow_teacher_a, smask = _vflip(img0, img1, gt, flow, flow_teacher)
    return img0a, img1a, gta, flow_a, flow_teacher_a, smask, 0


def rotater(flow, R):
    """
    flow: B, C, H, W (16, 4, 224, 224) tensor
    R: (2, 2) rotation matrix, tensor
    """
    flow_fst, flow_sec = torch.split(flow, 2, dim=1)
    # flow map left multiply by rotation matrix R
    flow_fst_rot = torch.einsum('jc, bjhw -> bchw', R, flow_fst)
    flow_sec_rot = torch.einsum('jc, bjhw -> bchw', R, flow_sec)
    flow_rot = torch.cat((flow_fst_rot, flow_sec_rot), dim=1)
    # visualize_flow(flow_fst[0].permute(1, 2, 0), 'flow.png')
    # visualize_flow(flow_fst_rot[0].permute(1, 2, 0), 'flow_rotate.png')
    # print(flow_fst[0, :, 112, 112])
    # print(flow_fst_rot[0, :, 112, 112])
    return flow_rot


def random_rotate(img0, img1, gt, flow, flow_teacher, shift_sigmas=None):
    if random.random() < 1/3.:
        angle = 90
    elif random.random() < 2/3.:
        angle = 180
    else:
        angle = 270
    # The two dimensional rotation matrix R which rotates points in the xy plane
    # anti-clockwise through an angle θ about the origin
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], dtype=np.float32) # size(2, 2)
    # Pytorch rotate: center of rotation, default is the center of the image. 
    # angle: value in degrees, counter-clockwise.
    img0a = rotate(img0.clone(), angle=angle)
    img1a = rotate(img1.clone(), angle=angle)
    gta = rotate(gt.clone(), angle=angle)
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    R = torch.from_numpy(R).to(img0.device)
    # merge flow_handler with aug_handler
    # s enumerates all scales.
    flow_a = []
    for s in np.arange(len(flow)):
        temp = rotate(flow[s], angle=angle)
        temp = rotater(temp, R)
        flow_a.append(temp)
    flow_teacher_a = rotate(flow_teacher, angle=angle)
    flow_teacher_a = rotater(flow_teacher_a, R)
    return img0a, img1a, gta, flow_a, flow_teacher_a, mask, 0


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.
    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.
    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle).astype(bool)


def random_affine(img0, img1, gt, flow, flow_teacher, shift_sigmas=None):
    B, C, H, W = img0.shape
    # OpenCV uses 3 points to generate an affine matrix.
    # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    W_shift_ratio_low1 = np.random.uniform(0, 0.3)
    W_shift_ratio_low2 = np.random.uniform(0, 0.3)
    W_shift_ratio_high = np.random.uniform(0.7, 1)
    H_shift_ratio_low1 = np.random.uniform(0, 0.3)
    H_shift_ratio_low2 = np.random.uniform(0, 0.3)
    H_shift_ratio_high = np.random.uniform(0.7, 1)
    srcTri = np.array([[0, 0], [W - 1, 0], [0, H - 1]]).astype(np.float32) # 4th pt: [W-1, H-1]
    dstTri = np.array([[W*W_shift_ratio_low1, H*H_shift_ratio_low1], \
        [W*W_shift_ratio_high, H*H_shift_ratio_low2], \
        [W*W_shift_ratio_low2, H*H_shift_ratio_high]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri).astype(np.float32) # (2, 3) affine transformation matrix
    # transform 4 vertices to get new vertices of affine transformed image
    # new vertices are used to generate mask
    fourth_pt = np.matmul(warp_mat, np.array([W-1, H-1, 1]))
    fourth_pt[0] = np.clip(fourth_pt[0], 0, W)
    fourth_pt[1] = np.clip(fourth_pt[1], 0, H)
    polygon = np.ndarray((4, 2), dtype=np.float32)
    polygon[:2, :] = dstTri[:2, :]
    polygon[2, :] = fourth_pt
    polygon[3, :] = dstTri[-1, :]
    mask = polygons_to_mask([np.array(polygon, np.float32)], H, W)
    # cv2.imwrite('mask.png', mask.astype(float)*255)
    mask = torch.from_numpy(mask).to(img0.device).view(1, 1, H, W)
    mask = mask.repeat(B, 4, 1, 1)

    aff0 = np.empty((B, H, W, C), dtype=np.float32)
    aff1 = np.empty((B, H, W, C), dtype=np.float32)
    aff2 = np.empty((B, H, W, C), dtype=np.float32)
    img0_copy = img0.permute(0, 2, 3, 1).cpu().numpy() # B, H, W, C
    img1_copy = img1.permute(0, 2, 3, 1).cpu().numpy()
    gt_copy = gt.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(B):
        aff0[i] = cv2.warpAffine(img0_copy[i], warp_mat, (H, W))
        aff1[i] = cv2.warpAffine(img1_copy[i], warp_mat, (H, W))
        aff2[i] = cv2.warpAffine(gt_copy[i], warp_mat, (H, W))
    img0a = torch.from_numpy(aff0).permute(0, 3, 1, 2).to(img0.device)
    img1a = torch.from_numpy(aff1).permute(0, 3, 1, 2).to(img0.device)
    gta = torch.from_numpy(aff2).permute(0, 3, 1, 2).to(img0.device)
    # cv2.imwrite('affine0.png', img0a[0].permute(1, 2, 0).cpu().numpy()*255)
    R = torch.from_numpy(warp_mat[:, :2]).to(img0.device)
    flow_all_a = []
    flow_all = flow + [flow_teacher]
    for s in np.arange(len(flow_all)):
        aff4 = np.empty((B, H, W, 4), dtype=np.float32)
        flow_copy = flow_all[s].clone().detach().permute(0, 2, 3, 1).cpu().numpy() # B, H, W, 4
        for i in range(B):
            aff4[i] = cv2.warpAffine(flow_copy[i], warp_mat, (H, W))
        aff4 = torch.from_numpy(aff4).permute(0, 3, 1, 2).to(img0.device)
        # affine transform to flow value
        temp = rotater(aff4, R)
        flow_all_a.append(temp)
    flow_a = flow_all_a[:-1]
    flow_teacher_a = flow_all_a[-1]
    # visualize_flow(flow[0][0].permute(1, 2, 0), 'flow.png')
    # visualize_flow(flow_a[0][0].permute(1, 2, 0), 'flow_affine.png')
    return img0a, img1a, gta, flow_a, flow_teacher_a, mask, 0


def calculate_consist_loss(img0, img1, gt, flow, flow_teacher, shift_sigmas, aug_handler):
    assert(aug_handler is not None)
    img0a, img1a, gta, flow_a, flow_teacher_a, smask, dxy = aug_handler(img0, img1, gt, flow, flow_teacher, shift_sigmas)
    return img0a, img1a, gta, flow_a, flow_teacher_a, smask, dxy