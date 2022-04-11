from cv2 import ROTATE_90_CLOCKWISE
from numpy import imag
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate
import cv2


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
def random_shift(img0, img1, gt, shift_sigmas=(16, 10)):
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
    return img0a, img1a, gta, mask, dxy


def _hflip(img0, img1, gt):
    # B, C, H, W
    img0a = hflip(img0.clone())
    img1a = hflip(img1.clone())
    gta = hflip(gt.clone())
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    # temp0 = img0a[0].permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite('flip0.png', temp0*255)
    # temp1 = img0[0].permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite('0.png', temp1*255)
    return img0a, img1a, gta, mask, 'h'


def _vflip(img0, img1, gt):
    # B, C, H, W
    img0a = vflip(img0.clone())
    img1a = vflip(img1.clone())
    gta = vflip(gt.clone())
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    return img0a, img1a, gta, mask, 'v'

def random_flip(img0, img1, gt, shift_sigmas=None):
    if random.random() > 0.5:
        img0a, img1a, gta, smask, sxy = _hflip(img0, img1, gt)
    else:
        img0a, img1a, gta, smask, sxy = _vflip(img0, img1, gt)
    return img0a, img1a, gta, smask, sxy


def random_rotate(img0, img1, gt, shift_sigmas=None):
    if random.random() < 1/3.:
        angle = 90
    elif random.random() < 2/3.:
        angle = 180
    else:
        angle = 270

    # angle: value in degrees, counter-clockwise.
    img0a = rotate(img0.clone(), angle=angle)
    img1a = rotate(img1.clone(), angle=angle)
    gta   = rotate(gt.clone(),   angle=angle)
    mask_shape = list(img0.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    # TODO: If images height != width, then rotation will crop images, and mask will contain 0s.
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    return img0a, img1a, gta, mask, angle


def flow_adder(flow_list, flow_teacher, offset):
    flow_list2 = flow_list + [flow_teacher]
    flow_list2_a = []
    for flow in flow_list2:
        flow_a = flow + offset
        flow_list2_a.append(flow_a)
    
    flow_list_a, flow_teacher_a = flow_list2_a[:-1], flow_list2_a[-1]
    return flow_list_a, flow_teacher_a

# flip_direction: 'h' or 'v'
def flow_flipper(flow_list, flow_teacher, flip_direction):
    flow_list2 = flow_list + [flow_teacher]
    if flip_direction == 'h':
        sxy = torch.tensor([ -1,  1, -1, 1], dtype=float, device=flow_teacher.device)
        OP = hflip  
    elif flip_direction == 'v':
        sxy = torch.tensor([ 1, -1, 1, -1], dtype=float, device=flow_teacher.device)
        OP = vflip
    else:
        breakpoint()

    sxy = sxy.view(1, 4, 1, 1)

    flow_list2_a = []
    for flow in flow_list2:
        flow_flip = OP(flow)
        flow_flip_trans = flow_flip * sxy
        flow_list2_a.append(flow_flip_trans)

    flow_list_a, flow_teacher_a = flow_list2_a[:-1], flow_list2_a[-1]
    return flow_list_a, flow_teacher_a

# angle: value in degrees, counter-clockwise.
def flow_rotator(flow_list, flow_teacher, angle):
    flow_list2 = flow_list + [flow_teacher]
    # The two dimensional rotation matrix R which rotates points in the uv plane
    # radians: angle * pi / 180
    # Flow values should be transformed accordingly.
    theta = np.radians(angle)
    R0 = torch.tensor([[  np.cos(theta), -np.sin(theta) ],
                      [  np.sin(theta), np.cos(theta)  ]], 
                      dtype=flow_teacher.dtype, 
                      device=flow_teacher.device)
    # Repeat for the flow of two directions.
    R = torch.zeros(4, 4, dtype=flow_teacher.dtype, device=flow_teacher.device)
    R[:2, :2] = R0
    R[2:, 2:] = R0

    # WRONG:
    # angle = 90:  R = [[0, 1], [-1, 0]],  i.e., (u, v) => ( v, -u)
    # angle = 180: R = [[-1, 0], [0, -1]], i.e., (u, v) => (-u, -v)
    # angle = 270: R = [[0, -1], [1, 0]],  i.e., (u, v) => (-v,  u)
    # RIGHT:
    # angle = 90:  R = [[0, -1], [1, 0]],  i.e., (u, v) => ( -v, u)
    # angle = 180: R = [[-1, 0], [0, -1]], i.e., (u, v) => (-u, -v)
    # angle = 270: R = [[0, 1],  [-1, 0]], i.e., (u, v) => ( v, -u)
    # But why?    

    flow_list2_a = []
    for flow in flow_list2:
        # counter-clockwise through an angle θ about the origin
        flow_rot = rotate(flow, angle=angle)
        # Pytorch rotate: center of rotation, default is the center of the image.     
        # flow: B, C, H, W (16, 4, 224, 224) tensor
        # R: (4, 2) rotation matrix, tensor
        # flow map left multiplied by rotation matrix R
        flow_rot_trans = torch.einsum('jc, bjhw -> bchw', R, flow_rot)
        # visualize_flow(flow_fst[0].permute(1, 2, 0), 'flow.png')
        # visualize_flow(flow_fst_rot[0].permute(1, 2, 0), 'flow_rotate_90.png')
        # print(flow_fst[0, :, 0, 0])
        # print(flow_fst_rot[0, :, 0, 0])
        flow_list2_a.append(flow_rot_trans)

    flow_list_a, flow_teacher_a = flow_list2_a[:-1], flow_list2_a[-1]
    return flow_list_a, flow_teacher_a

# flow_list include flow in all scales.
def calculate_consist_loss(img0, img1, gt, flow_list, flow_teacher, model, shift_sigmas, aug_handler, flow_handler):
    img0a, img1a, gta, smask, tidbit = aug_handler(img0, img1, gt, shift_sigmas)

    if tidbit is not None:
        imgsa = torch.cat((img0a, img1a), 1)
        flow_list_a, flow_teacher_a = flow_handler(flow_list, flow_teacher, tidbit)
        flow_list2, mask2, merged_img_list2, flow_teacher2, merged_teacher2, loss_distill2 = model(torch.cat((imgsa, gta), 1), scale_list=[4, 2, 1])
        loss_consist_stu = 0
        # s enumerates all scales.
        loss_on_scales = np.arange(len(flow_list))
        for s in loss_on_scales:
            loss_consist_stu += torch.abs(flow_list_a[s] - flow_list2[s])[smask].mean()

        # gradient can both pass to the teacher (flow of original images) 
        # and the student (flow of the augmented images).
        # So that they can correct each other.
        loss_consist_tea = torch.abs(flow_teacher_a - flow_teacher2)[smask].mean()
        loss_consist = (loss_consist_stu / len(loss_on_scales) + loss_consist_tea) / 2
        if not isinstance(tidbit, str):
            if isinstance(tidbit, int):
                mean_tidbit = str(tidbit)
            else:
                mean_tidbit = torch.tensor(tidbit).abs().float().mean().item()
                mean_tidbit = f"{mean_tidbit:.2f}"
        else:
            mean_tidbit = tidbit
    else:
        loss_consist = 0
        mean_tidbit = 0
        loss_distill2 = 0

    return loss_consist, loss_distill2, mean_tidbit