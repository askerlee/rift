from numpy import imag
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate
from torchvision.transforms import ColorJitter
import cv2

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

colorjitter_fun = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)

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


# img0, img1, mid_gt are 4D tensors of (B, 3, 224, 224). mid_gt are the middle frames.
def random_shift(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=(16, 10)):
    B, C, H, W = img0.shape
    u_shift_sigma, v_shift_sigma = shift_sigmas
    
    # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    # Make sure at most one of dx, dy is large. Otherwise the shift is too difficult.
    MAX_SHIFT = 120
    dx, dy = 0, 0

    # Avoid (0, 0) offset. 1 becomes 0 after rounding.
    while abs(dx) <=1 and abs(dy) <=1:
        if np.random.random() > 0.5:
            dx = np.random.laplace(0, u_shift_sigma / 4)
            dy = np.random.laplace(0, v_shift_sigma)
            # Cap the shift in either direction to 40, to avoid abnormal gradients
            # Sometimes the model output becomes rubbish and would never recover. 
            # The reason may be occasional large shifts (could be > 70).
            # Especially for sofi, the shift is doubled, therefore could be catasrophic.
            dx = np.clip(dx, -MAX_SHIFT, MAX_SHIFT)
            dy = np.clip(dy, -MAX_SHIFT, MAX_SHIFT)
        else:
            dx = np.random.laplace(0, u_shift_sigma)
            dy = np.random.laplace(0, v_shift_sigma / 4)
            dx = np.clip(dx, -MAX_SHIFT, MAX_SHIFT)
            dy = np.clip(dy, -MAX_SHIFT, MAX_SHIFT)

    # Make sure dx and dy are even numbers.
    dx = (int(dx) // 2) * 2
    dy = (int(dy) // 2) * 2
    dx2 = dx // 2
    dy2 = dy // 2
    
    # If flow=0, pixels at (dy, dx)_0a <-> (0, 0)_1a.
    if dx >= 0 and dy >= 0:
        # img0 is cropped off the bottom-right corner.               img0[:-dy, :-dx]
        img0_bound = (0,  H - dy,  0,  W - dx)
        # img1 is shifted by (dx, dy) to the left and up (or equivalently, cropped off the top-left corner). 
        # pixels at (dy, dx) ->(0, 0).
        #                                                           img1[dy:,  dx:]
        img1_bound = (dy, H,       dx, W)
    if dx >= 0 and dy < 0:
        # img0 is cropped off the right side, and shifted to the up. img0[-dy:, :-dx]
        img0_bound = (-dy, H,      0,  W - dx)
        # img1 is shifted to the left and cropped at the bottom.    img1[:dy,  dx:]
        img1_bound = (0,   H + dy, dx, W)
        # (dx, 0)_0 => (dx, dy)_0a, (dx, 0)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.          
    if dx < 0 and dy >= 0:
        # img0 is shifted to the left, and cropped at the bottom.   img0[:-dy, -dx:]
        img0_bound = (0,   H - dy, -dx, W)
        # img1 is cropped off the right side, and shifted to the up. img1[dy:,  :dx]
        img1_bound = (dy,  H,      0,   W + dx)
        # (0, dy)_0 => (dx, dy)_0a, (0, dy)_1 => (0, 0)_1a.
        # So if flow=0, i.e., (dx, dy)_0 == (dx, dy)_1, then (dx, dy)_0a => (0, 0)_1a.         
    if dx < 0 and dy < 0:
        # img0 is shifted by (-dx, -dy) to the left and up. img0[-dy:, -dx:]
        img0_bound = (-dy, H,      -dx, W)
        # img1 is cropped off the bottom-right corner.       img1[:dy,  :dx]
        img1_bound = (0,   H + dy, 0,   W + dx)

    # dxy is the motion of the middle frame. It's always half of the relative motion between frames 0 and 1.
    # dxy: offsets (from old to new flow) for two directions.
    # Note the middle frame is shifted by *half* of dx, dy.
    # Note the flows are for backward warping (from middle to 0/1).
    # From 0.5 -> 0: positive delta (from the old flow). old 0.5->0 flow + (dx, dy) = new 0.5->0 flow.
    # From 0.5 -> 1: negative delta (from the old flow). old 0.5->1 flow - (dx, dy) = new 0.5->1 flow.
    dxy = torch.tensor([ dx2,  dy2, -dx2, -dy2], dtype=float, device=img0.device)

    # T, B, L, R: top, bottom, left, right boundary.
    T0, B0, L0, R0 = img0_bound
    T1, B1, L1, R1 = img1_bound
    # For the middle frame, the numbers of cropped pixels at the left and right, or the up and the bottom are equal.
    # Therefore, after padding, the middle frame doesn't shift. It's just cropped at the center and 
    # zero-padded at the four sides.
    # This property makes it easy to compare the flow before and after shifting.
    dx2, dy2 = abs(dx2), abs(dy2)
    # |T0-T1| = |B0-B1| = dy, |L0-L1| = |R0-R1| = dx.
    img0a = img0[:, :, T0:B0, L0:R0]
    img1a = img1[:, :, T1:B1, L1:R1]
    # TM, BM, LM, RM: new boundary (valid area) of the middle frame. 
    # The invalid boundary is half of the invalid boundary of img0 and img1.
    TM, BM, LM, RM = dy2, H - dy2, dx2, W - dx2
    mid_gta   = mid_gt[:, :, TM:BM, LM:RM]

    # Pad img0a, img1a, mid_gta by half of (dy, dx), to the original size.
    # Note the pads are ordered as (x1, x2, y1, y2) instead of (y1, y2, x1, x2). 
    # The order is different from np.pad().
    img0a   = F.pad(img0a,      (dx2, dx2, dy2, dy2), value=img0a.mean())
    img1a   = F.pad(img1a,      (dx2, dx2, dy2, dy2), value=img1a.mean())
    mid_gta = F.pad(mid_gta,    (dx2, dx2, dy2, dy2), value=mid_gta.mean())

    mask_shape      = list(img0.shape)
    mask_shape[1]   = 4   # For 4 flow channels of two directions (2 for each direction).
    # mask removes the padded zeros.
    # mask for the middle frame. Both directions have the same mask.
    # The same mask is used for flow_sofi_a.
    mask = torch.zeros(mask_shape, device=img0.device, dtype=bool)
    mask[:, :, TM:BM, LM:RM] = True

    dxy = dxy.view(1, 4, 1, 1)

    offset_dict = { 'dxy': dxy, 'img_bounds': [img0_bound, img1_bound], 'pad': [dx2, dy2] }
    flow_list_a = flow_shifter(flow_list, offset_dict, sofi_start_idx)
    return img0a, img1a, mid_gta, flow_list_a, mask, dxy
                
def random_flip(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    if np.random.random() > 0.5:
        FLIP_OP = hflip
        flip_direction = 'h'
    else:
        FLIP_OP = vflip
        flip_direction = 'v'

    # B, C, H, W
    img0a   = FLIP_OP(img0.clone())
    img1a   = FLIP_OP(img1.clone())
    mid_gta = FLIP_OP(mid_gt.clone())
    # mask has the same shape as the flipped image.
    mask_shape      = list(img0a.shape)
    mask_shape[1]   = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)

    flow_list_a = flow_flipper(flow_list, flip_direction)
    return img0a, img1a, mid_gta, flow_list_a, mask, flip_direction

def swap_frames(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    img0a, img1a = img1, img0
    mid_gta = mid_gt
    
    flow_list_a = []
    for flow in flow_list:
        # The treatment for sofi flows is the same as RIFT flows.
        if flow is not None:
            # For sofi flows, these are flow_10 and flow_01. They are swapped as well.
            flow_m0, flow_m1 = flow.split(2, dim=1)
            flow_a = torch.cat([flow_m1, flow_m0], dim=1)
            flow_list_a.append(flow_a)
        else:
            flow_list_a.append(None)

    # mask has the same shape as the flipped image.
    mask_shape      = list(img0a.shape)
    mask_shape[1]   = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)

    return img0a, img1a, mid_gta, flow_list_a, mask, 'swap'

def random_rotate(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    if np.random.random() < 1/3.:
        angle = 90
    elif np.random.random() < 2/3.:
        angle = 180
    else:
        angle = 270

    # angle: value in degrees, counter-clockwise.
    img0a   = rotate(img0.clone(),   angle=angle)
    img1a   = rotate(img1.clone(),   angle=angle)
    mid_gta = rotate(mid_gt.clone(), angle=angle)
    # mask has the same shape as the rotated image.
    mask_shape = list(img0a.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    # TODO: If images height != width, then rotation will crop images, and mask will contain 0s.
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)

    flow_list_a = flow_rotator(flow_list, angle)

    return img0a, img1a, mid_gta, flow_list_a, mask, angle

def bgr2rgb(img0, img1, mid_gt):
    img0a = img0[:, [2, 1, 0], :, :]
    img1a = img1[:, [2, 1, 0], :, :]

    if mid_gt.shape[1] == 3:
        mid_gta = mid_gt[:, [2, 1, 0], :, :]
    else:
        # mid_gt is an empty tensor.
        mid_gta = mid_gt

    return img0a, img1a, mid_gta

# B, C, H, W
def color_jitter(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    # A small probability to do individual jittering. 
    # More challenging, therefore smaller prob.
    asym_jitter_prob = 0.2
    same_aug = np.random.random() > asym_jitter_prob

    if same_aug:
        if mid_gt.shape[1] == 3:
            pseudo_batch = torch.cat([img0, img1, mid_gt], dim=0)
            pseudo_batch_a = colorjitter_fun(pseudo_batch)
            img0a, img1a, mid_gta   = torch.split(pseudo_batch_a, img0.shape[0], dim=0)
        else:
            # mid_gt is an empty tensor.
            pseudo_batch    = torch.cat([img0, img1], dim=0)
            pseudo_batch_a  = colorjitter_fun(pseudo_batch)
            img0a, img1a    = torch.split(pseudo_batch_a, img0.shape[0], dim=0)
            mid_gta = mid_gt
    else:
        img0a = colorjitter_fun(img0)
        img1a = colorjitter_fun(img1)

        if mid_gt.shape[1] == 3:
            mid_gta   = colorjitter_fun(mid_gt)
        else:
            mid_gta   = mid_gt

    bgr2rgb_prob = 0.4
    if torch.rand(1).item() < bgr2rgb_prob:
        img0a, img1a, mid_gta = bgr2rgb(img0a, img1a, mid_gta)

    # mask has the same shape as the flipped image.
    mask_shape = list(img0a.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)

    return img0a, img1a, mid_gta, flow_list, mask, 'j'

# B, C, H, W
def random_erase(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    # Randomly choose a rectangle region to erase.
    # Erased height/width is within this range.
    hw_bounds = [40, 80]

    ht, wd = img0.shape[2:]
    if np.random.rand() < 0.5:
        changed_img = img0.clone()
        changed_img_idx = 0
    else:
        changed_img = img1.clone()
        changed_img_idx = 1
    # mean_color: B, C
    mean_color = changed_img.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    erased_pixel_count = 0

    for _ in range(np.random.randint(2, 4)):
        x0 = np.random.randint(0, wd)
        y0 = np.random.randint(0, ht)
        dx = np.random.randint(hw_bounds[0], hw_bounds[1])
        dy = np.random.randint(hw_bounds[0], hw_bounds[1])
        changed_img[:, :, y0:y0+dy, x0:x0+dx] = mean_color
        # y0+dy, x0+dx may go out of bounds. Therefore erased pixel count may be < dx*dy.
        erased_pixel_count += changed_img[0, 0, y0:y0+dy, x0:x0+dx].numel()

    if changed_img_idx == 0:
        img0a = changed_img
        img1a = img1
        mid_gta = mid_gt
    else:
        img0a = img0
        img1a = changed_img
        mid_gta = mid_gt

    # mask has the same shape as the flipped image.
    mask_shape = list(img0a.shape)
    mask_shape[1] = 4   # For 4 flow channels of two directions (2 for each direction).
    mask = torch.ones(mask_shape, device=img0.device, dtype=bool)
    return img0a, img1a, mid_gta, flow_list, mask, erased_pixel_count

def random_scale(img0, img1, mid_gt, flow_list, sofi_start_idx, shift_sigmas=None):
    # Randomly choose a scale factor.
    # Scale factor is within this range.
    H, W   = img0.shape[2:]
    scale_bounds = [0.8, 1.2]
    scale_H, scale_W = np.random.uniform(scale_bounds[0], scale_bounds[1], 2)
    H2 = int(H * scale_H)
    W2 = int(W * scale_W)
    scale_H = H2 / H
    scale_W = W2 / W

    flow_list_notnone = [ f for f in flow_list if f is not None ]
    # flow_block: B*K, 4, H, W
    flow_block = torch.cat(flow_list_notnone, dim=0)

    # To be consistent with images, mask has 3 channels. But only 1 channel is really needed.
    mask = torch.ones(img0.shape, device=img0.device, dtype=img0.dtype)

    if mid_gt.shape[1] == 3:
        imgs = torch.cat([img0, img1, mid_gt, mask], dim=0)
    else:
        imgs = torch.cat([img0, img1, mask], dim=0)

    scaled_imgs         = F.interpolate(imgs,       size=(H2, W2), mode='bilinear', align_corners=False)
    scaled_flow_block   = F.interpolate(flow_block, size=(H2, W2), mode='bilinear', align_corners=False)

    if H2 < H or W2 < W:
        pad_h  = max(H - H2, 0)
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        pad_w  = max(W - W2, 0)
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1

        pads        = (pad_w1, pad_w2, pad_h1, pad_h2)
        scaled_imgs         = F.pad(scaled_imgs,        pads, "constant", 0)
        scaled_flow_block   = F.pad(scaled_flow_block,  pads, "constant", 0)

    # After padding, scaled_imgs are at least H*W.
    # Extra borders have to be cropped.
    H2, W2  = scaled_imgs.shape[2:] 
    h_start = np.random.randint(H2 - H + 1)
    h_end   = h_start + H
    w_start = np.random.randint(W2 - W + 1)
    w_end   = w_start + W
        
    scaled_imgs         = scaled_imgs[      :, :, h_start:h_end, w_start:w_end]
    scaled_flow_block   = scaled_flow_block[:, :, h_start:h_end, w_start:w_end]
    assert scaled_imgs.shape[2:] == (H, W)

    B = img0.shape[0]
    img0a, img1a = scaled_imgs[0:B], scaled_imgs[B:2*B]
    if mid_gt.shape[1] == 3:
        mid_gta = scaled_imgs[2*B:3*B]
    else:
        mid_gta = mid_gt

    # Padding and cropping doesn't change the flow magnitude. Only scaling does.
    # Scale the flow magnitudes accordingly. flow is (x, y, x, y), so (scale_W, scale_H, scale_W, scale_H).
    flow_block_a  = scaled_flow_block * \
                        torch.tensor([scale_W, scale_H, 
                                      scale_W, scale_H], device=img0.device).reshape(1, 4, 1, 1)

    flow_list_a_notnone = flow_block_a.split(B, dim=0)
    flow_list_a = []
    notnone_idx = 0
    for flow in flow_list:
        if flow is not None:
            flow_list_a.append(flow_list_a_notnone[notnone_idx])
            notnone_idx += 1
        else:
            flow_list_a.append(None)

    mask = scaled_imgs[-B:]
    # mask: B, 4, H, W. Same mask for the two directions.
    mask = mask[:, [0]].repeat(1, 4, 1, 1)
    # At padded areas of imgs, mask is also padded with zeros.
    # Therefore, convert float mask to bool mask by thresholding.
    mask = (mask >= 0.5)

    return img0a, img1a, mid_gta, flow_list_a, mask, [scale_H, scale_W]

def flow_shifter(flow_list, offset_dict, sofi_start_idx=-1):
    offset, img_bounds, pad_xy = offset_dict['dxy'], offset_dict['img_bounds'], offset_dict['pad']

    flow_list_a = []
    for i, flow in enumerate(flow_list):
        if flow is None:
            flow_list_a.append(None)
            continue
        if i >= sofi_start_idx:
            flow_sofi = flow
            if flow_sofi is not None:
                img0_bound, img1_bound = img_bounds
                # T, B, L, R: top, bottom, left, right boundary.
                T0, B0, L0, R0 = img0_bound
                T1, B1, L1, R1 = img1_bound
                dx2, dy2 = pad_xy
                flow10, flow01 = flow_sofi.split(2, dim=1)
                # Sofi flow is the relative motion between img0 and img1. Each image is shifted by offset
                # relative to the middle frame, so 2*offset here.
                # Sofi flow in both directions is cropped, as in the shifted images, some areas are shifted 
                # outside the original images. We have to shift the same areas of the sofi flow 
                # outside the original flow. But img0/img1 are shifted in opposite directions. 
                # So we do the shifting on the two directions separately.
                # flow10 is cropped in the same way as img1.
                flow10a = flow10[:, :, T1:B1, L1:R1]
                # flow01 is cropped in the same way as img0.
                flow01a = flow01[:, :, T0:B0, L0:R0]
                flow_sofi_a = torch.cat([flow10a, flow01a], dim=1)
                # offset: [1, 4, 1, 1]
                flow_sofi_a = flow_sofi_a + 2 * offset
                # flow_sofi_a is smaller than original images. Pad flow_sofi_a in the same way as img0a, img1a.
                flow_sofi_a = F.pad(flow_sofi_a, (dx2, dx2, dy2, dy2))
            else:
                flow_sofi_a = None
            # sofi 0<->1 flow should be shifted double as compared to middle -> 0/1 flow.
            flow_list_a.append(flow_sofi_a)
        else:
            # The middle flow doesn't need cropping, as the middle flow doesn't shift, 
            # but only changes by value (as img0 and img1 are shifted.)
            flow_a = flow + offset
            flow_list_a.append(flow_a)
    
    return flow_list_a

# flip_direction: 'h' or 'v'
def flow_flipper(flow_list, flip_direction):
    if flip_direction == 'h':
        # x-flow takes negative. y-flow doesn't change.
        sxy = torch.tensor([ -1,  1, -1, 1], dtype=float, device=flow_list[0].device)
        OP = hflip  
    elif flip_direction == 'v':
        # x-flow doesn't change. y-flow takes negative.
        sxy = torch.tensor([ 1, -1, 1, -1], dtype=float, device=flow_list[0].device)
        OP = vflip
    else:
        breakpoint()

    sxy = sxy.view(1, 4, 1, 1)

    flow_list_a = []
    for i, flow in enumerate(flow_list):
        if flow is None:
            flow_list_a.append(None)
            continue

        flow_flip = OP(flow)
        flow_flip_trans = flow_flip * sxy
        flow_list_a.append(flow_flip_trans)

    return flow_list_a

# angle: value in degrees, counter-clockwise.
def flow_rotator(flow_list, angle):
    # The two dimensional rotation matrix R which rotates points in the uv plane
    # radians: angle * pi / 180
    # Flow values should be transformed accordingly.
    theta = np.radians(angle)
    R0 = torch.tensor([[  np.cos(theta), -np.sin(theta) ],
                       [  np.sin(theta),  np.cos(theta) ]], 
                      dtype=flow_list[0].dtype, 
                      device=flow_list[0].device)
    # Repeat for the flow of two directions.
    R = torch.zeros(4, 4, dtype=flow_list[0].dtype, device=flow_list[0].device)
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

    flow_list_a = []
    for i, flow in enumerate(flow_list):
        if flow is None:
            flow_list_a.append(None)
            continue

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
        flow_list_a.append(flow_rot_trans)

    return flow_list_a

# flow_list include flow in all scales.
def calculate_consist_loss(model, img0, img1, mid_gt, flow_list, flow_teacher, sofi_flow_list, 
                           shift_sigmas, aug_handlers, aug_types, mixed_precision):
    # Original flow_list: 3 flows in 3 scales.
    num_rift_scales = len(flow_list)
    # Put sofi flows at the end, so that they can be indexed by sofi_start_idx .. end of list.
    flow_list = flow_list + [flow_teacher] + sofi_flow_list
    sofi_start_idx = num_rift_scales + 1

    img0a, img1a, mid_gta, flow_list_a = img0, img1, mid_gt, flow_list
    aug_descs = []

    mask_shape      = list(img0a.shape)
    mask_shape[1]   = 4   # For 4 flow channels of two directions (2 for each direction).
    smask           = torch.ones(mask_shape, device=img0.device, dtype=bool)    

    for aug_idx in range(len(aug_handlers)):
        aug_handler = aug_handlers[aug_idx]
        aug_type    = aug_types[aug_idx]
        # NO-OP, i.e., no augmentation in this iteration.
        if aug_handler is None:
            aug_descs.append("")
            continue

        # smask doesn't need to updated iteratively. The first MAX_WHOLE_IMG_AUG_COUNT aug_handlers are whole image augmentors,
        # and smask is always the all-one mask. Only the last aug_handler is part image augmentor, 
        # and smask is a nontrivial mask that needs to be kept.
        img0a, img1a, mid_gta, flow_list_a, smask, tidbit = \
                aug_handler(img0a, img1a, mid_gta, flow_list_a, sofi_start_idx, shift_sigmas)

        if aug_type == 'shift':
            dx, dy = tidbit.flatten().tolist()[:2]
            aug_desc = f"({dx:.0f},{dy:.0f})"
        elif aug_type == 'rotate':
            aug_desc = f"rot{tidbit}"
        elif aug_type == 'flip':
            aug_desc = f"{tidbit}flip"
        elif aug_type == 'jitter':
            aug_desc = 'jit'
        elif aug_type == 'erase':
            aug_desc = f"e{tidbit}"
        elif aug_type == "scale":
            aug_desc = f"{tidbit[0]:.2f}*{tidbit[1]:.2f}"
        else:
            # swap, ...
            aug_desc = tidbit
        
        aug_descs.append(aug_desc)

    # If all three augs are NO-OP, aug_desc = '---'.
    aug_desc = '-'.join(aug_descs)

    flow_list_a, flow_teacher_a, sofi_flow_list_a = flow_list_a[:num_rift_scales], flow_list_a[num_rift_scales], \
                                                    flow_list_a[sofi_start_idx:]
    imgsa = torch.cat((img0a, img1a), 1)            

    with autocast(enabled=mixed_precision):
        flow_list2, sofi_flow_list2, mask2, crude_img_list2, refined_img_list2, teacher_dict2, \
            loss_distill2 = model(imgsa, mid_gta, scale_list=[4, 2, 1])

    loss_consist_stu = 0
    # s enumerates all (middle frame flow) scales.
    # Should not compute loss on 0-1 flow, as the image shifting needs 
    # different transformation to the new flow, which involves too many 
    # intermediate variables, and may not worth the trouble.
    for s in range(num_rift_scales):
        loss_consist_stu += torch.abs(flow_list_a[s] - flow_list2[s])[smask].mean()

    if flow_teacher_a is not None:
        # gradient can both pass to the teacher (flow of original images) 
        # and the student (flow of the augmented images).
        # So that they can correct each other.
        flow_teacher2    = teacher_dict2['flow_teacher']
        loss_consist_tea = torch.abs(flow_teacher_a - flow_teacher2)[smask].mean()
    else:
        loss_consist_tea = 0

    # There is no None flow in sofi_flow_list_a.
    # only_consist_on_final_sofi_flow: the consistency loss is only applied to the final sofi flow.
    # If False, the consistency loss is applied to all sofi flows, and the performance is slightly worse.
    only_consist_on_final_sofi_flow = True
    if only_consist_on_final_sofi_flow:
        sofi_flow_consist_set = [-1]
    else:
        sofi_flow_consist_set = range(len(sofi_flow_list_a))
    num_sofi_flow_in_loss = 0
    loss_consist_sofi = 0
    for sofi_idx in sofi_flow_consist_set:
        loss_consist_sofi += torch.abs(sofi_flow_list_a[sofi_idx] - sofi_flow_list2[sofi_idx])[smask].mean()
        num_sofi_flow_in_loss += 1
    loss_consist = ((loss_consist_stu / num_rift_scales + loss_consist_sofi / num_sofi_flow_in_loss) / 2 + loss_consist_tea) / 2
        
    return loss_consist, loss_distill2, aug_desc
