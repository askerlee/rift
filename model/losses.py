import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.laplacian import LapLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
            
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss

# flow could have any channels.
# https://github.com/coolbeam/OIFlow/blob/main/utils/tools.py
def flow_smooth_delta(flow, if_second_order=False):
    def gradient(x):
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        D_dy = x[:, :, 1:] - x[:, :, :-1]
        return D_dx, D_dy

    dx, dy = gradient(flow)
    # dx2, dxdy = gradient(dx)
    # dydx, dy2 = gradient(dy)
    if if_second_order:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
    else:
        smooth_loss = dx.abs().mean() + dy.abs().mean()
    # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
    # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
    return smooth_loss

# flow should have 4 channels.
# https://github.com/coolbeam/OIFlow/blob/main/utils/tools.py
def edge_aware_smoothness_order1(img0, img1, flow, constant=1.0, weight_type='gauss', error_type='L1'):
    def weight_fn(x):
        if weight_type == 'gauss':
            y = x ** 2
        elif weight_type == 'exp':
            y = torch.abs(x)
        else:
            raise ValueError('')
        return y

    def gradient_xy(img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx, gy

    def gradweight_xy(img0, img1):
        img0_gx, img0_gy = gradient_xy(img0)
        img1_gx, img1_gy = gradient_xy(img1)

        img0_wx = torch.exp(-torch.mean(weight_fn(constant * img0_gx), 1, keepdim=True))
        img0_wy = torch.exp(-torch.mean(weight_fn(constant * img0_gy), 1, keepdim=True))
        img1_wx = torch.exp(-torch.mean(weight_fn(constant * img1_gx), 1, keepdim=True))
        img1_wy = torch.exp(-torch.mean(weight_fn(constant * img1_gy), 1, keepdim=True))
        
        # First  two flow channels: 1->0 flow. So use img1 weights.
        # Second two flow channels: 0->1 flow. So use img0 weights.
        # weights_x and weights_y are for x and y's spatial gradients, respectively.
        weights_x = torch.cat([img1_wx, img1_wx, img0_wx, img0_wx], dim=1)
        weights_y = torch.cat([img1_wy, img0_wy, img0_wy, img1_wy], dim=1)

        return weights_x, weights_y

    def error_fn(x):
        if error_type == 'L1':
            y = torch.abs(x)
        elif error_type == 'abs_robust':
            y = (torch.abs(x) + 0.01).pow(0.4)
        else:
            raise ValueError('')
        return y

    # The flow gradients along x, y axes, respectively. 
    # flow_gx, flow_gy have the same number of channels as flow.
    # No matter the flow is x- or y-flow, it should be smooth along both x and y axes.
    # I.e., a y-flow should also be smooth along x-axis, and x-flow should also be smooth along y-axis.
    flow_gx, flow_gy        = gradient_xy(flow)
    # weights_x, weights_y both have 4 channels, same as flow_gx and flow_gy (if the input flow has 4 channels).
    weights_x, weights_y    = gradweight_xy(img0, img1)

    smoothness_x = error_fn(flow_gx) * weights_x
    smoothness_y = error_fn(flow_gy) * weights_y
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


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

if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
