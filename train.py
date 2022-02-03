import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from datetime import datetime

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")
timestamp = datetime.now().strftime("%m%d%H%M")
checkpoint_dir = f"checkpoints/{timestamp}"
local_rank = int(os.environ.get('LOCAL_RANK', 0))
if local_rank == 0:
    print("Model checkpoints will be saved to '%s'" %checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

def get_learning_rate(base_lr, step):
    M = base_lr # default: 3e-4
    # warmup. 0 -> 0.0001
    if step < 2000:
        mul = step / 2000.
        return M * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (M - M * 0.1) * mul + (M * 0.1)

# Only visualize the first two channels of flow_map_np.
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

# aug_shift_prob:  image shifting probability in the augmentation.
# cons_shift_prob: image shifting probability in the consistency loss computation.
def train(model, local_rank, base_lr, aug_shift_prob):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None

    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train', shift_prob=aug_shift_prob)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=6, pin_memory=False, num_workers=4)

    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for bi, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data.to(device, non_blocking=True) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(base_lr, step)
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/stu', info['loss_stu'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                tea_pred = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((tea_pred[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
                
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_stu:{:.4f}'.format(
                       epoch, bi+1, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_stu']), 
                       flush=True)

            step += 1
        nr_eval += 1

        model.save_model(checkpoint_dir, epoch, local_rank)
        if nr_eval % 1 == 0:
            evaluate(model, val_data, epoch, step, local_rank, writer_val)
          
        dist.barrier()

def evaluate(model, val_data, epoch, nr_eval, local_rank, writer_val):
    loss_stu_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()

    for i, data in enumerate(val_data):
        # scale images to [0, 1].
        data_gpu = data.cuda() / 255.
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            tea_pred = info['merged_tea']
        
        loss_stu_list.append(info['loss_stu'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((tea_pred[j] - gt[j]) * (tea_pred[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        tea_pred = (tea_pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(6):
                imgs = np.concatenate((tea_pred[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return

    psnr = np.array(psnr_list).mean()
    psnr_teacher = np.array(psnr_list_teacher).mean()
    loss_distill = np.array(loss_distill_list).mean()
    writer_val.add_scalar('psnr', psnr, nr_eval)
    writer_val.add_scalar('psnr_teacher', psnr_teacher, nr_eval)
    writer_val.flush()
    print('epoch:{}, iter:{}, psnr:{:.2f}, psnr_tea:{:.2f}, loss_distill:{:.2f}'.format( \
           epoch, nr_eval, psnr, psnr_teacher, loss_distill), 
           flush=True)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--cdecay', dest='conv_weight_decay', type=float, default=1e-3, 
                        help='weight decay for convolution layers (default: 1e-3)')
    parser.add_argument('--distill', dest='distill_loss_weight', type=float, default=0.015)
    parser.add_argument('--clip', default=0.1, type=float,
                        metavar='C', help='gradient clip to C (Set to -1 to disable)')
    parser.add_argument('--lr', dest='base_lr', default=3.5e-4, type=float)
    parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                        help='Output M groups of flow')
    parser.add_argument('--ctxmergeflow', dest='ctx_use_merged_flow', action='store_true', 
                        help='Use merged flow for contextnet.')
    parser.add_argument('--augshift', dest='aug_shift_prob', default=0, type=float,
                        help='Probability of shifting augmentation')
    parser.add_argument('--consshift', dest='cons_shift_prob', default=0, type=float,
                        help='Probability of shifting augmentation')

    args = parser.parse_args()
    args.multi = [ int(m) for m in args.multi.split(",") ]

    args.local_rank = local_rank
    if args.local_rank == 0:
        print(f"Args:\n{args}")

    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, distill_loss_weight=args.distill_loss_weight,
                  grad_clip=args.clip,
                  multi=args.multi,
                  ctx_use_merged_flow=args.ctx_use_merged_flow,
                  conv_weight_decay=args.conv_weight_decay,
                  cons_shift_prob=args.cons_shift_prob)

    train(model, args.local_rank, args.base_lr, args.aug_shift_prob)
        