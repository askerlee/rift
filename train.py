import os
import sys
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from datetime import datetime
from easydict import EasyDict as edict

from model.RIFT import RIFT, SOFI_Wrapper
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
    # linear warmup: 0 -> base_lr
    if step < 2000:
        mul = step / 2000.
        return M * mul
    else:
        # mul: begin: 1, midway: 0.5, end: 0.00025 -> extremely close to 0.
        mul = np.cos((step - 2000) / (args.total_epochs * args.steps_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        # returned lr: 0.9*base_lr * mul + 0.1*base_lr
        # begin: base_lr, midway: 0.55*base_lr, end: 0.1*base_lr.
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
def train(model, local_rank, base_lr, aug_shift_prob, shift_sigmas, aug_jitter_prob,
          esti_sofi=False, flow_train_stage=None, flow_val_stage=None, 
          flowprob=0, flowstartep=20):

    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None

    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train', aug_shift_prob=aug_shift_prob, shift_sigmas=shift_sigmas, aug_jitter_prob=aug_jitter_prob)
    if not args.debug:
        sampler = DistributedSampler(dataset)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    args.steps_per_epoch = train_loader.__len__()
    dataset_val = VimeoDataset('validation')
    val_loader = DataLoader(dataset_val, batch_size=6, pin_memory=False, num_workers=4)

    if args.flow_train_stage is not None:
        sys.path.append('../craft/core')
        import datasets
        # Disable shift aug implemented within the flow dataset, 
        # which needs flow groundtruth to work. 
        # Otherwise image1 and image2 are a shifted pair, 
        # and would be too difficult for sofi estimation.
        flow_args = edict({'stage': args.flow_train_stage, 'shift_aug_prob': 0,
                           'shift_sigmas': args.shift_sigmas, 'image_size': (224, 224),
                           'batch_size': args.batch_size, 'num_workers': 4, 'ddp': not args.debug
                           })

        flow_loader = datasets.fetch_dataloader(flow_args)
        if not args.debug:
            flow_loader.sampler.set_epoch(0)
        flow_epoch = 0
        flow_iter = iter(flow_loader)
    else:
        flow_iter = None

    print('training...')
    for epoch in range(args.total_epochs):
        if not args.debug:
            sampler.set_epoch(epoch)

        time_stamp = time.time()
        for bi, data in enumerate(train_loader):
            # Use flow data (no middle-frame, no flow gt) to train the model.
            # Note flowstartep is numbered from 0, the same as epoch.
            if (flow_iter is not None) and (epoch >= flowstartep) and (random.random() < flowprob):
                is_flow_iter = True
                try:
                    data_blob = next(flow_iter)
                    image1, image2, flow, valid = [x.cuda() for x in data_blob[:4]]
                except StopIteration:
                    flow_epoch += 1
                    if not args.debug:
                        flow_loader.sampler.set_epoch(flow_epoch)

                    flow_iter = iter(flow_loader)
                    data_blob = next(flow_iter)
                    image1, image2, flow, valid = [x.cuda() for x in data_blob[:4]]
                
                imgs    = torch.cat([image1, image2], dim=1) / 255.
                # Provide a 0-channel tensor as mid_gt, just to make the model happy.
                mid_gt  = imgs[:, :0]   
            # Use 3 frames to train the model.
            else:
                is_flow_iter = False
                data_gpu = data.to(device, non_blocking=True) / 255.
                imgs    = data_gpu[:, :6]
                mid_gt  = data_gpu[:, 6:9]

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            learning_rate = get_learning_rate(base_lr, step)
            pred, info = model.update(imgs, mid_gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/stu', info['loss_stu'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
                writer.add_scalar('loss/sofi', info['loss_sofi'], step)

            if step % 1000 == 1 and local_rank == 0 and not is_flow_iter:
                mid_gt = (mid_gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                tea_pred = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(2):
                    imgs = np.concatenate((tea_pred[i], pred[i], mid_gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
                
            if local_rank == 0:                
                if esti_sofi:
                    loss_sofi = f"{info['loss_sofi']:.4f}"
                else:
                    loss_sofi = "-"
            
                if is_flow_iter:
                    loss_stu  = '-     '
                    loss_dist = '-     '
                else:
                    loss_stu = f"{info['loss_stu']:.4f}"
                    loss_dist = f"{info['loss_distill']:.4f}"

                print(f"ep {epoch} {bi+1} time {data_time_interval:.2f}+{train_time_interval:.2f} "
                      f"stu {loss_stu} dist {loss_dist} sofi {loss_sofi} cons {info['loss_consist_str']}",
                      flush=True)

            time_stamp = time.time()
            step += 1
        nr_eval += 1

        model.save_model(checkpoint_dir, epoch, local_rank)
        if nr_eval % 1 == 0:
            evaluate(model, val_loader, epoch, step, local_rank, writer_val,
                     esti_sofi, flow_val_stage)
          
        dist.barrier()

def evaluate(model, val_loader, epoch, nr_eval, local_rank, writer_val, 
             esti_sofi=False, flow_val_stage=None):

    if local_rank != 0:
        return

    loss_stu_list       = []
    loss_distill_list   = []
    loss_tea_list       = []
    loss_sofi_list      = []
    psnr_list           = []
    psnr_teacher_list   = []
    psnr_sofi_crude0_list   = []
    psnr_sofi_crude1_list   = []
    psnr_sofi_refined0_list = []
    psnr_sofi_refined1_list = []
    time_stamp = time.time()

    for i, data in enumerate(val_loader):
        # scale images to [0, 1].
        data_gpu = data.cuda() / 255.
        imgs = data_gpu[:, :6]
        mid_gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info  = model.update(imgs, mid_gt, training=False)
            tea_pred    = info['merged_tea']
            crude_img0  = info['crude_img0']
            crude_img1  = info['crude_img1']
            refined_img0 = info['refined_img0']
            refined_img1 = info['refined_img1']

        loss_stu_list.append(info['loss_stu'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        loss_sofi_list.append(info['loss_sofi'].cpu().numpy())

        for j in range(mid_gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((mid_gt[j] - pred[j]) * (mid_gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((tea_pred[j] - mid_gt[j]) * (tea_pred[j] - mid_gt[j])).cpu().data)
            psnr_teacher_list.append(psnr)

            if esti_sofi:
                img0 = imgs[:, :3]
                img1 = imgs[:, 3:]
                psnr_crude_img0 = -10 * math.log10(torch.mean((crude_img0[j] - img0[j]) * (crude_img0[j] - img0[j])).cpu().data)
                psnr_crude_img1 = -10 * math.log10(torch.mean((crude_img1[j] - img1[j]) * (crude_img1[j] - img1[j])).cpu().data)
                psnr_refined_img0 = -10 * math.log10(torch.mean((refined_img0[j] - img0[j]) * (refined_img0[j] - img0[j])).cpu().data)
                psnr_refined_img1 = -10 * math.log10(torch.mean((refined_img1[j] - img1[j]) * (refined_img1[j] - img1[j])).cpu().data)
            else:
                psnr_crude_img0, psnr_refined_img0, psnr_crude_img1, psnr_refined_img1 = 0, 0, 0, 0

            psnr_sofi_crude0_list.append(psnr_crude_img0)
            psnr_sofi_crude1_list.append(psnr_crude_img1)
            psnr_sofi_refined0_list.append(psnr_refined_img0)
            psnr_sofi_refined1_list.append(psnr_refined_img1)

        mid_gt = (mid_gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        tea_pred = (tea_pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(6):
                imgs = np.concatenate((tea_pred[j], pred[j], mid_gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp

    psnr = np.array(psnr_list).mean()
    psnr_teacher = np.array(psnr_teacher_list).mean()
    loss_distill = np.array(loss_distill_list).mean()
    psnr_sofi_crude0    = np.array(psnr_sofi_crude0_list).mean()
    psnr_sofi_crude1    = np.array(psnr_sofi_crude1_list).mean()
    psnr_sofi_refined0  = np.array(psnr_sofi_refined0_list).mean()
    psnr_sofi_refined1  = np.array(psnr_sofi_refined1_list).mean()
    loss_sofi    = np.array(loss_sofi_list).mean()
    
    writer_val.add_scalar('psnr', psnr, nr_eval)
    writer_val.add_scalar('psnr_teacher', psnr_teacher, nr_eval)
    writer_val.flush()
    
    if esti_sofi:
        psnr_sofi = f"{psnr_sofi_crude0:.2f},{psnr_sofi_crude1:.2f}/{psnr_sofi_refined0:.2f},{psnr_sofi_refined1:.2f}"
    else:
        psnr_sofi = "-"

    print('ep {}, {}, stu {:.2f}, tea {:.2f} dstl {:.2f}, sofi {}'.format( \
          epoch, nr_eval, psnr, psnr_teacher, loss_distill, psnr_sofi),
          flush=True)

    if esti_sofi and (flow_val_stage is not None):
        sys.path.append('../craft')
        sys.path.append('../craft/core')        
        import evaluate
        sofi_wrapper = SOFI_Wrapper(model.flownet)
        if flow_val_stage == 'chairs':
            evaluate.validate_chairs(sofi_wrapper, 1)
        if flow_val_stage == 'things':
            evaluate.validate_things(sofi_wrapper, 1)        
        elif flow_val_stage == 'sintel':
            evaluate.validate_sintel(sofi_wrapper, 1)
        elif flow_val_stage == 'kitti':
            evaluate.validate_kitti(sofi_wrapper,  1)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sofi', dest='esti_sofi', action='store_true', 
                        help='Do SOFI estimation')
    parser.add_argument('--epoch', dest='total_epochs', default=500, type=int)
    parser.add_argument('--bs', dest='batch_size', default=16, type=int)
    parser.add_argument('--cp', type=str, default=None, help='Load checkpoint from this path')
    parser.add_argument('--flowts', dest='flow_train_stage', default=None, 
                        help="Which flow dataset to use for training")
    parser.add_argument('--flowvs', dest='flow_val_stage',   default=None, 
                        help="Which flow dataset to use for validation")
    parser.add_argument('--flowprob', type=float, default=0, 
                        help="Probability of using flow data")
    parser.add_argument('--flowstartep', type=int, default=20, 
                        help="The first epoch to begin using flow data")

    parser.add_argument('--decay', dest='weight_decay', type=float, default=1e-3, 
                        help='initial weight decay (default: 1e-3)')
    parser.add_argument('--distillweight', dest='distill_loss_weight', type=float, default=0.02)
    parser.add_argument('--clip', default=0.1, type=float,
                        metavar='C', help='gradient clip to C (Set to -1 to disable)')
    parser.add_argument('--lr', dest='base_lr', default=3e-4, type=float)
    parser.add_argument('--multi', dest='multi', default="8,8,4", type=str, metavar='M', 
                        help='Output M groups of flow')
    parser.add_argument('--augshiftprob', dest='aug_shift_prob', default=0, type=float,
                        help='Probability of shifting augmentation')
    parser.add_argument('--augjitterprob', dest='aug_jitter_prob', default=0.5, type=float,
                        help='Probability of color jittering augmentation (differnt from color jitter consistency loss)')

    parser.add_argument('--consshiftprob', dest='cons_shift_prob', default=0.2, type=float,
                        help='Probability of shifting consistency loss')
    parser.add_argument('--shiftsigmas', dest='shift_sigmas', default="24,16", type=str,
                        help='Stds of shifts for shifting consistency loss')
    parser.add_argument('--consflipprob', dest='cons_flip_prob', default=0.1, type=float,
                        help='Probability of flipping consistency loss')
    parser.add_argument('--consrotprob', dest='cons_rot_prob', default=0.1, type=float,
                        help='Probability of rotating consistency loss')
    parser.add_argument('--consjitterprob', dest='cons_jitter_prob', default=0.1, type=float,
                        help='Probability of color jitter consistency loss')
    parser.add_argument('--conseraseprob', dest='cons_erase_prob', default=0.3, type=float,
                        help='Probability of block erasing consistency loss')
    parser.add_argument('--consscaleprob', dest='cons_scale_prob', default=0.4, type=float,
                        help='Probability of scaling consistency loss')

    parser.add_argument('--consweight', dest='consist_loss_weight', default=0.02, type=float, 
                        help='Consistency loss weight.')
    parser.add_argument('--smoothweight', dest='smooth_loss_weight', default=0.02, type=float, 
                        help='Flow smooth loss weight.')

    # mixed_precision: not recommended. Using mixed precision will lead to nan.
    parser.add_argument('--mixed_precision', default=False, action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--debug', default=False, type=bool, 
                        help='When debug is true, do not use distributed launch')

    args = parser.parse_args()
    args.multi = [ int(m) for m in args.multi.split(",") ]
    args.shift_sigmas = [ int(s) for s in args.shift_sigmas.split(",") ]

    args.local_rank = local_rank
    if args.local_rank == 0:
        print(f"Args:\n{args}")

    if not args.debug:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    consistency_args = {
        'shift_sigmas': args.shift_sigmas,
        'shift_prob': args.cons_shift_prob,
        'flip_prob': args.cons_flip_prob,
        'rot_prob': args.cons_rot_prob,
        'jitter_prob': args.cons_jitter_prob,
        'erase_prob': args.cons_erase_prob,
        'scale_prob': args.cons_scale_prob,
        'consist_loss_weight': args.consist_loss_weight,
    }

    model = RIFT(args.local_rank, 
                  esti_sofi=args.esti_sofi,
                  grad_clip=args.clip,
                  distill_loss_weight=args.distill_loss_weight,
                  smooth_loss_weight=args.smooth_loss_weight,
                  multi=args.multi,
                  weight_decay=args.weight_decay,
                  consistency_args=consistency_args,
                  mixed_precision=args.mixed_precision,
                  debug=args.debug)
    if args.cp is not None:
        model.load_model(args.cp, 1)

    train(model, args.local_rank, args.base_lr, 
          args.aug_shift_prob, args.shift_sigmas, args.aug_jitter_prob,
          args.esti_sofi, args.flow_train_stage, args.flow_val_stage,
          args.flowprob, args.flowstartep)
        