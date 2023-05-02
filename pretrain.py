import argparse
import os
import time
import re
import sys
sys.path.append(os.getcwd())
import skimage.morphology as morph

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
# from torch.optim.lr_scheduler import PolynomialLR
import cv2

import transforms
import utils
from conf import settings
#from dataset.camvid_lmdb import CamVid
from lr_scheduler import PolynomialLR, WarmUpLR, WarmUpWrapper
from dataloader import IterLoader
import test_aug
from losses import DiceLoss, WeightedLossWarpper
import sampler as _sampler
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Dice
import torch.nn.functional as F




def train(net, train_dataloader, val_loader, writer, args):

    root_path = os.path.dirname(os.path.abspath(__file__))

    ckpt_path = os.path.join(
        root_path, settings.CHECKPOINT_FOLDER, args.prefix + '_' + settings.TIME_NOW)
    # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_manager = utils.CheckPointManager(ckpt_path, max_keep_ckpts=5)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # iter_per_epoch = len(train_dataset) / args.b

    # max_iter = args.e * len(train_loader)
    total_iter = args.iter
    warmup_iter = int(args.iter * 0.1)

    train_scheduler = PolynomialLR(optimizer, total_iters=total_iter - warmup_iter, power=0.9, min_lr=args.min_lr)
    warmup_scheduler = WarmUpLR(optimizer, total_iters=warmup_iter)
    lr_schduler = WarmUpWrapper(warmuplr_scheduler=warmup_scheduler, lr_scheduler=train_scheduler)


    gland_loss_fn_ce = nn.CrossEntropyLoss(ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    gland_loss_fn_dice = DiceLoss(ignore_index=train_dataloader.dataset.ignore_index, reduction='none')

    #cnt_weight = torch.tensor([0.52756701, 9.568812]).cuda()
    cnt_weight = None
    cnt_loss_fn_ce = nn.CrossEntropyLoss(weight=cnt_weight, ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    cnt_loss_fn_dice = DiceLoss(class_weight=cnt_weight, ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    #loss_l2 = nn.MSELoss()
    #loss_seg = SegmentLevelLoss(op=args.op)
    sampler = _sampler.OHEMPixelSampler(ignore_index=train_dataloader.dataset.ignore_index)
    cnt_loss_fn_ce = WeightedLossWarpper(cnt_loss_fn_ce, sampler)
    cnt_loss_fn_dice = WeightedLossWarpper(cnt_loss_fn_dice, sampler)
    gland_loss_fn_ce = WeightedLossWarpper(gland_loss_fn_ce, sampler)
    gland_loss_fn_dice = WeightedLossWarpper(gland_loss_fn_dice, sampler)
    #loss_fn_ce = WeightedLossWarpper(loss_fn_ce, sampler)
    #loss_fn_dice = WeightedLossWarpper(loss_fn_dice, sampler)


    #batch_start = time.time()
    # train_start = time.time()
    #total_load_time = 0
    train_iterloader = IterLoader(train_dataloader)
    #for batch_idx, (images, masks) in enumerate(train_loader):
    net.train()


    scaler = GradScaler()

    #total_metrics = ['total_F1', 'total_Dice', 'total_Haus']
    #testA_metrics = ['testA_F1', 'testA_Dice', 'testA_Haus']
    #testB_metrics = ['testB_F1', 'testB_Dice', 'testB_Haus']
    metrics = ['iou']

    #best = {
    #    'testA_F1': 0,
    #    'testA_Dice': 0,
    #    'testA_Haus': 9999,
    #    'testB_F1': 0,
    #    'testB_Dice': 0,
    #    'testB_Haus': 9999,
    #    'total_F1': 0,
    #    'total_Dice': 0,
    #    'total_Haus': 9999,
    #}
    #with torch.profiler.profile(
    #    activities=[
    #        torch.profiler.ProfilerActivity.CPU,
    #        torch.profiler.ProfilerActivity.CUDA],
    #    schedule=torch.profiler.schedule(
    #        wait=1,
    #        warmup=1,
    #        active=2),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
    #    # record_shapes=True,
    #    # profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    #    with_stack=True
    #) as p:
    #images, masks = next(train_iterloader)
    train_t = time.time()
    #for iter_idx, (images, masks, weight_maps) in enumerate(train_iterloader):
    for iter_idx, (images, masks) in enumerate(train_iterloader):
        # images, masks = images, masks

    # for iter_idx in range(1000000):
        # iter_start = time.time()

        data_time = time.time() - train_t

        # eval_start = time.time()
            # total = time.time() - batch_start
            # print(epoch, time.time() - batch_start)
            # print(total / (batch_idx + 1))
            # continue

        # for batch_idx, images in enumerate(train_loader):

        # images =

        if args.gpu:
            images = images.cuda()
            #weight_maps = weight_maps.cuda()
            masks = masks.cuda()

        optimizer.zero_grad()
        if args.fp16:
            with autocast():

                gland_preds, aux_preds, _ = net(images)
                loss = gland_loss_fn_ce(gland_preds, masks) + \
                                3 * gland_loss_fn_dice(gland_preds, masks)

                aux_loss = gland_loss_fn_ce(aux_preds, masks) + \
                                3 * gland_loss_fn_dice(aux_preds, masks)
                loss = loss + aux_loss
                loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = net(images)
            loss_ce = loss_fn_ce(preds, masks)
            loss_ce = loss_ce.mean()
            loss_dice = loss_fn_dice(preds, masks)
            loss = 3 * loss_dice + 1 * loss_ce
            scaler.scale(loss).backward()
            loss.backward()
            optimizer.step()


        lr_schduler.step(iter_idx)

        #if iter_idx > 1000:
            #break
        # bs = len(images)
        del images
        del masks


        iter_time = time.time() - train_t

        if (iter_idx + 1) % 50 == 0:
            print(('Training Iter: [{iter}/{total_iter}] '
                    'Lr:{lr:0.6f} Loss:{loss:0.4f}, Iter time:{iter_time:0.4f}s Data loading time:{data_time:0.4f}s').format(
                loss=loss.item(),
                iter=(iter_idx+1),
                total_iter = total_iter,
                lr=optimizer.param_groups[0]['lr'],
                iter_time=iter_time,
                data_time=data_time,
            ))

            # visualize
            utils.visualize_lastlayer(writer, net, iter_idx)
            utils.visualize_scalar(writer, 'loss', loss, iter_idx)
            utils.visualize_scalar(writer,
                    'learning rate',
                    optimizer.param_groups[0]['lr'],
                    iter_idx)




        # print log
        #if args.eval_iter % (iter_idx + 1) == 0:
        # print(args.eval_iter)
        if (iter_idx + 1) % args.eval_iter == 0:

            # evaluate()


            net.eval()
            print('evaluating.........')
            acc, dice = evaluate(net, val_loader, args)
            print(
                 'acc: {:.04f}'.format(acc),
                 'dice: {:.04f}'.format(dice),
            )


            #print(best)

            utils.visualize_metric(writer,
                ['acc', 'dice'], [acc, dice],iter_idx)



            utils.visualize_param_hist(writer, net, iter_idx)

            #if args.gpu:
            #    print('GPU INFO.....')
            #    print(torch.cuda.memory_summary(), end='')

            #finish = time.time()
            #total_training = finish - start
            #print(('Total time for training epoch {} : {:.2f}s, '
            #       'total time for loading data: {:.2f}s, '
            #       '{:.2f}% time used for loading data').format(
            #    iter_idx,
            #    total_training,
            #    total_load_time,
            #    total_load_time / total_training * 100
            #))

            ckpt_manager.save_ckp_iter(net, iter_idx)
            ckpt_manager.save_best(net, ['acc', 'dice'], [acc, dice], iter_idx)
            #ckpt_manager.save_best(net, testA_metrics,
            #    testA, iter_idx)
            #ckpt_manager.save_best(net, testB_metrics,
            #    testB, iter_idx)

            #print('best value:', ckpt_manager.best_value)
            net.train()


        train_t = time.time()

        if total_iter <= iter_idx:
            break


def evaluate(net, val_dataloader, args):
    net.eval()
        # test_loss = 0.0
    valid_dataset = val_dataloader.dataset
    #cls_names = valid_dataset.class_names
    #ig_idx = valid_dataset.ignore_index
    #count = 0
    #sampler = _sampler.OHEMPixelSampler(ignore_index=val_dataloader.dataset.ignore_index, min_kept=10000)
    metrics_inside = 0
    count = 0

    acc_metric = BinaryAccuracy(multidim_average='samplewise')
    dice_metric = Dice(mdmc_average='samplewise', num_classes=2)

    total_acc = 0
    total_dice = 0
    val_dataloader.dataset.sample_val()

    with torch.no_grad():
        for (imgs, masks) in val_dataloader:

            imgs = imgs.cuda()
            masks = masks.cuda()
            preds = net(imgs)
            dice_metric.to(imgs.device)

            preds = F.softmax(preds, dim=1)
            preds = preds.argmax(dim=1)
            # pred = pred.argmax(dim=1)

            acc = acc_metric(preds, masks)
            dice = dice_metric(preds, masks)
            #print(acc.shape, dice.shape)
            # print(acc, dice)
            total_acc += acc.sum()
            total_dice += dice * len(imgs)


    total_acc = total_acc / len(val_dataloader.dataset)
    total_dice = total_dice / len(val_dataloader.dataset)

    return total_acc.item(), total_dice.item()













if __name__ == '__main__':

    # from gpustats import GPUStats
    # gpu_stats = GPUStats(gpus_needed=1, sleep_time=10, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
    # gpu_stats.run()

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.007,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
    parser.add_argument('-iter', type=int, default=40000, help='training epoches')
    parser.add_argument('-eval_iter', type=int, default=2000, help='training epoches')
    parser.add_argument('-wd', type=float, default=0, help='training epoches')
    parser.add_argument('-resume', type=bool, default=False, help='if resume training')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-prefix', type=str, default='', help='checkpoint and runs folder prefix')
    parser.add_argument('-alpha', type=float, default=1, help='panalize parameter')
    parser.add_argument('-op', type=str, default='or', help='mask operation')
    parser.add_argument('-download', action='store_true', default=False,
        help='whether to download camvid dataset')
    parser.add_argument('-gpu', action='store_true', default=False, help='whether to use gpu')
    parser.add_argument('-baseline', action='store_true', default=False, help='base line')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
    parser.add_argument('-poly', action='store_true', default=False, help='poly decay')
    parser.add_argument('-min_lr', type=float, default=1e-4, help='min_lr for poly')
    parser.add_argument('-branch', type=str, default='hybird', help='dataset name')
    parser.add_argument('-fp16', action='store_true', default=False, help='whether to use mixed precision training')
    parser.add_argument('-wait', action='store_true', default=False, help='whether to wait until there is gpu aviliable')
    args = parser.parse_args()

    if args.wait:
        from gpustats import GPUStats
        gpu_stats = GPUStats(gpus_needed=1, sleep_time=60 * 5, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
        gpu_stats.run()

    root_path = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(
        root_path, settings.CHECKPOINT_FOLDER, args.prefix + '_' + settings.TIME_NOW)
    log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    print('saving tensorboard log into {}'.format(log_dir))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    print()

    train_loader = utils.data_loader(args, 'train')
    train_dataset = train_loader.dataset

    val_loader = utils.data_loader(args, 'val')

    #for img, masks in val_loader:
        #print(img.shape, masks.shape)

    # output = next(iter(val_loader))
    # print(type(output), len(output))

    # import sys; sys.exit()

    net = utils.get_model(args.net, 3, train_dataset.class_num, args=args)

    #ckpt_path = 'crctp/latest.pth'
    # crctp
    # ckpt_path = '/data/hdd1/by/mmclassification/work_dirs/gland/latest.pth'


    # print('Loading pretrained checkpoint from {}'.format(ckpt_path))
    # new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path)['state_dict'])
    # net.load_state_dict(new_state_dict)
    # print('Done!')

    if args.resume:
        weight_path = utils.get_weight_path(
            os.path.join(root_path, settings.CHECKPOINT_FOLDER))
        print('Loading weight file: {}...'.format(weight_path))
        net.load_state_dict(torch.load(weight_path))
        print('Done loading!')

    if args.gpu:
        net = net.cuda()

    tensor = torch.Tensor(1, 3, 480, 480)
    utils.visualize_network(writer, net, tensor)

    train(net, train_loader, val_loader, writer, args)
    # evaluate(net, val_loader, writer, args)
