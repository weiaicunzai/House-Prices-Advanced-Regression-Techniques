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
# from dataset.voc2012 import VOC2012Aug
from dataloader import IterLoader
import test_aug
from losses import DiceLoss, WeightedLossWarpper, GlandContrastLoss
import sampler as _sampler



from ignite.contrib.handlers import ProgressBar
from ignite.engine.engine import Engine
from ignite.handlers import FastaiLRFinder






def update_model(engine, batch):
    images, masks, weight_maps = batch

    if args.gpu:
            images = images.cuda()
            weight_maps = weight_maps.cuda()
            masks = masks.cuda()

    optimizer.zero_grad()
    if args.fp16:
            with autocast():

                gland_preds, aux_preds, out = net(images)

                loss = gland_loss_fn_ce(gland_preds, masks) + \
                                3 * gland_loss_fn_dice(gland_preds, masks)

                loss_aux = gland_loss_fn_ce(aux_preds, masks) + \
                                3 * gland_loss_fn_dice(aux_preds, masks)

                weight_maps = weight_maps.float().div(20)
                loss = loss * weight_maps + 0.4 * loss_aux * weight_maps




                contrasive_loss, xor_mask = contrasive_loss_fn(out, gland_preds, masks, queue=net.queue, queue_ptr=net.queue_ptr, fcs=net.fcs)

                sup_loss = (loss + xor_mask * loss).mean()
                loss = sup_loss + args.alpha * contrasive_loss


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return loss.item()




if __name__ == '__main__':


    # from gpustats import GPUStats
    # gpu_stats = GPUStats(gpus_needed=1, sleep_time=60 * 5, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
    # gpu_stats.run()


    print('cccccccccccccccc')
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Glas', help='dataset name')
    parser.add_argument('-alpha', type=float, default=0.1, help='panalize parameter')
    parser.add_argument('-gpu', action='store_true', default=False, help='whether to use gpu')
    parser.add_argument('-fp16', action='store_true', default=False, help='whether to use mixed precision training')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
    parser.add_argument('-download', action='store_true', default=False,
        help='whether to download camvid dataset')

    args = parser.parse_args()
    print('cccccccccccccccc')
    print(args)

    root_path = os.path.dirname(os.path.abspath(__file__))

    print()

    train_dataloader = utils.data_loader(args, 'train')
    train_dataset = train_dataloader.dataset
    # train_dataset.times = 3

    # val_loader = utils.data_loader(args, 'val')

    #net = utils.get_model('tgt', 3, train_dataset.class_num, args=args)
    net = utils.get_model('tg', 3, train_dataset.class_num, args=args)


    # eish + crag + glas
    ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_25_March_2023_23h_38m_31s/iter_39999.pt'

    print('Loading pretrained checkpoint from {}'.format(ckpt_path))
    #new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path)['state_dict'])
    new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path))
    net.load_state_dict(new_state_dict)
    print('Done!')

    if args.gpu:
        net = net.cuda()

    net.train()

    # train(net, train_loader, val_loader, writer, args)
    # evaluate(net, val_loader, writer, args)

    optimizer = optim.SGD(net.parameters(), lr=1e-07, momentum=0.9, weight_decay=5e-4)

    gland_loss_fn_ce = nn.CrossEntropyLoss(ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    gland_loss_fn_dice = DiceLoss(ignore_index=train_dataloader.dataset.ignore_index, reduction='none')

    #cnt_weight = torch.tensor([0.52756701, 9.568812]).cuda()
    cnt_weight = None
    cnt_loss_fn_ce = nn.CrossEntropyLoss(weight=cnt_weight, ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    cnt_loss_fn_dice = DiceLoss(class_weight=cnt_weight, ignore_index=train_dataloader.dataset.ignore_index, reduction='none')
    #var_loss_fn = LossVariance()

    contrasive_loss_fn = GlandContrastLoss(8, ignore_idx=train_dataloader.dataset.ignore_index, temperature=0.07)

    #loss_l2 = nn.MSELoss()
    #loss_seg = SegmentLevelLoss(op=args.op)
    sampler = _sampler.OHEMPixelSampler(ignore_index=train_dataloader.dataset.ignore_index)
    #cnt_loss_fn_ce = WeightedLossWarpper(cnt_loss_fn_ce, sampler)
    #cnt_loss_fn_dice = WeightedLossWarpper(cnt_loss_fn_dice, sampler)
    gland_loss_fn_ce = WeightedLossWarpper(gland_loss_fn_ce, sampler)
    gland_loss_fn_dice = WeightedLossWarpper(gland_loss_fn_dice, sampler)

    train_iterloader = IterLoader(train_dataloader)
    net.train()


    scaler = GradScaler()



    trainer = Engine(update_model)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})
    lr_finder = FastaiLRFinder()
    to_save={'model': net, 'optimizer': optimizer}
    print(trainer.state.epoch_length, trainer.state.max_epochs)
    # trainer.load_state_dict({
    #     'epoch_length':1,
    #     'max_epochs':1
    # })

    with lr_finder.attach(trainer, to_save, start_lr=1e-7, num_iter=200) as trainer_with_lr_finder:
        print(type(trainer_with_lr_finder))
        trainer_with_lr_finder.run(train_dataloader, max_epochs=1, epoch_length=1)

    print("Suggested LR", lr_finder.lr_suggestion())

    ax = lr_finder.plot()
    ax.figure.savefig("output.jpg")
