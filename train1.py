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
from lr_scheduler import PolynomialLR, WarmUpLR, WarmUpWrapper
from metric import eval_metrics, gland_accuracy_object_level
from dataloader import IterLoader
import test_aug
from losses import DiceLoss, WeightedLossWarpper, GlandContrastLoss, TI_Loss
import sampler as _sampler



def train(net, train_dataloader, val_loader, writer, args, val_set):

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

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
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
    #var_loss_fn = LossVariance()

    #contrasive_loss_fn = GlandContrastLoss(8, ignore_idx=train_dataloader.dataset.ignore_index, temperature=0.07)
    contrasive_loss_fn = GlandContrastLoss(4, ignore_idx=train_dataloader.dataset.ignore_index, temperature=0.07)

    #loss_l2 = nn.MSELoss()
    #loss_seg = SegmentLevelLoss(op=args.op)
    sampler = _sampler.OHEMPixelSampler(ignore_index=train_dataloader.dataset.ignore_index)
    #cnt_loss_fn_ce = WeightedLossWarpper(cnt_loss_fn_ce, sampler)
    #cnt_loss_fn_dice = WeightedLossWarpper(cnt_loss_fn_dice, sampler)
    gland_loss_fn_ce = WeightedLossWarpper(gland_loss_fn_ce, sampler)
    gland_loss_fn_dice = WeightedLossWarpper(gland_loss_fn_dice, sampler)
    #loss_fn_ce = WeightedLossWarpper(loss_fn_ce, sampler)
    #loss_fn_dice = WeightedLossWarpper(loss_fn_dice, sampler)

    ti_loss_weight = 1e-4
        #ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1,2]], exclusion=[[2,3],[3,4]])
    ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1,2]], exclusion=[])
    # ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
    # ti_loss_value = ti_loss_weight * ti_loss_value


    #batch_start = time.time()
    # train_start = time.time()
    #total_load_time = 0
    train_iterloader = IterLoader(train_dataloader)
    #for batch_idx, (images, masks) in enumerate(train_loader):
    net.train()


    scaler = GradScaler()

    total_metrics = ['total_F1', 'total_Dice', 'total_Haus']
    testA_metrics = ['testA_F1', 'testA_Dice', 'testA_Haus']
    testB_metrics = ['testB_F1', 'testB_Dice', 'testB_Haus']

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

    vis_idx = 5
    train_t = time.time()
    for iter_idx, (images, masks, weight_maps) in enumerate(train_iterloader):


        if args.vis and iter_idx % vis_idx == 0:

            for idx, ii in enumerate(images.clone()):
                ii = utils.to_img(ii.cpu()).numpy()
                ii = cv2.resize(ii, (0, 0), fx=0.5, fy=0.5)
                ii = cv2.imwrite('tmp/img_{}_{}.jpg'.format(iter_idx, idx), ii)


            for idx, ii in enumerate(masks.clone()):
            #for idx, ii in enumerate(masks):
                #print(torch.unique(ii))
                ii = ii.cpu().numpy()
                mm = ii == 255
                #ii[i==255] = 0 ### ?????
                ii[mm] = 0
                ii = ii / (ii.max() + 1e-7) * 255
                ii = cv2.imwrite('tmp/gt_{}_{}.png'.format(iter_idx, idx), ii.astype('uint8'))
                #print(idx + iter_idx * masks.shape[0])
                #ii = cv2.imwrite('tmp/gt_{}.png'.format(idx + iter_idx * masks.shape[0]), ii.astype('uint8'))



        #print(masks.max(), 'xiansu??', images.shape)



        data_time = time.time() - train_t

        if args.gpu:
            images = images.cuda()
            weight_maps = weight_maps.cuda()
            masks = masks.cuda()

        optimizer.zero_grad()
        if args.fp16:
            with autocast():

                #######  two branches
                #gland_preds, cnt_preds = net(images)
                #gland_masks = torch.zeros(size=masks.shape, device=images.device, dtype=masks.dtype)
                #cnt_masks = torch.zeros(size=masks.shape, device=images.device, dtype=masks.dtype)
                #gland_masks[masks==1] = 1
                #gland_masks[masks==255] = 255 # ignore_idx
                #cnt_masks[masks==2] = 1
                #cnt_masks[masks==255] = 255 # ignore _idx
                #loss_gland = gland_loss_fn_ce(gland_preds, gland_masks) + \
                #                3 * gland_loss_fn_dice(gland_preds, gland_masks)
                #loss_cnt = cnt_loss_fn_ce(cnt_preds, cnt_masks) + \
                #                3 * cnt_loss_fn_dice(cnt_preds, cnt_masks)
                #loss = loss_gland + loss_cnt
                #loss = loss.mean()
                #######  two branches

                gland_preds, aux_preds, out = net(images)

                #var_loss = var_loss_fn(gland_preds, masks)
                loss = gland_loss_fn_ce(gland_preds, masks) + \
                                3 * gland_loss_fn_dice(gland_preds, masks)

                loss_aux = gland_loss_fn_ce(aux_preds, masks) + \
                                3 * gland_loss_fn_dice(aux_preds, masks)

                weight_maps = weight_maps.float().div(20)
                #print(weight_maps.max(),  weight_maps.min())
                #print(loss.shape, weight_maps.shape)
                #print(weight_maps.mean())
                #mask = contrasive_loss(gland_preds, masks)

                #for g_p, m in zip(gland_preds, masks):
                y = torch.stack([masks==i for i in range(masks.max()+1)], dim=1).long()
                # print(y.shape, y[3, :, 4, 4])



                ti_loss_value = ti_loss_func(gland_preds, y) if ti_loss_weight != 0 else 0
                ti_loss_value = ti_loss_weight * ti_loss_value
                # print(ti_loss_value)


                loss = loss * weight_maps + 0.4 * loss_aux * weight_maps + ti_loss_value

                # if args.net == 'tg':
                loss = loss.mean()

                # if args.net == 'tgt':

                    # contrasive_loss, xor_mask = contrasive_loss_fn(out, gland_preds, masks, queue=net.queue, queue_ptr=net.queue_ptr, fcs=net.fcs)

                    # sup_loss = (loss + xor_mask * loss).mean()
                    # loss = sup_loss + args.alpha * contrasive_loss


                #loss = loss.mean()

                if args.vis and iter_idx % vis_idx == 0:

                    print('save images....')
                    # for idx, ii in enumerate(images.clone()):
                    #     # ii = ii.permute(1, 2, 0)
                    #     # ii = ii.argmax(dim=2)
                    #     #print(ii.shape)
                    #     #print(ii.max())
                    #     #print(masks[idx].max())
                    #     # mm = masks[idx] == 255
                    #     #cv2.imwrite('tmp/pred_c_{}.png'.format(idx), mm.cpu().numpy().astype('uint8') * 255)
                    #     #print(mm.shape)
                    #     # ii = ii * 255
                    #     # ii[mm] = 0
                    #     #ii = ii / (ii.max() + 1e-7) * 255
                    #     #print(ii.max())
                    #     ii = cv2.imwrite('tmp/img_{}_{}.jpg'.format(iter_idx, idx), ii.cpu().numpy())

                    print('save gland_preds.....')
                    for idx, ii in enumerate(gland_preds.clone()):
                        ii = ii.permute(1, 2, 0)
                        ii = ii.argmax(dim=2)
                        #print(ii.shape)
                        #print(ii.max())
                        #print(masks[idx].max())
                        mm = masks[idx] == 255
                        #cv2.imwrite('tmp/pred_c_{}.png'.format(idx), mm.cpu().numpy().astype('uint8') * 255)
                        #print(mm.shape)
                        ii = ii * 255
                        ii[mm] = 0
                        #ii = ii / (ii.max() + 1e-7) * 255
                        #print(ii.max())
                        ii = cv2.imwrite('tmp/pred_{}_{}.png'.format(iter_idx, idx), ii.cpu().numpy())

                    for idx, ii in enumerate(gland_preds.clone()):
                        ii = ii.permute(1, 2, 0)
                        ii = ii.softmax(dim=2)
                        #print(ii[3, 4])
                        ii_min = ii.min(dim=2)[0]
                        # print(ii_min)
                        ii = ii.max(dim=2)[0]
                        mm = masks[idx] == 255
                        mm = masks[idx] == 255
                        # print(ii.dtype)
                        diff = (ii - ii_min) * 255
                        ii = ii * 255
                        diff[mm] = 0
                        #print(diff.max(), diff.min())
                        #print(ii)
                        ii[mm] = 0
                        ii = cv2.imwrite('tmp/uncertain_{}_{}.png'.format(iter_idx, idx), ii.detach().cpu().numpy())
                        ii = cv2.imwrite('tmp/diff_{}_{}.png'.format(iter_idx, idx), diff.detach().cpu().numpy())

                    for idx, ii in enumerate(xor_mask.clone()):
                        #print(ii.shape)
                        #ii = ii.shape
                        #print(torch.unique(ii))
                        #print(ii.shape)
                        ii = ii / (ii.max() + 1e-7) * 255

                        cv2.imwrite('tmp/xor_{}_{}.png'.format(iter_idx, idx), ii.cpu().numpy().astype('uint8'))

                    #for idx, ii in enumerate(contrasive_loss_fn.store_values['gt'][0]):
                    for hook_name in contrasive_loss_fn.store_values.keys():
                    #for idx, ii in enumerate(contrasive_loss_fn.store_values['pred']):
                        #print(ii.shape)
                        for idx, ii in enumerate(contrasive_loss_fn.store_values[hook_name]):
                            ii = ii.clone()

                            if torch.is_tensor(ii):
                                ii = ii.cpu().numpy()

                            if ii.ndim == 3:
                                ii = ii[0]


                            #print('zuida', ii.max())
                            if ii.max() != 0:
                                #if idx == 8:
                                    #print(hook_name, ii.max())
                                ii = ii  / ii.max() * 255
                            #print(ii.shape)
                            #print(ii.shape)
                            cv2.imwrite('tmp/{}_{}_{}.png'.format(hook_name, iter_idx, idx), ii.astype('uint8'))

                    #for idx, ii in enumerate(contrasive_loss_fn.store_values['pred'][0]):
                    ###for idx, ii in enumerate(contrasive_loss_fn.store_values['gt']):
                    #    #print(ii.shape)
                    #    if torch.is_tensor(ii):
                    #        ii = ii.cpu().numpy()

                    #    #ii = ii[0]
                    #    if ii.ndim == 3:
                    #        ii = ii[0]

                    #    ii = ii.copy()
                    #    #print('zuida', ii.max())
                    #    ii = ii / (ii.max() + 1e-7) * 255
                    #    #print(ii.shape)
                    #    cv2.imwrite('tmp/hook_xor_{}.png'.format(idx), ii.astype('uint8'))

                    print('stopping...........')
                    # import sys; sys.exit()




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

        #if iter_idx == 2000:
        #    torch.save(net.state_dict(), '2000.pth')
        #    torch.save(images, 'images.pt')
        #    torch.save(masks, 'masks.pt')
        #    torch.save(cnt_masks, 'cnt_masks.pt')
        #    torch.save(gland_masks, 'gland_masks.pt')
        #if args.poly:
            #train_scheduler.step()
        #p.step()
        lr_schduler.step(iter_idx)

        #if iter_idx > 1000:
            #break
        # bs = len(images)
        del images
        del masks
        #loss =  loss_l2(preds, masks)
        # print(torch.cuda.utilization())

        #if not args.baseline:
        #    seg_loss = loss_seg(preds, masks)
        #    loss[seg_loss == 1] *= args.alpha

        iter_time = time.time() - train_t

        if (iter_idx + 1) % 50 == 0:
            print(('Training Iter: [{iter}/{total_iter}] '
                    # 'Lr:{lr:0.6f} Total Loss:{loss:0.4f}, Sup Loss:{sup_loss:0.4f}, Contrasive Loss: {con_loss:0.4f} Iter time:{iter_time:0.4f}s Data loading time:{data_time:0.4f}s').format(
                    'Lr:{lr:0.6f} Total Loss:{loss:0.4f}, Iter time:{iter_time:0.4f}s Data loading time:{data_time:0.4f}s').format(
                loss=loss.item(),
                #sup_loss=sup_loss.item(),
                #con_loss=contrasive_loss.item(),
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
            #total, testA, testB = evaluate(net, val_loader, args)
            results = evaluate(net, val_loader, args, val_set)

            #print('total: F1 {}, Dice:{}, Haus:{}'.format(*total))

            for key, values in results.items():
                print('{}: F1 {}, Dice:{}, Haus:{}'.format(key, *values))


            #if args.dataset == 'Glas':

            #    print('testA: F1 {}, Dice:{}, Haus:{}'.format(*testA))
            #    print('testB: F1 {}, Dice:{}, Haus:{}'.format(*testB))

            #if best[0] <
            # if best['testA_F1'] < testA[0]:

                # best['testA_F1'] = test

            #best['testA_F1'] = max(best['testA_F1'], testA[0])
            #best['testA_Dice'] = max(best['testA_Dice'], testA[1])
            #best['testA_Haus'] = min(best['testA_Haus'], testA[2])


            #best['testB_F1'] = max(best['testB_F1'], testB[0])
            #best['testB_Dice'] = max(best['testB_Dice'], testB[1])
            #best['testB_Haus'] = min(best['testB_Haus'], testB[2])

            #best['total_F1'] = max(best['total_F1'], total[0])
            #best['total_Dice'] = max(best['total_Dice'], total[1])
            #best['total_Haus'] = min(best['total_Haus'], total[2])

            #print(best)

            utils.visualize_metric(writer,
                #['total_F1', 'total_Dice', 'total_Haus'], total, iter_idx)
                #total_metrics, total, iter_idx)
                total_metrics, results['total'], iter_idx)

            if args.dataset == 'Glas':
                if val_set == 'val':
                    utils.visualize_metric(writer,
                        #['testA_F1', 'testA_Dice', 'testA_Haus'], testA, iter_idx)
                        testA_metrics, results['testA'], iter_idx)

                    utils.visualize_metric(writer,
                        #['testB_F1', 'testB_Dice', 'testB_Haus'], testB, iter_idx)
                        testB_metrics, results['testB'], iter_idx)

                if val_set == 'testA':
                    utils.visualize_metric(writer,
                        #['testA_F1', 'testA_Dice', 'testA_Haus'], testA, iter_idx)
                        testA_metrics, results['testA'], iter_idx)

                if val_set == 'testB':
                    utils.visualize_metric(writer,
                        #['testB_F1', 'testB_Dice', 'testB_Haus'], testB, iter_idx)
                        testB_metrics, results['testB'], iter_idx)

            #print(('Training Epoch:{epoch} [{trained_samples}/{total_samples}] '
            #print(('Training Iter:{iter} [{trained_samples}/{total_samples}] '
            #        'Lr:{lr:0.8f} Loss:{loss:0.4f} Data loading time:{time:0.4f}s').format(
            #    loss=loss.item(),
            #    # epoch=epoch,
            #    # iter=batch_idx,
            #    iter=iter_idx,
            #    # trained_samples=iter_idx * args.b + len(images),
            #    trained_samples=iter_idx * args.b + len(images),
            #    # total_samples=len(train_dataset),
            #    total_samples=total_iter,
            #    lr=optimizer.param_groups[0]['lr'],
            #    #beta=optimizer.param_groups[0]['betas'][0],
            #    #time=batch_finish - train_start
            #    time=eval_finish - eval_start
            #))


            #total_load_time += batch_finish - batch_start
            # total_load_time += eval_finish - train_start

            # utils.visulaize_lastlayer(
            #     writer,
            #     net,
            #     iter_idx,
            # )

            # batch_start = time.time()

            # continue
            # utils.visualize_scalar(
            #     writer,
            #     'Train/LearningRate',
            #     optimizer.param_groups[0]['lr'],
            #     #optimizer.param_groups[0]['betas'][0],
            #     # epoch,
            #     iter_idx,
            # )

            #utils.visualize_scalar(
            #    writer,
            #    'Train/Beta1',
            #    optimizer.param_groups[0]['betas'][0],
            #    epoch,
            #)
            #print(total_load_time, total_training)

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
            ckpt_manager.save_best(net, total_metrics,
                results['total'], iter_idx)

            if args.dataset == 'Glas':
                if val_set == 'val':
                    ckpt_manager.save_best(net, testA_metrics,
                        results['testA'], iter_idx)
                    ckpt_manager.save_best(net, testB_metrics,
                        results['testB'], iter_idx)

                if val_set == 'testA':
                    ckpt_manager.save_best(net, testA_metrics,
                        results['testA'], iter_idx)

                if val_set == 'testB':
                    ckpt_manager.save_best(net, testB_metrics,
                        results['testB'], iter_idx)

            print('best value:', ckpt_manager.best_value)
            net.train()


        train_t = time.time()

        if total_iter <= iter_idx:
            break

def evaluate(net, val_dataloader, args, val_set):
    net.eval()
        # test_loss = 0.0

    # F1, Dice, Haus
    #testA = np.array([0, 0, 0])
    #testB = np.array([0, 0, 0])
    #total = np.array([0, 0, 0])
    testA = 0
    testB = 0
    total = 0
    # best =  np.array([0, 0, 9999])

    count_A = 0
    count_B = 0

    # crag_res = 0

    # testA = {
    #     "F1" : 0,
    #     'Dice' : 0,
    #     'Haus' : 9999,
    # }
    # best = {
    #     "F1" : 0,
    #     'Dice' : 0,
    #     'Haus' : 9999,
    # }

    # testB = {
    #     "F1" : 0,
    #     'Dice' : 0,
    #     'Haus' : 9999,
    # }

    # total = {
    #     'F1'
    # }

    # test_start = time.time()
    # iou = 0
    # all_acc = 0
    # acc = 0

    valid_dataset = val_dataloader.dataset
    #cls_names = valid_dataset.class_names
    #ig_idx = valid_dataset.ignore_index
    count = 0
    #sampler = _sampler.OHEMPixelSampler(ignore_index=val_dataloader.dataset.ignore_index, min_kept=10000)
    out ={}

    # print('dataset size:', len())
    with torch.no_grad():
        for img_metas in tqdm(val_dataloader):
            for img_meta in img_metas:



                imgs = img_meta['imgs']
                gt_seg_map = img_meta['seg_map']
                ori_shape = gt_seg_map.shape[:2]


                pred, seg_logit = test_aug.aug_test(
                    imgs=imgs,
                    flip_direction=img_meta['flip'],
                    ori_shape=ori_shape,
                    model=net,

                    mode='whole',
                    crop_size=None,
                    stride=None,

                    # rescale=True,
                    rescale=False,

                    #mode='slide',
                    #stride=(256, 256),
                    ## crop_size=(480, 480),
                    ## crop_size=(416, 416),
                    #crop_size=settings.CROP_SIZE_GLAS if args.dataset=='Glas' else settings.CROP_SIZE_CRAG,
                    num_classes=valid_dataset.class_num
                )
                # print(img_meta['img_name'])

                # print(seg_logit.max(), seg_logit.min())

                #pred[pred > 1] = 0
                pred = pred == 1
                #pred = morph.remove_small_objects(pred, 100) # 0.84097868
                #pred = morph.remove_small_objects(pred, 100 * 2) # 0.85248084
                #pred = morph.remove_small_objects(pred, 100 * 2 * 2) # 0.8694393423149404
                #pred = morph.remove_small_objects(pred, 100 * 5) # 0.87442937
                #pred = morph.remove_small_objects(pred, 100 * 6) # 0.8763693
                #pred = morph.remove_small_objects(pred, 100 * 7) # 0.87812435
                #pred = morph.remove_small_objects(pred, 100 * 8) # 0.88134712
                #pred = morph.remove_small_objects(pred, 100 * 9) # 0.88131605
                #pred = morph.remove_small_objects(pred, 100 * 8 + 50) # 0.88219517
                # pred = morph.remove_small_objects(pred, 100 * 8 + 50) # 0.88219517
                #                                                         0.88228165

                #img_name = img_meta['img_name']
                #if 'testA' in img_name:
                    #pred = morph.remove_small_objects(pred, 100 * 8 + 50) # 0.88219517
                    #pred = morph.remove_small_objects(pred, 100 * 12 + 50)
                pred = morph.remove_small_objects(pred, 100 * 8 + 50)

                #if 'testB' in img_name:
                #    #pred = morph.remove_small_objects(pred, 100 * 14 + 50) # 0.88219517
                #    pred = morph.remove_small_objects(pred, 100 * 8 + 50) # 0.88219517
                # multi scale
                #pred = morph.remove_small_objects(pred, 100 * 8 + 50) #  0.87709376
                #pred = morph.remove_small_objects(pred, 100 * 7) # 0.87236826
                #pred = morph.remove_small_objects(pred, 100 * 9) # 0.87694807

                pred[pred > 1] = 0

                h, w = gt_seg_map.shape
                pred = cv2.resize(pred.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)

                img_name = img_meta['img_name']

                #if 'testA_39' in img_name:
                #    #print(img_name)
                #    assert len(imgs) == 1
                #    #torch.save(imgs[0], 'tmp/testA_39_input.pt')
                #    #torch.save(seg_logit, 'tmp/testA_39_output.pt')
                #    cv2.imwrite('{}_fff.png'.format('testA_39'), pred)
                #print(torch.unique(gt_seg_map))

                #gland = seg_logit[:, :2, :, :]
                #cnt = seg_logit[:, 2:, :, :]
                #gt_seg_map[gt_seg_map == 2] = 1
                #cnt_map = gt_seg_map == 2
                #cnt_map = cnt_map.unsqueeze(0)
                #cnt_map = cnt_map.unsqueeze(0)
                #seg_weight = sampler.sample(cnt, cnt_map.long())
                #print(seg_weight.shape)

                #print(pred.shape, gt_seg_map.shape)
                #count += 1
                #print(np.unique(pred))
                #print(torch.unique(gt_seg_map))
                #gt_seg_map[gt_seg_map == 2] = 1
                #import skimage.morphology as morph
                #pred = morph.label(pred, connectivity=2)
                #pred = morph.remove_small_objects(pred, 100)
                #pred[pred!=0] = 1
                #cv2.imwrite('tmp/test{}.jpg'.format(count), pred * 255)
                #cv2.imwrite('tmp/test{}_mask.jpg'.format(count), gt_seg_map.numpy() * 255)
                #gland = seg_logit[:, :2, :, :]
                #cnt = seg_logit[:, 2:, :, :]
                #r = cv2.imwrite('tmp/test{}_gland.jpg'.format(count), gland[0, 1, :, :].numpy() * 255)
                #j = cv2.imwrite('tmp/test{}_cnt.jpg'.format(count), cnt[0, 1, :, :].numpy() * 255)
                #count += 1
                #cv2.imwrite('tmp/test{}_sampler.jpg'.format(count), seg_weight[0].numpy() * 255)

                #pred =
                #print(pred.shape,  gt_seg_map.shape)
                #print(np.unique(pred))



                #t3 = time.time()
                #print(t3 - t2)
                #pred = cv2.imread('/data/hdd1/by/FullNet-varCE/tmp/testA_39_pred.png', -1)
                #print(pred.shape)
                #cv2.imwrite('test11.png', pred * 255)
                #cv2.imwrite('test22.png', gt_seg_map * 255)
                _, _, F1, dice, _, haus = gland_accuracy_object_level(pred, gt_seg_map)
                #print(F1, dice, haus)
                #t4 = time.time()
                #print(count, F1, dice, haus)
                #if 'testA_39' in img_name:
                    #print(F1, dice, haus)
                #print(img_name, F1, dice, haus)
                #print(F1, dice, haus)
                #print(t4 - t3, 'gland_acc time')
                #print()
                #print(count, F1, dice, haus)

                #if count > 10:

                res = np.array([F1, dice, haus])


                count += 1
                total += res


                if args.dataset == 'Glas':
                    if 'testA' in img_name:
                        count_A += 1
                        testA += res

                    if 'testB' in img_name:
                        count_B += 1
                        testB += res


    total = total / count
    out['total'] = total


    if args.dataset == 'Glas':
        #total = (testA + testB) / (count_A + count_B)

        if val_set == 'testA':
            testA = testA / count_A
            out['testA'] = testA
            assert count == count_A

        if val_set == 'testB':
            testB = testB / count_B
            out['testB'] = testB
            assert count == count_B

        if val_set == 'val':
            testA = testA / count_A
            testB = testB / count_B
            out['testA'] = testA
            out['testB'] = testB


    #if args.dataset == 'carg':
    #    total = crag_res / count
    #    testA = 0.1
    #    testB = 0.1

    #return total, testA, testB
    return out



        #    if args.gpu:
        #        images = images.cuda()
        #        masks = masks.cuda()
        #        #masks = images

        #    with torch.no_grad():
        #        preds = net(images)
        #        # loss = loss_fn(preds, masks)
        #        # loss = loss.mean()
        #    # continue


        #    #if not args.baseline:
        #    #    seg_loss = loss_seg(preds, masks)
        #    #    loss[seg_loss == 1] *= args.alpha

        #    # test_loss += loss.item()

        #    preds = preds.argmax(dim=1)
        #    #tmp_all_acc, tmp_acc, tmp_mean_iou = eval_metrics(
        #    #    , masks.detach().cpu().numpy(), len(cls_names), ig_idx
        #    #)
        #    tmp_all_acc, tmp_acc, tmp_iou = eval_metrics(
        #        preds.detach().cpu().numpy(),
        #        masks.detach().cpu().numpy(),
        #        len(cls_names),
        #        ignore_index=ig_idx,
        #        metrics='mIoU',
        #        nan_to_num=-1
        #    )

        #    all_acc += tmp_all_acc * len(images)
        #    acc += tmp_acc * len(images)
        #    iou += tmp_iou * len(images)

    # continue
    #all_acc /= len(val_dataloader.dataset)
    #acc /= len(val_dataloader.dataset)
    #iou /= len(val_dataloader.dataset)
    #test_finish = time.time()
    #print('Evaluation time comsumed:{:.2f}s'.format(test_finish - test_start))
    #print('Iou for each class:')
    #utils.print_eval(cls_names, iou)
    #print('Acc for each class:')
    #utils.print_eval(cls_names, acc)
    ##print('%, '.join([':'.join([str(n), str(round(i, 2))]) for n, i in zip(cls_names, iou)]))
    ##iou = iou.tolist()
    ##iou = [i for i in iou if iou.index(i) != ig_idx]
    #miou = sum(iou) / len(iou)
    #print('Epoch {}  Mean iou {:.4f}  All Pixel Acc {:.4f}'.format(epoch, miou, all_acc))
    ##print('%, '.join([':'.join([str(n), str(round(a, 2))]) for n, a in zip(cls_names, acc)]))
    ##print('All acc {:.2f}%'.format(all_acc))

    #utils.visualize_scalar(
    #    writer,
    #    'Test/mIOU',
    #    miou,
    #    epoch,
    #)

    #utils.visualize_scalar(
    #    writer,
    #    'Test/Acc',
    #    all_acc,
    #    epoch,
    #)

    #utils.visualize_scalar(
    #    writer,
    #    'Test/Loss',
    #    test_loss / len(valid_dataset),
    #    epoch,
    #)

    if best_iou < miou:
        best_iou = miou
        if prev_best:
            os.remove(prev_best)

        torch.save(net.state_dict(),
                        checkpoint_path.format(epoch=epoch, type='best'))
        prev_best = checkpoint_path.format(epoch=epoch, type='best')
        # continue

    if not epoch % settings.SAVE_EPOCH:
        torch.save(net.state_dict(),
                        checkpoint_path.format(epoch=epoch, type='regular'))




if __name__ == '__main__':




    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.007,
                        help='initial learning rate')
    parser.add_argument('-e', type=int, default=120, help='training epoches')
    parser.add_argument('-iter', type=int, default=40000, help='training epoches')
    parser.add_argument('-eval_iter', type=int, default=2000, help='training epoches')
    parser.add_argument('-wd', type=float, default=1e-4, help='training epoches')
    parser.add_argument('-resume', type=bool, default=False, help='if resume training')
    parser.add_argument('-net', type=str, required=True, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Glas', help='dataset name')
    parser.add_argument('-prefix', type=str, default='', help='checkpoint and runs folder prefix')
    parser.add_argument('-alpha', type=float, default=0.1, help='panalize parameter')
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
    parser.add_argument('-vis', action='store_true', default=False, help='vis result of mid layer')
    parser.add_argument('-imgset', type=str, default='train', help='default training set')
    args = parser.parse_args()
    print(args)

    if args.wait:
        from gpustats import GPUStats
        gpu_stats = GPUStats(gpus_needed=1, sleep_time=60 * 5, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
        gpu_stats.run()

    root_path = os.path.dirname(os.path.abspath(__file__))

    # checkpoint_path = os.path.join(
    #     root_path, settings.CHECKPOINT_FOLDER, args.prefix + '_' + settings.TIME_NOW)
    log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    print('saving tensorboard log into {}'.format(log_dir))

    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # print()
    # train_img_set = 'trainB'
    # val_img_set = 'val'
    # img_set = 'trainA'
    assert args.imgset in ['train', 'trainA', 'trainB']
    img_set = args.imgset
    print(img_set)

    if img_set == 'trainA':
        val_set = 'testB'

    if img_set == 'trainB':
        val_set = 'testA'

    if img_set == 'train':
        val_set = 'val'


    train_loader = utils.data_loader(args, img_set)
    train_dataset = train_loader.dataset

    val_loader = utils.data_loader(args, val_set)

    net = utils.get_model(args.net, 3, train_dataset.class_num, args=args)
    #net.load_state_dict(torch.load('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_16_January_2023_05h_11m_49s/iter_39999.pt'))


    if args.resume:
        weight_path = utils.get_weight_path(
            os.path.join(root_path, settings.CHECKPOINT_FOLDER))
        print('Loading weight file: {}...'.format(weight_path))
        net.load_state_dict(torch.load(weight_path))
        print('Done loading!')

    #new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_16_January_2023_05h_11m_49s/iter_39999.pt'))
    # test_pretrain_crag_glas_rings_prostate
    # best pretrain
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Tuesday_24_January_2023_01h_28m_51s/iter_39999.pt'

    # test_pretrain_crag_glas_rings_prostate_with_upsampling
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Tuesday_24_January_2023_01h_24m_58s/iter_39999.pt'

    # test_pretrain_crag_glas_rings_with_upsampling
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Sunday_22_January_2023_18h_25m_35s/iter_39999.pt'

    # test_pretrain_crag_glas_rings
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Sunday_22_January_2023_11h_37m_21s/iter_39999.pt'

    # test_pretrain_glas_crag
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_21_January_2023_06h_37m_14s/iter_39999.pt'

    # test_pretrain_glas
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Sunday_22_January_2023_03h_06m_13s/iter_39999.pt'

    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/epoch_200.pth'

    # glas+crag+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_densecl/latest.pth'

    # glas + crag + lizard + sin + densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_lizard_sin/latest.pth'

    # glas+crag+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup//latest.pth'

    # glas+densecl
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/work_dir_glas/latest.pth'

    # glas+crag+rings+densecl
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/work_dir_glas_crag_sings/latest.pth'

    # glas+crag+rings+sins_densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin_rings_densecl/latest.pth'

    # glas+crag+rings+sins+crc+lizard_densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin_lizard_crc/latest.pth'

    # glas+crag+_mocov2
    # ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_mocov2_bs64/latest.pth'
    # glas + crag + rings + sin + lizard + crc
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_rings_lizard_sin_crc_mocov2/latest.pth'
    #ckpt_path = 'best_pretrain/iter_39999.pt'

    # for debug
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/unet_branch_SGD_473_Wednesday_08_March_2023_23h_28m_07s/iter_39999.pt'
    #ckpt_path = 'best_total_Dice_0.8844_iter_5999.pt'
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/unet_branch_SGD_473_Sunday_12_March_2023_18h_52m_04s/iter_39999.pt'

    # pretrained
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Wednesday_15_March_2023_23h_47m_53s/iter_39999.pt'

    # test_pretraining_eihs
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_18_March_2023_22h_57m_50s/iter_39999.pt'

    # test_CRCTP
    #ckpt_path = '/data/hdd1/by/mmclassification/work_dirs/gland/latest.pth'

    # eish + crag
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_25_March_2023_23h_38m_31s/iter_39999.pt'

    # eish + crag + glas
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Sunday_26_March_2023_17h_48m_38s/iter_39999.pt'


    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_25_March_2023_23h_51m_46s/iter_39999.pt'
    #ckpt_path = '/data/smb/syh/checkpoints/only_pretrained_on_ehis/iter_39999.pt'

    # tgt pretrained on crops
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Wednesday_05_April_2023_23h_43m_23s/iter_79999.pt'

    #tg pretrained on crops
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Wednesday_05_April_2023_23h_43m_23s/iter_79999.pt'

    # tgt percent 30 pretrain
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_08_April_2023_23h_39m_46s/iter_39999.pt'

    # tgt percent 60 pretrain
    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Saturday_08_April_2023_23h_46m_27s/iter_39999.pt'

    # tgt percent 100 pretrain
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Sunday_09_April_2023_23h_37m_23s/iter_39999.pt'

    # zuihao pretrain
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_27_March_2023_21h_23m_46s/iter_39999.pt'

    # swav_crag_train_crag_test_glas_train_ehis
    # ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_swav_ehis_glas_crag/latest.pth'

    # trained on classification glas val
    #ckpt_path = '/data/hdd1/by/mmclassification/test_glas_val/latest.pth'

    # crag with graph attention
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_27_March_2023_21h_23m_46s/iter_39999.pt'
    # glas+crag+rings+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin_rings_densecl/latest.pth'
    # glas+crag+sin+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin/latest.pth'


    # milestone no crag
    ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Tuesday_18_April_2023_22h_09m_55s/iter_45999.pt'


    # draw uncertain map
    print('Loading pretrained checkpoint from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    new_state_dict = utils.on_load_checkpoint(net.state_dict(), ckpt)
    #new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path))
    net.load_state_dict(new_state_dict)
    print('Done!')

    if args.gpu:
        net = net.cuda()

    tensor = torch.Tensor(1, 3, 480, 480)
    # net.eval()
    #utils.visualize_network(writer, net, tensor)
    net.train()

    train(net, train_loader, val_loader, writer, args, val_set)
