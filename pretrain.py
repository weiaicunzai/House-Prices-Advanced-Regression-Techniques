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
from dataset.voc2012 import VOC2012Aug
#from dataset.camvid_lmdb import CamVid
from lr_scheduler import PolynomialLR, WarmUpLR, WarmUpWrapper
from metric import eval_metrics, gland_accuracy_object_level, accuracy_pixel_level
from loss import SegmentLevelLoss
from dataloader import IterLoader
import test_aug
from losses import DiceLoss, WeightedLossWarpper
import sampler as _sampler



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

    optimizer = optim.SGD(net.parameters(), lr=args.lr * args.scale, momentum=0.9, weight_decay=5e-4)
    # iter_per_epoch = len(train_dataset) / args.b

    # max_iter = args.e * len(train_loader)
    total_iter = args.iter
    warmup_iter = int(args.iter * 0.1)

    train_scheduler = PolynomialLR(optimizer, total_iters=total_iter - warmup_iter, power=0.9, min_lr=args.min_lr * args.scale)
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

                gland_preds, aux_preds, _ = net(images)
                loss = gland_loss_fn_ce(gland_preds, masks) + \
                                3 * gland_loss_fn_dice(gland_preds, masks)

                aux_loss = gland_loss_fn_ce(aux_preds, masks) + \
                                3 * gland_loss_fn_dice(aux_preds, masks)
                #weight_maps = weight_maps.float().div(20)
                #loss = loss * weight_maps
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

        #if not args.baseline:
        #    seg_loss = loss_seg(preds, masks)
        #    loss[seg_loss == 1] *= args.alpha

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
            acc, iou, recall, precision, F1, performance, total, testA, testB = evaluate(net, val_loader, args)
            print(
                'acc: {:.04f}'.format(acc),
                'iou: {:.04f}'.format(iou),
                'recall: {:.04f}'.format(recall),
                'precision: {:.04f}'.format(precision),
                'F1: {:.04f}'.format(F1),
                'performance: {:.04f}'.format(performance),
                #'total:', total,
                'total_F1:{:.4f}, total_dice:{:.4f}, total_haus:{:.4f}'.format(*total),
                'testA_F1:{:.4f}, testA_dice:{:.4f}, testA_haus:{:.4f}'.format(*testA),
                'testB_F1:{:.4f}, testB_dice:{:.4f}, testB_haus:{:.4f}'.format(*testB),
                #'testB', testB
            )

            #print('total: F1 {}, Dice:{}, Haus:{}'.format(*total))
            #print('testA: F1 {}, Dice:{}, Haus:{}'.format(*testA))
            #print('testB: F1 {}, Dice:{}, Haus:{}'.format(*testB))

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

            #utils.visualize_metric(writer,
            #    #['total_F1', 'total_Dice', 'total_Haus'], total, iter_idx)
            #    total_metrics, total, iter_idx)

            #utils.visualize_metric(writer,
            #    #['testA_F1', 'testA_Dice', 'testA_Haus'], testA, iter_idx)
            #    testA_metrics, testA, iter_idx)

            #utils.visualize_metric(writer,
            #    #['testB_F1', 'testB_Dice', 'testB_Haus'], testB, iter_idx)
            #    testB_metrics, testB, iter_idx)


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
            #ckpt_manager.save_best(net, total_metrics,
            #    total, iter_idx)
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
    #count = 0
    #sampler = _sampler.OHEMPixelSampler(ignore_index=val_dataloader.dataset.ignore_index, min_kept=10000)
    metrics_inside = 0
    count = 0
    with torch.no_grad():
        #print(val_dataloader.batch_size)
        for img_metas in tqdm(val_dataloader):
            for img_meta in img_metas:
                count += 1

                imgs = img_meta['imgs']
                #if 'testA_39' not in img_meta['img_name']:
                #    continue
                #print(len(imgs))
                #import sys; sys.exit()
                gt_seg_map = img_meta['seg_map']
                ori_shape = gt_seg_map.shape[:2]


                #t1 = time.time()
                pred, seg_logit = test_aug.aug_test(
                    imgs=imgs,
                    flip_direction=img_meta['flip'],
                    ori_shape=ori_shape,
                    model=net,
                    #crop_size=(480, 480),
                    #stride=(256, 256),
                    crop_size=None,
                    stride=None,
                    #mode='slide',
                    #rescale=True,
                    rescale=False,
                    mode='whole',
                    num_classes=valid_dataset.class_num
                )

                #t2 = time.time()
                #print(t2 - t1)
                #pred[pred > 1] = 0
                pred = pred == 1
                pred = morph.remove_small_objects(pred, 100)

                pred[pred > 1] = 0


                #print(pred.shape, np.unique(pred))
                h, w = gt_seg_map.shape
                pred = cv2.resize(pred.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)
                #print(pred.shape)

                img_name = img_meta['img_name']

                #if 'testA_39' in img_name:
                #    #print(img_name)
                #    assert len(imgs) == 1
                #    #torch.save(imgs[0], 'tmp/testA_39_input.pt')
                #    #torch.save(seg_logit, 'tmp/testA_39_output.pt')
                #    cv2.imwrite('{}_fff.png'.format('testA_39'), pred)
                #    import sys; sys.exit()
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
                #import sys; sys.exit()



                t3 = time.time()
                #print(t3 - t2)
                #pred = cv2.imread('/data/hdd1/by/FullNet-varCE/tmp/testA_39_pred.png', -1)
                #print(pred.shape)
                #metrics_inside += compute_pixel_level_metrics(pred, gt_seg_map)
                metrics_inside += accuracy_pixel_level(pred, gt_seg_map)
                _, _, F1, dice, _, haus = gland_accuracy_object_level(pred, gt_seg_map)
                #print(img_name, F1, dice, haus)
                t4 = time.time()
                #if 'testA_39' in img_name:
                    #print(F1, dice, haus)
                    #import sys; sys.exit()
                #print(img_name, F1, dice, haus)
                #print(t4 - t3)
                #print()
                #print(count, F1, dice, haus)

                #if count > 10:
                    #import sys; sys.exit()

                res = np.array([F1, dice, haus])


                if 'testA' in img_name:
                    count_A += 1
                    testA += res

                if 'testB' in img_name:
                    count_B += 1
                    testB += res

    total = (testA + testB) / (count_A + count_B)

    testA = testA / count_A
    testB = testB / count_B
    metrics_inside = metrics_inside / count

    acc, iou, recall, precision, F1, performance = metrics_inside
    return acc, iou, recall, precision, F1, performance, total, testA, testB


    #return total, testA, testB



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

    if best_iou < miou and epoch > args.e // 4:
    #if best_iou < miou:
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
    parser.add_argument('-scale', type=float, default=1, help='min_lr for poly')
    args = parser.parse_args()

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

    net = utils.get_model(args.net, 3, train_dataset.class_num, args=args)

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
