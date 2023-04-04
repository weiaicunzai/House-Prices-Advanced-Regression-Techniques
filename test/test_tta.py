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
from losses import DiceLoss, WeightedLossWarpper, GlandContrastLoss
import sampler as _sampler



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

    with torch.no_grad():
        for img_metas in tqdm(val_dataloader):
            for img_meta in img_metas:



                imgs = img_meta['imgs']
                gt_seg_map = img_meta['seg_map']
                ori_shape = gt_seg_map.shape[:2]

                print(img_meta['img_name'])

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

                # print(pred.shape)
                # output = pred.argmax(dim=1).squeeze(0)
                cv2.imwrite('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/after_{}.jpg'.format('final'), pred * 255)



                import sys; sys.exit()
                continue
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


   # total = total / count
   # out['total'] = total


   # if args.dataset == 'Glas':
   #     #total = (testA + testB) / (count_A + count_B)

   #     testA = testA / count_A
   #     testB = testB / count_B
   #     out['testA'] = testA
   #     out['testB'] = testB


   # #if args.dataset == 'carg':
   # #    total = crag_res / count
   # #    testA = 0.1
   # #    testB = 0.1

   # #return total, testA, testB
   # return out



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

    # if best_iou < miou and epoch > args.e // 4:
    # #if best_iou < miou:
    #     best_iou = miou
    #     if prev_best:
    #         os.remove(prev_best)

    #     torch.save(net.state_dict(),
    #                     checkpoint_path.format(epoch=epoch, type='best'))
    #     prev_best = checkpoint_path.format(epoch=epoch, type='best')
    #     # continue

    # if not epoch % settings.SAVE_EPOCH:
    #     torch.save(net.state_dict(),
    #                     checkpoint_path.format(epoch=epoch, type='regular'))




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
    parser.add_argument('-net', type=str, help='if resume training')
    parser.add_argument('-dataset', type=str, default='Glas', help='dataset name')
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
    args = parser.parse_args()
    print(args)

    # if args.wait:
    #     from gpustats import GPUStats
    #     gpu_stats = GPUStats(gpus_needed=1, sleep_time=60 * 5, exec_thresh=3, max_gpu_mem_avail=0.01, max_gpu_util=0.01)
    #     gpu_stats.run()

    # root_path = os.path.dirname(os.path.abspath(__file__))

    # checkpoint_path = os.path.join(
    #     root_path, settings.CHECKPOINT_FOLDER, args.prefix + '_' + settings.TIME_NOW)
    # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    # print('saving tensorboard log into {}'.format(log_dir))

    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # writer = SummaryWriter(log_dir=log_dir)

    print()

    train_loader = utils.data_loader(args, 'train')
    train_dataset = train_loader.dataset

    val_loader = utils.data_loader(args, 'val')

    args.net = 'tgt'
    args.gpu = True

    net = utils.get_model(args.net, 3, train_dataset.class_num, args=args)
    #net.load_state_dict(torch.load('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_16_January_2023_05h_11m_49s/iter_39999.pt'))



    # trained on classification glas val
    #ckpt_path = '/data/hdd1/by/mmclassification/test_glas_val/latest.pth'

    #ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Wednesday_15_March_2023_23h_47m_53s/iter_39999.pt'

    ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/unet_branch_SGD_473_Monday_27_March_2023_22h_00m_31s/best_total_F1_0.9041_iter_35999.pt'


    # crag with graph attention
    # ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Monday_27_March_2023_21h_23m_46s/iter_39999.pt'
    # glas+crag+rings+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin_rings_densecl/latest.pth'
    # glas+crag+sin+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin/latest.pth'
    print('Loading pretrained checkpoint from {}'.format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    print(type(ckpt))
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    new_state_dict = utils.on_load_checkpoint(net.state_dict(), ckpt)
    #new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path))
    net.load_state_dict(new_state_dict)
    print('Done!')

    if args.gpu:
        net = net.cuda()

    tensor = torch.Tensor(1, 3, 480, 480)
    net.eval()
    #utils.visualize_network(writer, net, tensor)
    #net.train()

    #train(net, train_loader, val_loader, writer, args)
    evaluate(net, val_loader, args)
