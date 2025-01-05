import os
import argparse
import re

import sys
sys.path.insert(0, '/home/baiyu/miniconda3/envs/torch1.13/lib/python3.10/site-packages')

import cv2

import torch
import torch.nn as nn

import skimage.morphology as morph
import transforms
from conf import settings
import utils
from metric import eval_metrics
from train import evaluate
from conf import settings
from losses import GlandContrastLoss
from tqdm import tqdm
import test_aug


import argparse
import os
import time
import re
# import sys

import numpy
print(numpy.__file__)
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
from losses import DiceLoss, WeightedLossWarpper, GlandContrastLoss, TI_Loss, soft_dice_cldice
import sampler as _sampler

import numpy as np
import skimage.morphology as morph
from skimage import measure
from scipy.spatial.distance import directed_hausdorff as hausdorff
import cv2
import unet3


#from dataset.camvid import CamVid
#from metrics import Metrics
#from model import UNet
counter=0
rate=30

def segment_level_loss(gt, pred, op='xor', out_size=(160, 160)):
        ignore_idx = 255

        
        gt = cv2.resize(gt, out_size[::-1], interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, out_size[::-1], interpolation=cv2.INTER_NEAREST)

        count_over_conn = 0
        count_under_conn = 0

        count_over_conn_pred = 0
        count_under_conn_gt = 0

        count_over_conn_gt = 0

        count_A = 0
        count_B = 0


        if op == 'none':
            return np.zeros(gt.shape, dtype=np.uint8)


        # remove ignore_idx
        # pred idx only contains 0 or 1, so we need to remove the blank region acoording = gt
        pred[gt==ignore_idx] = 0



        # set connectivity to 1 and set min_size to 64 as default
        # because some of the groud truth gland will have sawtooth effect due to the data augmentation
        pred = morph.remove_small_objects(pred == 1, connectivity=1)
        gt = morph.remove_small_objects(gt == 1, connectivity=1)



        # set connectivity to 1 to avoid glands clustered together due to resize
        # only counts cross connectivity
        pred_labeled, pred_num = measure.label(pred, return_num=True, connectivity=1)
        gt_labeled, gt_num = measure.label(gt, return_num=True, connectivity=1)


        # iterate through prediction
        #print(np.unique(gt), np.unique(pred))
        results = []
        #pred
        res = np.zeros(gt.shape, dtype=np.uint8)
        ans = np.zeros(gt.shape, dtype=np.uint8)
        process_list=[]
        # based on pred glands
        for i in range(0, pred_num):
            i += 1

            # gt != 0 is gt gland
            # pred_labeled == i is the ith gland of pred
            # pred_labeled_i的样子
            # [[False False False False  True]
            # [False False False False  True]
            # [False False False False  True]
            # [False False False  True  True]
            # [False False False False  True]]
            pred_labeled_i = pred_labeled == i
            # mask和pred_labeled_i一样
            mask = (pred_labeled_i) & (gt != 0)

            # for each pixel of mask in corresponding gt img
            #print(gt_labeled[mask].shape[0], len(gt_labeled[mask]))
            # gt_labeled[mask]是一维的[2 2 2 2 2]
            # 这种情况是pred有，gt没有，也
            if len(gt_labeled[mask]) == 0:
                # no gt gland instance in corresponding
                # location of gt image

                #res[pred_labeled == i] = 1
                res[pred_labeled_i] = 1
                ans[pred_labeled_i] = 1
                # count_under_conn_pred += 1

                continue

            # one pred gland contains more than one gt glands
            if gt_labeled[mask].min() != gt_labeled[mask].max():
                # more than 1 gt gland instances in corresponding
                # gt image
                #res[pred_labeled == i] = 1
                res[pred_labeled_i] = 1
                # 处理gt
                gt_unique_values=np.unique(gt_labeled[mask])
                new_pic = np.zeros(gt.shape, dtype=np.uint8)
                
                for value in gt_unique_values:
                    new_pic[gt_labeled == value] = 1
                    
                w = unet3.weight_add_np(new_pic, rate)
                # 是否正则？
                w = w / w.max() 
                process_list.append(w)
                

                # count_under_conn_pred += 1

                #print(len(np.unique(gt_labeled[mask])) - 1, np.unique(gt_labeled[mask]))

                count_under_conn_gt += len(np.unique(gt_labeled[mask])) - 1

            else:
                # corresponding gt gland area is less than 50%
                if mask.sum() / pred_labeled_i.sum() < 0.5:
                    #res[pred_labeled == i] = 1
                    res[pred_labeled_i] = 1
                    ans[pred_labeled_i] = 1
                    # count_under_conn_pred += 1
                    #pred_labeled_i_xor = np.logical_xor(pred_labeled_i, mask)
                    #res[pred_labeled_i_xor] = 1


        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/pred_region_wrong_number.png', res * 255)
        ###################################################

        #gt
        results.append(res)

        res = np.zeros(gt.shape, dtype=np.uint8)
        for i in range(0, gt_num):
            i += 1
            gt_labeled_i = gt_labeled == i
            #mask = (gt_labeled == i) & (pred != 0)
            mask = gt_labeled_i & (pred != 0)

            if len(pred_labeled[mask]) == 0:
                # no pred gland instance in corresponding
                # predicted image

                #res[gt_labeled == i] = 1
                res[gt_labeled_i] = 1
                ans[gt_labeled_i] = 1
                #cv2.imwrite('resG{}.png'.format(i), res)
                # count_over_conn_pred += 1

                count_over_conn_gt += 1
                continue

            if pred_labeled[mask].min() != pred_labeled[mask].max():
                #res[gt_labeled == i] = 1
                res[gt_labeled_i] = 1
                # 处理pred
                pred_unique_values=np.unique(pred_labeled[mask])
                new_pic = np.zeros(pred.shape, dtype=np.uint8)
                
                for value in pred_unique_values:
                    new_pic[pred_labeled == value] = 1
                    
                w = unet3.weight_add_np(new_pic, rate)
                w = w / w.max() 
                process_list.append(w)
                #cv2.imwrite('resG{}.png'.format(i), res)
                # count_over_conn += 1
                count_over_conn_gt += 1

                # count_under_conn_pred += len(np.unique(pred_labeled[mask])) - 1

            else:
                if mask.sum() / gt_labeled_i.sum() < 0.5:
                    #print(mask.sum() / gt_labeled_i.sum(), 'ccccccccccc')
                    #print(i, i, i, i)
                    #res[gt_labeled == i] = 1
                    res[gt_labeled_i] = 1
                    ans[gt_labeled_i] = 1
                    # count_over_conn += 1

                    count_over_conn_gt += 1
            #print(mask.sum() / (gt_labeled == i).sum(), 'cc111')
            #start = time.time()
            #for i in range(100):
            #    np.unique(test)
            #finish = time.time()
            #print('unique', finish - start)

            #start = time.time()
            #for i in range(100):
            #    if len(test) == 0:
            #        print('no')
            #        break
            #    test.max()

            #finish = time.time()
            #print('max min', finish - start)

        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/gt_region_wrong_number.png', res * 255)
        ###################################################
        results.append(res)


        res = cv2.bitwise_or(results[0], results[1])

        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/merge_all_wrong_number_region.png', res * 255)
        ###################################################
        # print('predict_num', pred_num, 'under', count_under_conn, 'over', count_over_conn)
        if op == 'or':
            return res

        elif op == 'xor':

            #cc = res.copy()
            gt_res = np.zeros(gt.shape, dtype=np.uint8)
            for i in range(0, gt_num):
                i += 1
                if res[gt_labeled == i].max() != 0:
                    gt_res[gt_labeled == i] = 1

            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/gt_1_final_candidate_region.png', gt_res * 255)
            ###################################################
            pred_res = np.zeros(gt.shape, dtype=np.uint8)
            for i in range(0, pred_num):
                i += 1
                if res[pred_labeled == i].max() != 0:
                    pred_res[pred_labeled == i] = 1

            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/pred_1_final_candidate_region.png', pred_res * 255)
            ###################################################

            #print(pred_res.shape, 'pred_res.shape')
            res = cv2.bitwise_xor(pred_res, gt_res)
            
            global counter
            counter +=1
            def save_pred_as_image(pred, output_filename):
                # 确保 pred 是 uint8 类型
                # pred_uint8 = (pred * 255).astype('uint8') if pred.dtype != 'uint8' else pred
                pred_uint8 = (pred * 255).astype('uint8') 
                # 保存为PNG文件
                cv2.imwrite(output_filename, pred_uint8)
                
                print(f"Prediction saved to {output_filename}")    
            # for pic in process_list:
            for index, pic in enumerate(process_list):
                save_pred_as_image(pic,f'./unetans2/{counter}xor_{index}.png')
                pic_xor = pic * res
                # pic_xor = pic & res
                ans[pic_xor!=0]=1
            save_pred_as_image(ans,f'./unetans2/{counter}xor.png')
            save_pred_as_image(gt_res, f'./unetans2/{counter}eval_gt.png')    
            save_pred_as_image(pred_res, f'./unetans2/{counter}eval_pred.png')    
            save_pred_as_image(res, f'./unetans2/{counter}eval_xor.png')    
            # exit(0)
            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/final_result.png', res * 255)
            ###################################################
            #print(pred_num)
            #return res

        #print('hello', count_under_conn_gt, count_over_conn_gt, gt_num)
        return {
            'pred': pred_num,
            'under': count_under_conn_gt,
            'over': count_over_conn_gt,
            'gt': gt_num,
        }

def connect(net, val_dataloader, args, val_set):
    count_A = 0
    count_B = 0
    total = 0
    # def evaluate(net, val_dataloader, args, val_set):

    pred_num = 0
    over_num = 0
    under_num = 0
    gt_num = 0

    with torch.no_grad():
        #for img_metas in tqdm(val_dataloader):
        print(val_dataloader)
        for img_metas in val_dataloader:
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
                    #num_classes=valid_dataset.class_num
                    num_classes=val_dataloader.dataset.class_num
                )

                pred = pred == 1

                pred = morph.remove_small_objects(pred, 100 * 8 + 50)



                pred[pred > 1] = 0

                h, w = gt_seg_map.shape
                pred = cv2.resize(pred.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)
                def save_pred_as_image(pred, output_filename):
                    # 确保 pred 是 uint8 类型
                    # pred_uint8 = (pred * 255).astype('uint8') if pred.dtype != 'uint8' else pred
                    pred_uint8 = (pred * 255).astype('uint8') 
                    # 保存为PNG文件
                    cv2.imwrite(output_filename, pred_uint8)
                    
                    print(f"Prediction saved to {output_filename}")    
                # save_pred_as_image(pred, 'eval_pred.png')    
                # exit(0)        
                # return pred

                # print(np.unique(pred))
                # cv2.imwrite('test.png', pred * 255)
                # import sys; sys.exit()

                img_name = img_meta['img_name']
                # print(img_name)
                # exit(0)
                # contrasive_loss_fn.segment_level_loss(gt, pred, op='xor', out_size=ori_shape):
                #segment_level_loss(gt, pred, op='xor', out_size=ori_shape)
                # print(gt_seg_map.dtype)
                out = segment_level_loss(gt_seg_map, pred, op='xor', out_size=ori_shape)
                # print(out)
                pred_num += out['pred']
                under_num += out['under']
                over_num += out['over']
                gt_num += out['gt']







                # res = np.array([F1, dice, haus])


                # count += 1
                # total += res


                if args.dataset == 'Glas':
                    if 'testA' in img_name:
                        count_A += 1
                        # testA += res

                    if 'testB' in img_name:
                        count_B += 1
                        # testB += res

    out = {}

    # total = total / count
    # out['total'] = total



    if args.dataset == 'Glas':
        #total = (testA + testB) / (count_A + count_B)
        # print(pred_num, under_num, over_num)
        # print(type(pred))
        #print(under_num / pred_num, over_num / pred_num)
        print('under_num', under_num, 'over_num', over_num, 'gt_num', gt_num, 'under ratio', under_num / gt_num, 'over ratio', over_num / gt_num)

        # if val_set == 'testA':
            # testA = testA / count_A
            # testA = out['pred']
            # out['testA'] = testA
            # assert count == count_A

        # if val_set == 'testB':
            # testB = testB / count_B
            # out['testB'] = testB
            # testA = out['over']
            # assert count == count_B

        # if val_set == 'val':
            # testB = out['under']
            # testA = testA / count_A
            # testB = testB / count_B
            # out['testA'] = testA
            # out['testB'] = testB


    if args.dataset == 'crag':
    #    total = crag_res / count
    #    testA = 0.1
    #    testB = 0.1
        # print(under_num / pred_num, over_num / pred_num)
        print('under_num', under_num, 'over_num', over_num, 'gt_num', gt_num, 'under ratio', under_num / gt_num, 'over ratio', over_num / gt_num)

    #return total, testA, testB
    return out


def gen_trans(combo):
    trans = transforms.MultiScaleFlipAug(
            # img_ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0],
            # rescale=True,
            img_ratios=[1],
            # flip=True,
            #flip=False,
            #flip_direction=['horizontal', 'vertical', ],
            #flip_direction=['h', 'v'],
            #flip_direction=['h', 'v', 'hv', 'r90'],
            #flip_direction=['h', 'v', 'r90'], + 1
            #flip_direction=['none', 'h', 'v', 'hv'],
            flip_direction=combo,
            #flip_direction=['h', 'v', 'r90h'],
            #flip_direction=['h', 'v', 'r90v'],
            # flip_direction=['h', 'v', 'hv', 'r90', 'r90h', 'r90v', 'r90hv', 'none'],
            #flip_direction=['h', 'v'],
            #flip_direction=['horizontal'],
            # transforms=[
                # transforms.ToTensor(),
                # transforms.Normalize(settings.MEAN, settings.STD),
            # ]
            resize_to_multiple=False,
            #min_size=208,
            #min_size=None,
            #min_size=480,
            #min_size=1024,
            # min_size=crop_size[0],
            #min_size=None, # F1 0.8776095444114832, Dice:0.8882521941014574,
            #min_size=args.size,
            min_size=480,
            mean=settings.MEAN,
            std=settings.STD
        )


    return trans



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-net', type=str, required=True, help='network name')
    parser.add_argument('-download', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=1,
                        help='batch size for dataloader')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
    parser.add_argument('-branch', type=str, default='hybird', help='dataset name')
    parser.add_argument('-imgset', type=str, default='train', help='dataset name')

    args = parser.parse_args()
    print(args)

    # contrasive_loss_fn = GlandContrastLoss(4, ignore_idx=train_dataloader.dataset.ignore_index, temperature=0.07)
    #m = re.search(r'[a-zA-Z]+_[0-9]+_[a-zA-z]+_[0-9]+h_[0-9]+m_[0-9]s','Thursday_21_January_2021_09h_24m_07s')
    #m = re.search(r'[a-zA-Z]+_[0-9]+_[a-zA-Z]+_[0-9]+h_[0-9]+m','Thursday_21_January_2021_09h_24m_07s')
    #print(m.group())
    checkpoint = os.path.basename(os.path.dirname(args.weight))

    # for  i in range(100, 800, 10):
        # args.size = i
    # if args.dataset == 'Glas':
            #test_dataloader = utils.data_loader(args, 'testA')
    test_dataloader = utils.data_loader(args, 'val')
    test_dataset = test_dataloader.dataset
    # print(test_dataset)
    # print(test_dataset.transforms)
    #print(test_dataset.class_num)
    #import sys; sys.exit()
    net = utils.get_model(args.net, 3, test_dataset.class_num, args=args)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()
    print(args.weight)
    net.eval()
    #print('Glas testA')


    #augs = ['h', 'v', 'hv', 'r90', 'r90h', 'r90v', 'r90hv', 'none']
    augs = ['h', 'v', 'hv', 'r90', 'r90h', 'r90v', 'r90hv', 'none']
    from itertools import combinations
    count = 0
    # transforms = test_dataset.transforms

    img_set = args.imgset
    if img_set == 'trainA':
        val_set = 'testB'

    if img_set == 'trainB':
        val_set = 'testA'

    if img_set == 'train':
        val_set = 'val'

    test_dataloader = utils.data_loader(args, val_set)
    dataset = test_dataloader.dataset
    print('dataset:', len(dataset))
    multiscale_collate = test_dataloader.collate_fn



    for i in range(1, len(augs) + 1):
    #     # print(i)
        for combo in list(combinations(augs, i)):
            count += 1
            print(combo)

    #         # test_dataset.transforms.flip_direction = combo

    #         # transforms.flip_direction = combo
    #         #print(combo)
    #         trans = gen_trans(list(combo))
    #         test_dataset.transforms = gen_trans(list(combo))
    #         test_dataloader.dataset.transforms = trans
    #         print(id(test_dataset.transforms), id(test_dataloader.dataset.transforms), id(trans))


            #test_dataloader.dataset.transforms.flip_direction = ['none', 'hv']
            #test_dataloader.dataset.transforms.flip_direction = list(combo)
            #if count == 1:
            #    combo1 = ['none', 'hv']
            #else:
            #    combo1 = ['r90', 'r90hv']
            # test_dataloader = utils.data_loader(args, 'val')
            dataset.transforms.flip_direction = combo
            dataset.transforms.min_size = 768

            test_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True,
                collate_fn=multiscale_collate)

            # print(len(test_dataloader))

            # print(combo1,  test_dataloader.dataset.transforms.flip_direction, id(test_dataloader.dataset.transforms.flip_direction))
            with torch.no_grad():
                    ##################################3
                    #results = evaluate(net, test_dataloader, args, val_set)
                    ##print('results.tiems', results)
                    #print('test dataset transforms is: ', test_dataloader.dataset.transforms)
                    #for key, values in results.items():
                    #    print('{}: F1 {}, Dice:{}, Haus:{}'.format(key, *values))
                    ##################################3
                    # print(id(test_dataloader))
                    pred = connect(net, test_dataloader, args, val_set)

#
#total: F1 0.8328158261132039, Dice:0.8954908049063768, Haus:108.60207262579972
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b765b700>
#0.1519434628975265 0.0636042402826855
#
#total: F1 0.846402819873217, Dice:0.8983382251526733, Haus:106.11144328631421
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b7711810>
#0.14462081128747795 0.06701940035273368
#
#total: F1 0.8492221900870847, Dice:0.8993064472884781, Haus:100.35744541849944
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3bbabff10>
#0.14938488576449913 0.05975395430579965
#
#total: F1 0.8393460905241366, Dice:0.8908754570916887, Haus:140.4478331309946
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b7711810>
#0.15752212389380532 0.07079646017699115
#
#total: F1 0.8434747060625817, Dice:0.8894970964422366, Haus:136.7822589361515
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b5f4b310>
#0.15061295971978983 0.08056042031523643
#
#total: F1 0.8361946475798538, Dice:0.8918918202230157, Haus:105.67535797915896
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b7711810>
#0.15 0.06206896551724138
#
#total: F1 0.8436962950963485, Dice:0.895920530711632, Haus:131.54596573750746
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3bbb0c070>
#0.164021164021164 0.06701940035273368
#
#total: F1 0.8346056740882929, Dice:0.8922671210304383, Haus:104.04074575008444
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b7711810>
#0.15520282186948853 0.07583774250440917
#
#total: F1 0.8414675578201598, Dice:0.8967951704346063, Haus:104.01194782467448
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b5f4b310>
#0.14485165794066318 0.06806282722513089
#
#total: F1 0.8404228090917742, Dice:0.8979035937560809, Haus:96.51755084255852
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b5d7f970>
#0.1480836236933798 0.06794425087108014
#
#total: F1 0.841013931190032, Dice:0.8886137471990633, Haus:150.47503130505726
#<torch.utils.data.dataloader.DataLoader object at 0x7fc3b5f4b310>
#0.14964788732394366 0.0721830985915493