import glob
import os

import cv2
import torch
import torch.nn as nn
import numpy as np
import skimage.morphology as morph
from skimage import measure



#pred_path = '/data/by/FullNet-varCE/experiments/GlaS/1/best/testB_prob_maps/testB_6_prob_inside.png'
#gt_path = '/data/by/FullNet-varCE/experiments/GlaS/1/best/testB_segmentation/testB_6_seg.tiff'
##pred_path = '/data/by/FullNet-varCE/experiments/GlaS/1/best/testB_prob_maps/testB_2_prob_inside.png'
##gt_path = '/data/by/FullNet-varCE/experiments/GlaS/1/best/testB_segmentation/testB_2_seg.tiff'
#
#pred = cv2.imread(pred_path, -1)
#gt = cv2.imread(gt_path, -1)
#pred[pred < 127] = 0
#pred[pred >= 127] = 255
#print(np.unique(pred))
#print(np.unique(gt))
#
##gt[gt!=0] = 255
#gt = gt / gt.max() * 255
#cv2.imwrite('pred.png', pred)
#cv2.imwrite('gt.png', gt)
#
#print(gt.shape)
#pred = cv2.resize(pred, gt.shape[::-1])
#print(pred.shape)
#
#gt[gt != 0] = 1
#pred[pred != 0] = 1
#
#gt = gt.astype('uint8')

def assign_colors(img, num):
    colors = [
        [1, 122, 33],
  	    (255, 33,255),
 		(255,0,0),
 		(0,255,0),
 		(0,0,255),
 		(255,255,0),
 		(0,255,255),
        (255,0,255),
        (192,192,192),
        (128,128,128),
        (128,0,0),
        (128,128,0)
    ]
    gt_colors = cv2.cvtColor(np.zeros(img.shape).astype('uint8'), cv2.COLOR_GRAY2BGR)

    for i in range(num):
        i += 1
        gt_colors[img == i] = colors[i]

    return gt_colors

def segment_level_loss(gt, pred, op='or'):
    #gt_cpu = gt.cpu().numpy()
    #pred_cpu = pred.cpu().numpy()
    if op == 'none':
        return np.zeros(gt.shape, dtype=np.uint8)

    pred = morph.remove_small_objects(pred == 1, 100)
    gt = morph.remove_small_objects(gt == 1, 3)
    #diff = np.bitwise_xor(pred, pred2)
    #pred = pred2
    #cv2.imwrite('diff.png', diff / diff.max() * 255)

    #print(type(pred))
    #print(pred.dtype)
    #pred = measure.label(pred)
    pred_labeled, pred_num = measure.label(pred, return_num=True)
    #print(pred_num)
    gt_labeled, gt_num = measure.label(gt, return_num=True)

    #gt_colors = assign_colors(gt_labeled, gt_num)
    #pred_colors = assign_colors(pred_labeled, pred_num)
    #cv2.imwrite('gt_colors.jpg', gt_colors)
    #cv2.imwrite('pred_colors.jpg', pred_colors)
    #pred1 = morph.dilation(pred, selem=morph.selem.disk(4))

    #cv2.imwrite('pred_loss.png', pred_labeled / pred_labeled.max() * 255)

    #num_pred_objs = len(np.unique(pred))
    #print(num_pred_objs)
    #res = np.zeros(pred_labeled.shape)

    #colors = random.choices(colors, k=pred_num)
    #gt_colors = cv2.cvtColor(np.zeros(gt_labeled.shape).astype('uint8'), cv2.COLOR_GRAY2BGR)
    #pred_colors = gt_colors.copy()
    #g_num = np.unique(gt_labeled)
    #print('g_num', g_num)
    #p_num = np.unique(pred_labeled)
    #print('p_num', p_num)
    #for i in g_num:
    #    if i == 0:
    #        continue
    #    gt_colors[gt_labeled == i] = colors[i]
    #cv2.imwrite('gt_colors.png', gt_colors)

    #for i in p_num:
    #    if i == 0:
    #        continue
    #    pred_colors[pred_labeled == i] = colors[i]
    #cv2.imwrite('pred_colors.png', pred_colors)



    # iterate through prediction
    #print(np.unique(gt), np.unique(pred))
    results = []
    #pred
    res = np.zeros(gt.shape, dtype=np.uint8)
    for i in range(0, pred_num):
        i += 1
        mask = (pred_labeled == i) & (gt != 0)

        # for each pixel of mask in corresponding gt img
        if len(gt_labeled[mask]) == 0:
            # no gt gland instance in corresponding
            # location of gt image

            res[pred_labeled == i] = 1
            #cv2.imwrite('resP{}.png'.format(i), res)
            continue

        if gt_labeled[mask].min() != gt_labeled[mask].max():
            # more than 1 gt gland instances in corresponding
            # gt image
            res[pred_labeled == i] = 1
            #cv2.imwrite('resP{}.png'.format(i), res)

    #gt
    results.append(res)
    res = np.zeros(gt.shape, dtype=np.uint8)
    for i in range(0, gt_num):
        i += 1
        mask = (gt_labeled == i) & (pred != 0)

        if len(pred_labeled[mask]) == 0:
            # no pred gland instance in corresponding
            # predicted image

            res[gt_labeled == i] = 1
            #cv2.imwrite('resG{}.png'.format(i), res)
            continue

        if pred_labeled[mask].min() != pred_labeled[mask].max():
            res[gt_labeled == i] = 1
            #cv2.imwrite('resG{}.png'.format(i), res)
        #finish = time.time()
        #print('sum', finish - start)

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
    results.append(res)
    res = cv2.bitwise_or(results[0], results[1])
    if op == 'or':
        return res

    elif op == 'xor':

        #cc = res.copy()
        gt_res = np.zeros(gt.shape, dtype=np.uint8)
        #print(gt_num, pred_num, np.unique(gt_labeled), np.unique(pred_labeled))
        for i in range(0, gt_num):
            i += 1
            if res[gt_labeled == i].max() != 0:
                gt_res[gt_labeled == i] = 1

        pred_res = np.zeros(gt.shape, dtype=np.uint8)
        for i in range(0, pred_num):
            i += 1
            if res[pred_labeled == i].max() != 0:
                pred_res[pred_labeled == i] = 1

        res = cv2.bitwise_xor(pred_res, gt_res)
        return res
        #return pred_res
        #return gt_res, pred_res, res, cc
            #if len(pred_labeled[mask]) == 0:
            #if gt_labeled[]:
            #    # no pred gland instance in corresponding
            #    # predicted image

            #    res[gt_labeled == i] = 1
            #    #cv2.imwrite('resG{}.png'.format(i), res)
            #    continue
            #res = cv2.bitwise_or(pred_res, gt_res)
    else:
        raise ValueError('operation not suportted')


def segment_mask(gts, preds, op='or'):
    bs = gts.shape[0]
    preds = np.argmax(preds, axis=1)
    #for b_idx in range(batch_size):
    #    res.append(segment_level_loss(gt[b_idx], preds[b_idx]))
    res = [segment_level_loss(gts[b], preds[b], op=op) for b in range(bs)]
    res = np.stack(res, axis=0)
    return res

class SegmentLevelLoss(nn.Module):
    def __init__(self, op='or'):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.op = op

    def forward(self, preds, gts):
        res = segment_mask(gts.detach().cpu().numpy(), preds.detach().cpu().numpy(), op=self.op)
        #res = torch.from_numpy(res)
        res = torch.from_numpy(res).to(gts.device)
        return res
        #preds = self.softmax(preds)
        #loss = 1 + self.nll_loss(preds, res)
        #loss = loss.mean()
        #return loss


def check_mask(gt, pred):
    print(type(gt))
    #gt_gray = cv2.cvtColor(gt , cv2.COLOR_BGR2GRAY)
    #pred_gray = cv2.cvtColor(pred , cv2.COLOR_BGR2GRAY)
    print(np.unique(gt), np.unique(pred))
    return np.concatenate((gt , pred), axis=1)
    #gt_gray[gt_gray < 125] = 0
    #gt_gray[gt_gray >= 125] = 1
    #pred_gray[pred_gray < 125] = 0
    #pred_gray[pred_gray >= 125] = 1
#    print(gt_gray.shape, pred_gray.shape)
    #res = segment_level_loss(gt_gray, pred_gray)
#    #res = np.stack([res, res, res], axis=2)
#
    #im2, contours, hierarchy = cv2.findContours(gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(pred, contours, -1, (40,20,255), 3)
#
#
#    print(res.shape)
    #res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    #gt_gray = np.stack([gt_gray, gt_gray, gt_gray], axis=2)
    #return np.concatenate((gt, pred, res * 255), axis=1)
    #return np.concatenate((gt_gray * 255, pred_gray * 255, res * 255), axis=1)
    #return np.concatenate((gt_gray * 255, pred_gray * 255, res * 255), axis=1)

def test_mask(path):
    search_path = os.path.join(path, '**', '*.png')
    for idx, i in enumerate(glob.iglob(search_path, recursive=True)):
        if 'gt' in i:
            #print(i)
            #print(os.path.exists(i.replace('gt', 'pred')))
            print(i, idx)
            #gt = cv2.imread(i, -1)
            #pred = cv2.imread(i.replace('gt', 'pred'), -1)
            #img = check_mask(gt, pred)
            #cv2.imwrite('result/{}.png'.format(idx), img)

class LossVariance(nn.Module):
    """ The instances in target should be labeled
    """
    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        return loss



#test_mask('save')

#import time
#start = time.time()
#segment_level_loss(gt, pred)
#finish = time.time()
#print(finish - start)