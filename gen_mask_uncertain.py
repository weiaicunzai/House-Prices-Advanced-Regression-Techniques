import os
import glob

import cv2
import torch
import numpy as np
import re


from losses import GlandContrastLoss



loss = GlandContrastLoss(4, ignore_idx=255, temperature=0.07)


def gen_xor_mask(pred, gt):
    return loss.segment_level_loss(gt=gt, pred=pred, op='xor', out_size=gt.shape[-2:])


def gen_uncertain(pred_logits, xor_mask):
    return loss.cal_uncertain_mask(pred_logits, xor_mask)
    #print(uncertain_mask.shape)





logits_path = 'pred_glas'
#logits_filename = 'testA_16.bmp_39.pt'
gt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'

save_folder = 'tmp'


# for i in glob.iglob(os.path.join(gt_path, '**', "*_anno.bmp"), recursive=True):
#     #print(i)
#     basename = os.path.basename(i).replace('.bmp', '.png')
#     gt = cv2.imread(i, -1)
#     #print(basename)
#     gt[gt>0] = 255
#     cv2.imwrite('gts/{}'.format(basename), gt)


# import sys; sys.exit()

count = 1
for logits_filename in os.listdir(logits_path):
    #pattern = "testA_17.+9999.+"
    #pattern = ".+9999.+"
    pattern = "testA_49.+"
    #if re.search(logits_filename, pattern) is None:
    if re.search(pattern, logits_filename) is None:
        continue


    count += 1
    #if count == 10:
       #import sys; sys.exit()
# gen xor mask

    gt_filename = logits_filename.split('.')[0] + '_anno.bmp'
    pred_logits = torch.load(os.path.join(logits_path, logits_filename))
    gt = cv2.imread(os.path.join(gt_path, gt_filename), -1)
    gt = cv2.resize(gt, (pred_logits.shape[-2:][::-1]), cv2.INTER_NEAREST)
    pred = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    gt_clone = gt.copy()
    gt_clone[gt_clone > 0] = 1
    xor_mask = gen_xor_mask(pred, gt_clone)
    xor_mask_filename = logits_filename.replace('.pt', '_xor_mask.png')

    print(os.path.join(save_folder, xor_mask_filename))
    cv2.imwrite(os.path.join(save_folder, xor_mask_filename), xor_mask * 255)




# gen uncertain
    xor_mask = torch.tensor(xor_mask).long().unsqueeze(0)
    uncertain = gen_uncertain(pred_logits.cpu(), xor_mask)
    uncertain = uncertain.squeeze(0).numpy()
    if uncertain.max() != 0:
        uncertain = uncertain / uncertain.max() * 255
    uncertain = cv2.applyColorMap(np.uint8(uncertain), cv2.COLORMAP_JET)
    uncertain_filename = logits_filename.replace('.pt', '_uncertain.png')
    cv2.imwrite(os.path.join(save_folder, uncertain_filename), uncertain)




# gen pred
    pred = pred_logits.softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy()
    pred_filename = logits_filename.replace('.pt', '_pred.png')
    cv2.imwrite(os.path.join(save_folder, pred_filename), pred * 255)

    #import sys; sys.exit()
    #print(xor_mask)
    #cv2.imwrite('heihei1.png', pred * 255)
    #cv2.imwrite('heihei2.png', gt / gt.max() * 255)
    #print(pred.shape, gt.shape)
