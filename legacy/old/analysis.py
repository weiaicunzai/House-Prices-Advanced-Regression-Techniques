import os
import glob

import cv2
import numpy as np
import skimage.morphology as morph

import utils
from loss import segment_level_loss


def test_mask(path, op):
    search_path = os.path.join(path, '**', '*.png')
    save_path = os.path.join(path, 'anal')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    no_glance = 0
    for i in glob.iglob(search_path, recursive=True):
        file_name = os.path.basename(i)
        if 'gt' in file_name:
            count += 1
            #print(i)
            #print(os.path.exists(i.replace('gt', 'pred')))
            gt_path = i
            dir_name = os.path.dirname(i)
            file_name = file_name.replace('gt', 'pred')
            pred_path = os.path.join(dir_name, file_name)

            img_path = os.path.join(dir_name, file_name.replace('pred', 'img'))

            gt = cv2.imread(gt_path, -1)
            pred = cv2.imread(pred_path, -1)
            img = cv2.imread(img_path, -1)

            gt_colors = utils.assign_colors(gt, np.max(gt))
            pred_colors = utils.assign_colors(pred, np.max(pred))

            # jjjjjjjjjjjjjjj
            #gt_res, pred_res, res, cc = res


            #gt_res = morph.remove_small_objects(gt_res == 1, 100)  # remove small object
            #gt_res = morph.label(gt_res, connectivity=2)

            #pred_res = morph.remove_small_objects(pred_res == 1, 100)  # remove small object
            #pred_res = morph.label(pred_res, connectivity=2)

            #cc = morph.remove_small_objects(cc == 1, 100)  # remove small object
            #cc = morph.label(cc, connectivity=2)
            #print(i, res.max())
            #print(i, res.max())
            res = check_mask(gt.copy(), pred.copy(), op=op)
            res = morph.remove_small_objects(res == 1, 100)  # remove small object
            res = morph.label(res, connectivity=2)

            if res.max() == 0:
                no_glance += 1

            res_colors = utils.assign_colors(res, np.max(res))
            #pred_res = utils.assign_colors(pred_res, np.max(pred_res))
            #gt_res = utils.assign_colors(gt_res, np.max(gt_res))
            #cc = utils.assign_colors(cc, np.max(cc))


            # draw contour
            gt[gt != 0] = 1
            im2, contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (40,20, 123), 3)
            cv2.drawContours(pred_colors, contours, -1, (40,20, 123), 3)
            #print(np.unique(gt))

            #gt = cv2.imread(i, -1)
            #pred = cv2.imread(i.replace('gt', 'pred'), -1)
            #img = check_mask(gt, pred)
            file_name = os.path.basename(i)
            file_name = file_name.replace('gt', '')
            img = np.concatenate((img, gt_colors, pred_colors, res_colors), axis=1)
            #img = np.concatenate((img, gt_colors, pred_colors, cc, gt_res, pred_res, res_colors), axis=1)
            cv2.imwrite(os.path.join(save_path, file_name), img)

    print(no_glance)
    print(count)

def check_mask(gt, pred, op):
    #gt_gray = cv2.cvtColor(gt , cv2.COLOR_BGR2GRAY)
    #pred_gray = cv2.cvtColor(pred , cv2.COLOR_BGR2GRAY)
    #gt_gray[gt_gray < 125] = 0
    #gt_gray[gt_gray >= 125] = 1
    #pred_gray[pred_gray < 125] = 0
    #pred_gray[pred_gray >= 125] = 1
#    print(gt_gray.shape, pred_gray.shape)
    gt_cp = gt.copy()
    pred_cp = pred.copy()
    gt_cp[gt_cp != 0] = 1
    pred_cp[pred_cp != 0] = 1
    res = segment_level_loss(gt_cp, pred_cp, op)
    return res
#    #res = np.stack([res, res, res], axis=2)
#
    #im2, contours, hierarchy = cv2.findContours(gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(pred, contours, -1, (40,20,255), 3

#test_mask('/data/by/House-Prices-Advanced-Regression-Techniques/result/Thursday_21_January_2021_09h_24m_07s')
#test_mask('/data/by/House-Prices-Advanced-Regression-Techniques/result/Tuesday_26_January_2021_20h_20m_03s')
#test_mask('/data/by/House-Prices-Advanced-Regression-Techniques/result/Wednesday_27_January_2021_04h_29m_53s')
#test_mask('/data/by/House-Prices-Advanced-Regression-Techniques/result/Tuesday_26_January_2021_14h_15m_38s')
test_mask('/data/by/House-Prices-Advanced-Regression-Techniques/result/Tuesday_26_January_2021_08h_11m_14s', 'xor')
