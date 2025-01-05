from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imsave

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) #分别对应通道 R G B
def weight_add(path,rate):
    gt = io.imread(path)
    # print(type(gt))
    gt = 1 * (gt > 0)
    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    # 【2】归一化
    c_weights /= c_weights.max()
    # 【3】得到c_w字典
    c_weights.tolist()
    cw = {}
    for i in range(len(c_weights)):
        cw[i]=c_weights[i]
    weightMap_ = UnetWeightMap(gt,rate, cw)
    return weightMap_

def weight_add_np(gt,rate):
    # print(type(gt))
    gt = 1 * (gt > 0)
    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    # 【2】归一化
    c_weights /= c_weights.max()
    # 【3】得到c_w字典
    c_weights.tolist()
    cw = {}
    for i in range(len(c_weights)):
        cw[i]=c_weights[i]
    weightMap_ = UnetWeightMap(gt,rate, cw)
    return weightMap_
def UnetWeightMap(mask,rate, wc=None, w0=10, sigma=5):
 
    mask_with_labels = label(mask)
    no_label_parts = mask_with_labels == 0
    label_ids = np.unique(mask_with_labels)[1:]
    # print(label_ids)
    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            # cv2.imwrite('{}.jpg'.format(label_id), (mask_with_labels != label_id) * 255)
            # 不等于label_id的是True，等于label_id的是False，计算True到False的距离
            distances[:, :, i] = distance_transform_edt(mask_with_labels != label_id)
            # cv2.imwrite('{}_distance.jpg'.format(label_id), distances[:, :, 1] / distances[:, :, 1].max() * 255)
        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        
        # d1 = d1 / d1.max()  * rate
        # d2 = d2 / d2.max() * rate
        # print(d1.mean(), d1.std(), d2.mean(), d2.std())
        # print((d1 + d2).max())
        sum = (d1 + d2) / (d1 + d2).max() * rate

        # weight_map = w0 * np.exp(-1/2 * ((d1+d2)/sigma) ** 2) * no_label_parts
        weight_map = w0 * np.exp(-1/2 * (sum / sigma) ** 2) * no_label_parts
        # weight_map = weight_map + np.ones_like(weight_map)

 
        # if wc is not None:
        #     class_weights = np.zeros_like(mask)
        #     for k, v in wc.items():
        #         class_weights[mask == k] = v
        #     weight_map = weight_map + class_weights
 
    else:
        weight_map = np.zeros_like(mask)
    return weight_map

def get_gt(path):
    gt = io.imread(path)
    gt = 1 * (gt > 0)
    return gt
    
# image_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_25_anno.bmp' 
image_x_path = '/home/xuxinan/mmCode2/evalAns/4eval_xor.png'
image_path = '/home/xuxinan/mmCode2/evalAns/4eval_pred.png'
# for rate in range(10, 101, 5):
#     w = weight_add(image_path,rate)
#     w = w / w.max() * 255.0
#     w = w.astype(np.uint8)
#     imsave(f'unet/{rate}unet_pred.png', w) 
#     w1 = get_gt(image_x_path)
#     ans = w * w1
#     ans = ans / ans.max() * 255.0
#     ans = ans.astype(np.uint8)
#     imsave(f'unet/{rate}unet_xor.png', ans) 
rate=30
w = weight_add(image_path,rate)
w = w / w.max() * 255.0
w = w.astype(np.uint8)
imsave(f'{rate}unet_pred.png', w) 
w1 = get_gt(image_x_path)
ans = w * w1
ans = ans / ans.max() * 255.0
ans = ans.astype(np.uint8)
imsave(f'{rate}unet_xor.png', ans) 
print("ok")
# print(np.unique(w), w.max(), w.min())
# w1 = get_gt(image_path)
# print(f"w: {w}, type(w): {type(w)}")
# print(f"w1: {w1}, type(w1): {type(w1)}")
# ans = w * w1
# print(ans.max(), 111)
# print(w.max(), w.min())
# print(np.unique(w))
# np.savetxt('unet3w.txt', w, fmt='%1.3f')
# print(np.unique(w))  # 在这里之前是挺多值的
# w = w / w.max() * 255.0
# ans = ans / ans.max() * 255.0
# print(w)
# print(np.unique(w))
# np.savetxt('unet3w1.txt', w, fmt='%1.3f')
# ans = ans.astype(np.uint8)
# w = w.astype(np.uint8)
# np.savetxt('unet3.txt', ans, fmt='%1.3f')
# imsave('unet3.png', ans) 
# imsave('unet3.png', w) 
