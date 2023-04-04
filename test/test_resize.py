import os
import sys
sys.path.append(os.getcwd())
import argparse

import cv2
import numpy as np

import transforms
import utils


trans = transforms.Resize(
    range=[0.5, 1.5],
    # size=(300, 400)
)





#def __init__(self, size=None, range=None, keep_ratio=True, min_size=None):

# segmap = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_22_anno.bmp'
# img = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_22.bmp'

# img = cv2.imread(img, -1)
# segmap = cv2.imread(segmap, -1)
# h, w = img.shape[:2]
# h, w = (300, 400)
# print(np.unique(segmap), img.shape)

# img, segmap = trans(img, segmap)

# print(np.unique(segmap))
# print(img.shape, segmap.shape, img.shape[0] / h,  img.shape[1] / w)

# cv2.imwrite('tmp/img.jpg', img)
# cv2.imwrite('tmp/segmap.png', segmap / segmap.max() * 255)


parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=64 * 4,
                    help='batch size for dataloader')
parser.add_argument('-net', type=str, help='if resume training')
parser.add_argument('-dataset', type=str, default='Glas', help='dataset name')
parser.add_argument('-download', action='store_true', default=False,
    help='whether to download camvid dataset')
parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
args = parser.parse_args()
print(args)

# args.


train_loader = utils.data_loader(args, 'train')
# train_loader.dataset.transforms = test_data_trans
dataset = train_loader.dataset
dataset.transforms = trans

for img, anno, weight_map in dataset:
    #print(img.shape, anno.shape, weight_map.shape)
    pass
