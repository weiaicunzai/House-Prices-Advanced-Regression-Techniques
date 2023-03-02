import os
import zipfile
import shutil
import glob
import re

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import numpy as np

class Glas(Dataset):
    def __init__(self, path, image_set, transforms=None, download=False):
        url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        md5 = '495b2a9f3d694545fbec06673fb3f40f'
        #self.weight_map_path = '/data/hdd1/by/FullNet-varCE/data/GlaS/weight_maps/train'
        #self.weight_map_path_val = '/data/hdd1/by/FullNet-varCE/data/GlaS/weight_maps/val'
        self.weight_map_path = os.path.join(path, 'weight_maps', 'train')
        self.weight_map_path_val = os.path.join(path, 'weight_maps', 'val')

        if download:
            download_url(url, path, file_name, md5=md5)

        self.class_names = ['background', 'gland']
        #self.class_names = ['background', 'gland', 'cnt']
        # self.ignore_index = -100
        self.ignore_index = 255
        self.class_num = len(self.class_names)

        data_folder = os.path.join(path, 'Warwick QU Dataset (Released 2016_07_08)')
        if not os.path.exists(data_folder):
            if not os.path.exists(os.path.join(path, file_name)):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

            with zipfile.ZipFile(os.path.join(path, file_name), "r") as f:
                f.extractall(path=path)

        self.images = []
        self.labels = []
        self.weight_maps = []
        search_path = os.path.join(data_folder, '**', '*.bmp')
        #image_re = re.escape(image_set + '_[0-9]+\.bmp')
        #image_re = image_set + '_[0-9]+\.bmp'
        #label_re = re.escape(image_set + '_[0-9]+_anno\.bmp')
        if image_set not in ['train', 'testA', 'testB', 'val']:
            raise ValueError('wrong image_set argument')
        label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        #label_re =  'anno' + '\.bmp'
        if image_set == 'val':
            label_re = 'test[A|B]' +  '_[0-9]+_' + 'anno' + '\.bmp'
        self.image_set = image_set

        self.image_names = []
        for bmp in glob.iglob(search_path, recursive=True):
            if re.search(label_re, bmp):
                #print(bmp)
                self.labels.append(
                    self.construct_contour(cv2.imread(bmp, -1)))
                bmp = bmp.replace('_anno', '')
                self.image_names.append(os.path.basename(bmp))
            #elif re.search(image_re, bmp):
                self.images.append(cv2.imread(bmp, -1))

                if self.image_set == 'train':
                    bs_name = os.path.basename(bmp)
                    bs_name = bs_name.replace('.bmp', '_anno_weight.png')
                    #if 'train' in bs_name:
                    weight_map_filename = os.path.join(
                            self.weight_map_path, bs_name)
                    #else:
                    #    weight_map_filename = os.path.join(
                    #        self.weight_map_path_val, bs_name)

                    #print(bmp, weight_map_filename)
                    self.weight_maps.append(
                        cv2.imread(weight_map_filename, -1)
                    )

        assert len(self.images) == len(self.labels)
        self.transforms = transforms
        self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)
        self.times = 30000

    def construct_contour(self, label):
        if self.image_set == 'train' and self.class_num == 3:
            kernel = np.ones((3, 3), np.uint8)
            label[label > 0] = 1
            label_tmp = label.copy()
            label = cv2.erode(label, kernel, iterations=4)
            label_tmp[label_tmp > 0] = 1
            cnt = label_tmp - label
            label[cnt == 1] = 2

            return label
        else:
            label[label > 0] = 1
            return label

    def __len__(self):
        if self.image_set == 'train':
            return len(self.images) * self.times
        else:
            return len(self.images)

    def __getitem__(self, index):

        if self.image_set == 'train':
            index = index % len(self.images)

        #print(len(self))
        image = self.images[index]
        label = self.labels[index]
        #print(np.unique(label))
        #label[label > 0] = 1



        if self.image_set != 'train':
            img_meta = self.transforms(image, label)
            img_meta['img_name'] = self.image_names[index]
            return img_meta

        else:
            if self.transforms is not None:
                weight_map = self.weight_maps[index]
                #print(self.transforms)
                image, label, weight_map = self.transforms(image, label, weight_map)
                #h, w = label.shape
                #weight_map = cv2.resize()
                #weight_map = torch.from_numpy(weight_map)
            return image, label, weight_map








#import sys
#sys.path.append(os.getcwd())
#import transforms
#
#crop_size=(480, 480)
#trans = transforms.Compose([
#            transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=0.5),
#            transforms.Resize(range=[0.5, 1.5]),
#            # transforms.RandomRotation(degrees=90, expand=False),
#            transforms.RandomRotation(degrees=90, expand=True),
#            transforms.RandomCrop(crop_size=crop_size, cat_max_ratio=0.75, pad_if_needed=True),
#            transforms.RandomVerticalFlip(),
#            transforms.RandomHorizontalFlip(),
#            transforms.RandomApply(
#                transforms=[transforms.PhotoMetricDistortion()]
#            ),
#            #transforms.ToTensor(),
#            #transforms.Normalize(settings.MEAN, settings.STD)
#        ])
#
##import transforms
##trans = transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=1)
#data = Glas('data', 'train', download=True, transforms=trans)
#print(len(data))
###
#import random
#image, label = random.choice(data)
##print(label[label==255])
##
#cv2.imwrite('test.jpg', image)
#cv2.imwrite('test1.jpg', label * 100)

#print()
#print(name)
#print(np.unique(label))
#
##label[label != 0] = 255
#label = label / label.max() * 255
#
#cv2.imwrite('img.jpg', image)
#cv2.imwrite('label.jpg', label)
#
#
#gt_label = cv2.imread(name, -1)
#gt_label[gt_label != 0] = 255
#
#cv2.imwrite('label_gt.jpg', label)
#
#name = name.replace('_anno', '')
#img_gt = cv2.imread(name, -1)
#
#cv2.imwrite('img_gt.jpg', img_gt)