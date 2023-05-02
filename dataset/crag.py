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

class CRAG(Dataset):
    def __init__(self, path, image_set, transforms=None):
        # url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        # file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        # md5 = '495b2a9f3d694545fbec06673fb3f40f'
        #self.weight_map_path = '/data/hdd1/by/FullNet-varCE/data/GlaS/weight_maps/train'
        #self.weight_map_path_val = '/data/hdd1/by/FullNet-varCE/data/GlaS/weight_maps/val'
        #self.weight_map_path = os.path.join(path, 'weight_maps', 'train')
        self.weightmap_path = '/data/smb/syh/gland_segmentation/CRAGV2/weight_maps/'
        #self.weight_map_path_val = os.path.join(path, 'weight_maps', 'val')

        #if download:
            #download_url(url, path, file_name, md5=md5)

        #self.class_names = ['background', 'gland']
        self.class_names = ['background', 'gland', 'cnt']
        # self.ignore_index = -100
        self.ignore_index = 255
        self.class_num = len(self.class_names)

        # data_folder = os.path.join(path, 'Warwick QU Dataset (Released 2016_07_08)')
        # if not os.path.exists(data_folder):
        #     if not os.path.exists(os.path.join(path, file_name)):
        #         raise RuntimeError('Dataset not found or corrupted.' +
        #                            ' You can use download=True to download it')

        #     with zipfile.ZipFile(os.path.join(path, file_name), "r") as f:
        #         f.extractall(path=path)

        self.image_set = image_set
        self.images = []
        self.labels = []
        self.weight_maps = []
        if image_set == 'train':
            search_path = os.path.join(path, image_set, '**', '*.png')

        elif image_set == 'all':
            search_path = os.path.join(path, '**', '*.png')
        else:
            search_path = os.path.join(path, 'valid', '**', '*.png')
        # if image_set not in ['train', 'testA', 'testB', 'val']:
            # raise ValueError('wrong image_set argument')
        # label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        #label_re =  'anno' + '\.bmp'
        # if image_set == 'val':
            # label_re = 'test[A|B]' +  '_[0-9]+_' + 'anno' + '\.bmp'

        self.image_names = []
        count = 0
        for img_filename in glob.iglob(search_path, recursive=True):
            if 'Images' not in img_filename:
                continue

            count += 1
            #if image_set not in ['train', 'all']:
            #    if count > 5:
            #        continue
            self.images.append(cv2.imread(img_filename, -1))
            #if re.search(label_re, bmp):
                #print(bmp)
            segmap_filename = img_filename.replace('Images', 'Annotation')
            self.labels.append(
                    self.construct_contour(cv2.imread(segmap_filename, -1)))
            #elif re.search(image_re, bmp):
            basename = os.path.basename(img_filename)
            self.image_names.append(basename)
            weightmap_filename = os.path.join(self.weightmap_path, basename.replace('.png', '_weight.png'))
            # if image_set == 'train':


            print(count)
            if image_set in ['train', 'all']:
                self.weight_maps.append(
                    cv2.imread(weightmap_filename, -1))

                #if self.image_set == 'train':
                #    bs_name = os.path.basename(bmp)
                #    bs_name = bs_name.replace('.bmp', '_anno_weight.png')
                #    #if 'train' in bs_name:
                #    weight_map_filename = os.path.join(
                #            self.weight_map_path, bs_name)
                #    #else:
                #    #    weight_map_filename = os.path.join(
                #    #        self.weight_map_path_val, bs_name)

                #    #print(bmp, weight_map_filename)
                #    self.weight_maps.append(
                #        cv2.imread(weight_map_filename, -1)
                #    )

        assert len(self.images) == len(self.labels)
        self.transforms = transforms
        self.mean = (0.8572969 , 0.72236917, 0.83065554),
        self.std = (0.09965163, 0.17253199, 0.1335271)
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
        elif self.image_set == 'train' and self.class_num == 2:
            kernel = np.ones((3, 3), np.uint8)
            label[label > 0] = 1
            # prevent glands clustered together when resize the img to a smaller size
            label = cv2.erode(label, kernel, iterations=1)

            return label

        else:
            label[label > 0] = 1
            return label

    def __len__(self):
        if self.image_set in ['train', 'all']:
            return len(self.images) * self.times
        else:
            return len(self.images)

    def __getitem__(self, index):

        if self.image_set in ['train', 'all']:
            index = index % len(self.images)

        image = self.images[index]
        label = self.labels[index]


        if self.image_set not in ['train', 'all']:
            img_meta = self.transforms(image, label)
            #print(index, self.image_set, len(self), len(self.images))
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
            #return image, label


# import sys
# sys.path.append(os.getcwd())
# import utils
# dataset = CRAG('/data/smb/syh/gland_segmentation/CRAGV2/CRAG/', image_set='train')
# print(len(dataset))
