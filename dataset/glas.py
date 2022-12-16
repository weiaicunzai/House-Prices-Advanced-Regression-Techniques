import os
import zipfile
import shutil
import glob
import re

import cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import numpy as np

class Glas(Dataset):
    def __init__(self, path, image_set, transforms=None, download=False):
        url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        md5 = '495b2a9f3d694545fbec06673fb3f40f'

        if download:
            download_url(url, path, file_name, md5=md5)

        self.class_names = ['background', 'gland']
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
        search_path = os.path.join(data_folder, '**', '*.bmp')
        #image_re = re.escape(image_set + '_[0-9]+\.bmp')
        #image_re = image_set + '_[0-9]+\.bmp'
        #label_re = re.escape(image_set + '_[0-9]+_anno\.bmp')
        if image_set not in ['train', 'testA', 'testB', 'val']:
            raise ValueError('wrong image_set argument')
        label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        if image_set == 'val':
            label_re = 'test[A|B]' +  '_[0-9]+_' + 'anno' + '\.bmp'

        self.image_names = []
        for bmp in glob.iglob(search_path, recursive=True):
            if re.search(label_re, bmp):
                self.labels.append(cv2.imread(bmp, -1))
                bmp = bmp.replace('_anno', '')
                self.image_names.append(os.path.basename(bmp))
            #elif re.search(image_re, bmp):
                self.images.append(cv2.imread(bmp, -1))

        assert len(self.images) == len(self.labels)
        self.transforms = transforms
        self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)
        self.image_set = image_set

        self.times = 30000

    def __len__(self):
        if self.image_set == 'train':
            return len(self.images) * self.times
        else:
            return len(self.images)

    def __getitem__(self, index):

        if self.image_set == 'train':
            index = index % len(self.images)

        image = self.images[index]
        label = self.labels[index]
        #print(np.unique(label))
        label[label > 0] = 1



        if self.image_set != 'train':
            img_meta = self.transforms(image, label)
            img_meta['img_name'] = self.image_names[index]
            return img_meta

        if self.transforms is not None:
                image, label = self.transforms(image, label)

        return image, label









#data = Glas('data', 'testB', download=True)
#print(len(data))
#
#image, label, name = data[17]
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