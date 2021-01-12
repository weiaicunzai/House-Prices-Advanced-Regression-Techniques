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

class Glas(nn.Module):
    def __init__(self, path, image_set, transforms=None, download=False):
        url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        md5 = '495b2a9f3d694545fbec06673fb3f40f'

        if download:
            download_url(url, path, file_name, md5=md5)

        self.class_names = ['background', 'gland']
        self.ignore_index = -100
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
        if image_set not in ['train', 'testA', 'testB']:
            raise ValueError('wrong image_set argument')
        label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        self.names = []
        for bmp in glob.iglob(search_path, recursive=True):
            if re.search(label_re, bmp):
                self.labels.append(cv2.imread(bmp, -1))
                bmp = bmp.replace('_anno', '')
            #elif re.search(image_re, bmp):
                self.images.append(cv2.imread(bmp, -1))

                self.names.append(bmp)
        assert len(self.images) == len(self.labels)
        self.transforms = transforms


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        print(self.names[index])

        image = self.images[index]
        label = self.labels[index]
        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label, self.names[index]








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