import random
import os
import pickle

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import lmdb

import sys
sys.path.append(os.getcwd())
from conf import settings



#from transforms_pretrain import RandomScaleCrop
import transforms_pretrain




def training_transforms():
    trans = transforms_pretrain.Compose([
            #transforms_pretrain.EncodingLable(),
            transforms_pretrain.RandomHorizontalFlip(),
            transforms_pretrain.RandomVerticalFlip(),
            transforms_pretrain.RandomRotation(15, fill=0),
            transforms_pretrain.ColorJitter(0.4, 0.4),
            transforms_pretrain.RandomGaussianBlur(),
            transforms_pretrain.RandomScaleCrop(settings.IMAGE_SIZE),
            transforms_pretrain.ToTensor(),
            transforms_pretrain.Normalize(settings.MEAN, settings.STD),
        ])

    return trans


class PreTraining(Dataset):
    def __init__(self, path, image_set, transforms=None):
        lmdb_path = os.path.join(path, image_set)
        print(lmdb_path)
        self.env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=True, lock=False)

        cache_file = os.path.join(lmdb_path, '_cache')

        if os.path.isfile(cache_file):
            self.image_names = pickle.load(open(cache_file, 'rb'))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names= [key for key in txn.cursor().iternext(keys=True, values=False)]
                pickle.dump(self.image_names, open(cache_file, 'wb'))

        self.transforms = transforms

        self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)
        self.image_set = image_set
        #self.class_names = [str(i) for i in range(2)]
        self.class_names = ['background', 'gland']

        self.ignore_index = -100
        self.class_num = len(self.class_names)
    #def _store_label(self, img):
    #    self.label = label.copy()

    def _extract_label(self, img):
        #self.label
        #    "expect input image to be a 512x512 numpy array"
        row_idx = random.choice(range(3))
        col_idx = random.choice(range(3))

        H, W = img.shape[:2]
        assert H, W #only consider the condition when H == W

        label_size = int(H / 3)

        #if row_idx == 2:
        #    row_idx = W - label_size - 1
        #else:
        row_idx *= label_size
        col_idx *= row_idx
        if row_idx + label_size !=  H - 1:
            row_idx == H - label_size - 1
        if col_idx + label_size !=  W - 1:
            col_idx == W - label_size - 1

        #if col_idx == 2:
        #col_idx = H - label_size - 1
        label = img[row_idx:row_idx + label_size, col_idx:col_idx + label_size]
        cls_id = col_idx + 1
        return label, row_idx, col_idx, label_size

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:

            if idx == 92:
                idx = 91
            img_bytes = txn.get(self.image_names[idx])
            #print(self.image_names[idx], idx)
            #print(self.image_names[idx])
            file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if self.transforms:
                for trans in self.transforms.transforms:
                    if isinstance(trans, transforms_pretrain.RandomScaleCrop):
                        img = trans(img)

                        #label = img.copy()
                        #print(label.shape)
                    else:
                        img = trans(img)
            #else:
            #    label = img.copy()

            #label, row_idx, col_idx, label_size = self._extract_label(label)

            ##img[row_idx:row_idx + label_size, col_idx:col_idx + label_size] = 0
            #img1 = img.copy()
            #print(img1.shape)
            #img1[row_idx:row_idx + label_size, col_idx:col_idx + label_size] = 0
            #mask = np.random.random((473, 473)) < 0.15
            #print(label.shape, mask.shape)
            #src = img.copy()
            #img[mask] = 0
            ##label[mask] = 0
            ##label = torch.from_numpy(mask).long()
            #print(label.shape, label.dtype)
            ##print(label.shape)
            #cv2.imwrite('tmp/random{}.jpg'.format(idx), img)
            #cv2.imwrite('tmp/src{}.jpg'.format(idx), src)
            #cv2.imwrite('tmp/crop{}.jpg'.format(idx), img1)
            #print(label.shape, row_idx, col_idx, label_size)
            #cv2.imwrite('tmp/{}label{}.jpg'.format(idx, row_idx, col_idx), label)

            return img


#for i in

#trans = training_transforms()
#dataset = PreTraining('/data/by/House-Prices-Advanced-Regression-Techniques/data/pre_training/', 'train', trans)
#for i in range(len(dataset)):
#    print(dataset[i].shape)
##
#import random
#import os
#a = random.sample(range(len(dataset)), k=5)
#for i in a:
#    print(dataset[i][0].shape)