import random
import os
import pickle
import glob

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision

import sys
sys.path.append(os.getcwd())
from conf import settings
# print(os.getcwd())
# import utils

import scipy.io as sio


class CropPretraining(Dataset):
    def __init__(self, img_set, transforms=None):


        imgs, segmaps = self.get_data_pair()


        self.total_imgs = imgs

        self.total_segmaps = segmaps


        self.img_set = img_set
        self.class_names = ['background', 'gland']
        self.ignore_index = 255
        self.class_num = len(self.class_names)


        self.transforms = transforms
        self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)


        self.times = 30
        if self.img_set != 'train':
            # self.val_imgs = None
            #self.val_segmaps = None
            # sample_val
            self.sample_val()



    def get_data_pair(self):
        imgs = []
        seg_maps = []

        img, seg_map = self.get_ebhi()
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs), 'ebhi')

        # # glas train
        path = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/'
        img, seg_map = self.get_crops(path)
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs), 'glas train')

        # glas eval
        path = '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/'
        img, seg_map = self.get_crops(path)
        # index = int(len(img) * 0.6)
        # imgs.extend(img[0: index])
        # seg_maps.extend(seg_map[0: index])
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs), 'glas eval')

        # crag train
        #path = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/'
        #img, seg_map = self.get_crops(path)
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs), 'crag train')

        ## crag val
        #path = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/'
        #img, seg_map = self.get_crops(path)
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs), 'crag val')

        # s1-v1_crop
        path = '/data/smb/syh/gland_segmentation/PATH-DT-MSU/S1-v1_crop/'
        img, seg_map = self.get_crops(path)
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs), 's1-v1_crop')

        # s1-v2_crop
        path = '/data/smb/syh/gland_segmentation/PATH-DT-MSU/S1-v1_crop/'
        path = '/data/smb/syh/gland_segmentation/PATH-DT-MSU/S1-v2_crop/'
        img, seg_map = self.get_crops(path)
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs), 's1-v2_crop')

        return imgs, seg_maps


    def get_ebhi(self):

        imgs = []
        seg_maps = []
        path = '/data/smb/syh/gland_segmentation/EBHI-SEG/'
        for img in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):

            #basename = os.path.basename(img)
            dir_name = os.path.basename(os.path.dirname(img))
            if dir_name != 'image':
                continue

            # lack of label
            if 'GTGT2012149-2-400-001.png' in img:
                continue

            if 'GT2012149-2-400-001.png'  in img:
                continue


            img_info = {}
            assert os.path.isfile(img)
            segmap = img.replace('image', 'label')
            assert os.path.isfile(segmap)

            imgs.append(img)

            seg_maps.append(segmap)


        return imgs, seg_maps



    def sample_val(self):
        num_sample = len(self.total_imgs)
        k = int(len(self.total_imgs) * 0.1)
        idxes = random.sample(range(len(self.total_imgs)), k=k)

        val_imgs = [self.total_imgs[idx] for idx in idxes]
        val_segmaps = [self.total_segmaps[idx] for idx in idxes]

        self.val_imgs = val_imgs
        self.val_segmaps = val_segmaps


    def get_crops(self, path):
        imgs = []
        seg_maps = []

        for img in glob.iglob(os.path.join(path, '**', '*.jpg'), recursive=True):

           if 'Images' not in img:
               continue

           label_path = img.replace('Images', 'Annotation').replace('.jpg', '.png')

           imgs.append(img)
           seg_maps.append(label_path)

        return imgs, seg_maps

    def __len__(self):
        if self.img_set == 'train':
            return len(self.total_imgs) * self.times
        else:
            return len(self.val_imgs)

    def __getitem__(self, index):


        if self.img_set == 'train':
            index = index % len(self.total_imgs)
            img_filename = self.total_imgs[index]
            segmap_filename = self.total_segmaps[index]
        else:
            img_filename = self.val_imgs[index]
            segmap_filename = self.val_segmaps[index]

        img = cv2.imread(img_filename)
        # print(img_filename)
        # print(segmap_filename)
        segmap = cv2.imread(segmap_filename, -1)

        assert os.path.isfile(segmap_filename)
        segmap[segmap > 0] = 1

        if self.transforms is not None:
            img, segmap = self.transforms(img, segmap)

        # print(index)
        return img, segmap

def overlay(img, mask):

    overlay = np.zeros(img.shape, img.dtype)


    overlay[mask > 0] = (0, 255, 0)

    alpha = 0.7
    beta = 1 - alpha
    return cv2.addWeighted(img, alpha, overlay, beta, 0.0)






# dataset = CropPretraining(img_set='val')

# dataset.sample_val()

# import utils
# # dataset.transforms = utils.pretrain_test_transforms()
# dataset.transforms = utils.pretrain_training_transforms()


# count = 0
# for i in range(100):
#     img, segmap = random.choice(dataset)
#     # print(img.shape)
#     # print(segmap.shape)
#     # print(img.shape, segmap.shape)
#     res = overlay(img, segmap)
#     #res = cv2.resize(res, (0,0), fx=0.5, fy=0.5)
#     count += 1
#     # print(count)
#     cv2.imwrite('tmp/test{}.jpg'.format(count), res)
#print(type(output))
    #print(type(output))
    #print(output.shape)
    ##print(i.shape)
    #print(img.shape, segmap.shape)
    #count += 1

    #if count == 50:
    #    import sys; sys.exit()
