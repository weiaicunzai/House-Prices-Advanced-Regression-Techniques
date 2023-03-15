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


class PreTraining(Dataset):
    def __init__(self, image_set, transforms=None):

        if image_set == 'train':
            imgs, seg_maps = self.get_data_pair()
        else:
            imgs, seg_maps = self.get_glas_val()

        #print(imgs)
        #for i in imgs:
            #print(i)

        self.imgs = imgs
        #print(self.imgs[:10])
        #for i in self.imgs[:10]:
            #print(i)
        self.seg_maps = seg_maps

        #print()
        #for i in self.imgs[:2]:
            #print(i)

        self.image_set = image_set
        #url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        #file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        #md5 = '495b2a9f3d694545fbec06673fb3f40f'
        #self.weight_map_path = '/data/hdd1/by/FullNet-varCE/data/GlaS/weight_maps/train'

        #if download:
        #    download_url(url, path, file_name, md5=md5)

        self.class_names = ['background', 'gland']
        #self.class_names = ['background', 'gland', 'cnt']
        ## self.ignore_index = -100
        self.ignore_index = 255
        self.class_num = len(self.class_names)

        #data_folder = os.path.join(path, 'Warwick QU Dataset (Released 2016_07_08)')
        #if not os.path.exists(data_folder):
        #    if not os.path.exists(os.path.join(path, file_name)):
        #        raise RuntimeError('Dataset not found or corrupted.' +
        #                           ' You can use download=True to download it')

        #    with zipfile.ZipFile(os.path.join(path, file_name), "r") as f:
        #        f.extractall(path=path)

        #self.images = []
        #self.labels = []
        #self.weight_maps = []
        #search_path = os.path.join(data_folder, '**', '*.bmp')
        ##image_re = re.escape(image_set + '_[0-9]+\.bmp')
        ##image_re = image_set + '_[0-9]+\.bmp'
        ##label_re = re.escape(image_set + '_[0-9]+_anno\.bmp')
        #if image_set not in ['train', 'testA', 'testB', 'val']:
        #    raise ValueError('wrong image_set argument')
        #label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        #if image_set == 'val':
        #    label_re = 'test[A|B]' +  '_[0-9]+_' + 'anno' + '\.bmp'
        #self.image_set = image_set

        #self.image_names = []
        #for bmp in glob.iglob(search_path, recursive=True):
        #    if re.search(label_re, bmp):
        #        self.labels.append(
        #            self.construct_contour(cv2.imread(bmp, -1)))
        #        bmp = bmp.replace('_anno', '')
        #        self.image_names.append(os.path.basename(bmp))
        #    #elif re.search(image_re, bmp):
        #        self.images.append(cv2.imread(bmp, -1))

        #        if self.image_set == 'train':
        #            bs_name = os.path.basename(bmp)
        #            bs_name = bs_name.replace('.bmp', '_anno_weight.png')
        #            weight_map_filename = os.path.join(
        #                self.weight_map_path, bs_name)

        #            #print(bmp, weight_map_filename)
        #            self.weight_maps.append(
        #                cv2.imread(weight_map_filename, -1)
        #            )

        #assert len(self.images) == len(self.labels)
        self.transforms = transforms
        self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)

            #self.imgs = self.imgs[::10]
        #imgs = []
        #seg_maps = []
        #for idx in range(len(self.imgs)):
        #    if self.image_set == 'train':
        #        if idx % 30 != 0:
        #            imgs.append(self.imgs[idx])
        #            seg_maps.append(self.seg_maps[idx])
        #    else:
        #        if idx % 30 == 0:
        #            imgs.append(self.imgs[idx])
        #            seg_maps.append(self.seg_maps[idx])


        #self.imgs = imgs
        #self.seg_maps = seg_maps

            #for idx, img in enumerate(self.imgs):
                #if idx % 10 != 0:
                    #imgs.append(img)
                    #seg_maps.append(self.seg_maps[idx])



        #self.times = 30000
        #import sys; sys.exit()

    def over_sampling(self, img, seg_map, over_samples):
        num_imgs = len(img)
        imgs = []
        seg_maps = []
        assert num_imgs < over_samples
        for i in range(int(over_samples / num_imgs)):
            imgs.extend(img)
            seg_maps.extend(seg_map)
        return imgs, seg_maps

    def get_data_pair(self):
        imgs = []
        seg_maps = []

        img, seg_map = self.get_prostate_sin()
        num_imgs_prostate_sin = len(img)
        imgs.extend(img)
        seg_maps.extend(seg_map)

        img, seg_map = self.get_prostate_rings()
        #num_imgs_rings = len(img)
        #img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        imgs.extend(img)
        seg_maps.extend(seg_map)
        #print(len(imgs))

        img, seg_map = self.get_glas()
        num_imgs_glas = len(img)
        #img, seg_map = self.over_sampling(img, seg_map, num_imgs_rings)
        #print(len(img))
        #img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        #print(len(img))
        imgs.extend(img)
        seg_maps.extend(seg_map)

        img, seg_map = self.get_crag()
        print(len(img))
        #img, seg_map = self.over_sampling(img, seg_map, num_imgs_rings)
        #img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        #print(len(img))
        print(len(img))
        #import sys; sys.exit()
        imgs.extend(img)
        seg_maps.extend(seg_map)

        return imgs, seg_maps

    def get_crag(self):
        path = '/data/hdd1/by/datasets/original/CRAG/train'
        path = '/data/hdd1/by/datasets/original/CRAGV2/CRAG/train'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, 'Images',  '**', '*.png'), recursive=True):
            #print(img)
            imgs.append(img)
            seg_map = img.replace('Images', 'Annotation')
            #print(seg_map)
            seg_maps.append(seg_map)

        return imgs, seg_maps

    def get_glas_val(self):
        path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'
        imgs = []
        seg_maps = []
        #count = 0
        for img in glob.iglob(os.path.join(path, '**', '*.bmp'), recursive=True):
            if '_anno.bmp' in img:
                continue

            if 'train' in img:
                continue

            #count += 1
            #if  count > 10:
            #    break
            imgs.append(img)
            seg_map = img.replace(
                '.bmp',
                '_anno.bmp'
            )
            #print(img)
            #print(seg_map)
            seg_maps.append(seg_map)

        return imgs, seg_maps

    def get_glas(self):
        path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'
        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', 'train*.bmp'), recursive=True):
            if '_anno.bmp' in img:
                continue

            imgs.append(img)
            seg_map = img.replace(
                '.bmp',
                '_anno.bmp'
            )
            #print(img)
            #print(seg_map)
            seg_maps.append(seg_map)

        return imgs, seg_maps


    def get_prostate_rings(self):
        path = '/data/smb/syh/gland_segmentation/RINGS'
        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', 'IMAGES', '*.png'), recursive=True):
            imgs.append(img)
            seg_map = img.replace('IMAGES', 'MANUAL GLANDS')
            seg_maps.append(seg_map)

        return imgs, seg_maps

    def get_prostate_sin(self):
        path = '/data/smb/syh/gland_segmentation/Singapore/gland_seg/gland_segmentation_dataset/cropped_patches__complete_and_partial_glands_50_50_512_tars/imgs'
        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', 'img', '*.png'), recursive=True):
            seg_map = img.replace('.png', '_mask.png')
            #img =
            #print(img)
            seg_map_dir = os.path.dirname(seg_map)
            seg_map_dir = os.path.dirname(seg_map_dir)
            seg_map_dir = os.path.join(seg_map_dir, 'mask')
            seg_map = os.path.join(seg_map_dir, os.path.basename(seg_map))
            #print(img)
            #print(seg_map)
            imgs.append(img)
            seg_maps.append(seg_map)

            #import sys; sys.exit()
            #print(seg_map)

        return imgs, seg_maps

    def __len__(self):
        return len(self.imgs)
        #if self.image_set == 'train':
            #return len(self.images) * self.times
        #else:
            #return len(self.images)

    def __getitem__(self, index):

        #if self.image_set == 'train':
        #    index = index % len(self.images)

        #image = self.images[index]
        #label = self.labels[index]
        #print(np.unique(label))
        #label[label > 0] = 1
        img = self.imgs[index]
        seg_map = self.seg_maps[index]

        image = cv2.imread(img)
        label = cv2.imread(seg_map, -1)

        if image is None:
            print(img)
        if label is None:
            print(seg_map)


        #if label
        #print(seg_map)
        #print(np.unique(label))
        label[label > 0] = 1



        if self.image_set != 'train':
            img_meta = self.transforms(image, label)
            img_meta['img_name'] = os.path.basename(self.imgs[index])
            #print(img_meta)
            return img_meta

        else:
            if self.transforms is not None:
                #weight_map = self.weight_maps[index]
                #print(self.transforms)
                #print('before', label.shape, image.shape, seg_map, img)
                image, label = self.transforms(image, label)
                #print(label.shape, image.shape, seg_map)
                #h, w = label.shape
                #weight_map = cv2.resize()
                #weight_map = torch.from_numpy(weight_map)
            return image, label

#dataset = PreTraining(
#    image_set='train'
#    #image_set='test'
#)
#
#print(len(dataset))
#
##dataset[]
##for i in range(100):
#import random
#for i in range(100):
#    #dataset[i]
#    random.choice(dataset)


#count = 0
#for img, label in dataset:
    #count += 1
    #if count % 100 ==0:
        #print(count)

    #assert img.shape[:2] == label.shape[:2]
#    pass