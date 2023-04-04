import os
import glob
import csv

import numpy as np
import cv2
from torch.utils.data import Dataset

import sys
sys.path.append(os.getcwd())

import scipy.io as sio

datasets = {
    'CoCaHis' : '/data/smb/syh/colon_dataset/CoCaHis/cocahis',
    'GlasTrain' : '/data/smb/syh/gland_segmentation/Glas/Warwick QU Dataset (Released 2016_07_08)',
    'GlasTest' : '/data/smb/syh/gland_segmentation/Glas/Warwick QU Dataset (Released 2016_07_08)',
    'CRAGTrain' : '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/',
    'CRAGTest' : '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/',
    'EBHI' : '/data/smb/syh/gland_segmentation/EBHI-SEG/',
    'ProstateSin' : '/data/smb/syh/gland_segmentation/Singapore/gland_seg/gland_segmentation_dataset/cropped_patches__complete_and_partial_glands_50_50_512_tars/imgs',
    'RINGS' : '/data/smb/syh/gland_segmentation/RINGS',
}

class BaseDataset(Dataset):
    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        self.path = path
        self.repeat_factor = repeat_factor
        self.return_label = return_label
        self.binary = binary

        img_infos = self.read_img_infos()
        self.img_infos = self.expand_img_infos(img_infos)

    def expand_img_infos(self, img_infos):

        for _ in range(self.repeat_factor - 1):
            img_infos.extend(img_infos)

        return img_infos

    def read_img_infos(self, path):
        """ return two lists: img_names,  seg_maps """
        raise NotImplementedError

    def read_img(self, img_filename):
        raise NotImplementedError

    def read_segmap(self, segmap_filename):
        raise NotImplementedError

    def read_label(self, file_idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        img_info = self.img_infos[idx]
        #img_filename = self.img_filenames[idx]
        #segmap_filename = self.segmap_filenames[idx]
        img = self.read_img(img_info['img_filename'])

        if self.return_label is None:
            return img

        elif self.return_label == 'pixel':
            segmap = self.read_segmap(img_info['label'])
            if segmap is None:
                print(img_info['label'])

            if self.binary:
                segmap[segmap > 0] = 1

            return img, segmap

        elif self.return_label == 'img':
            cls_idx = self.read_label(idx)
            if not self.binary:
                #raise ValueError('{} does not support img level non-binary labels'.format(self))
                return img, cls_idx
            else:
                if idx > 0:
                    return img, 1 # cancer
                else:
                    return img, 0

        else:
            raise ValueError('wrong value error')



############
# segmentation datasets
class CoCaHis(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


         for img in glob.iglob(os.path.join(self.path, '**', '*.jpg'), recursive=True):
            img_info = {}
            assert os.path.isfile(img)
            segmap = img.replace('images', 'labels').replace('.jpg', '.png')
            assert os.path.isfile(segmap)

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        # normal vs cancer
        return 1

    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap


class GlasTrain(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


        for img in glob.iglob(os.path.join(self.path, '**', '*.bmp'), recursive=True):
            basename = os.path.basename(img)
            if 'train' not in basename:
                continue

            if '_anno' in img:
                continue

            img_info = {}
            assert os.path.isfile(img)
            #segmap = img.replace('images', 'labels').replace('.jpg', '.png')
            segmap = img.replace('.bmp', '_anno.bmp')
            assert os.path.isfile(segmap)

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        if not hasattr(self, 'cls_labels'):
            self.cls_labels = {}
            with open(os.path.join(self.path, 'Grade.csv')) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    self.cls_labels[row['name'] + '.bmp'] = row[' grade (GlaS)']


        img_info = self.img_infos[idx]
        img_filename = img_info.get('img_filename')
        #print(img_filename)
        basename = os.path.basename(img_filename)
        cls_name = self.cls_labels[basename]
        if cls_name.strip() == 'benign':
            return 0
        else:
            return 1

    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap


class GlasTest(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


         for img in glob.iglob(os.path.join(self.path, '**', '*.bmp'), recursive=True):
            basename = os.path.basename(img)
            if 'test' not in basename:
                continue

            if '_anno' in img:
                continue

            img_info = {}
            assert os.path.isfile(img)
            #segmap = img.replace('images', 'labels').replace('.jpg', '.png')
            segmap = img.replace('.bmp', '_anno.bmp')
            assert os.path.isfile(segmap)

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        if not hasattr(self, 'cls_labels'):
            self.cls_labels = {}
            with open(os.path.join(self.path, 'Grade.csv')) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    #name,
                    self.cls_labels[row['name'] + '.bmp'] = row[' grade (GlaS)']
                    #print(row['name'], row[' grade (GlaS)'])
                    #print(row)
                    #print(row.split(','))

        img_info = self.img_infos[idx]
        img_filename = img_info.get('img_filename')
        #print(img_filename)
        basename = os.path.basename(img_filename)
        cls_name = self.cls_labels[basename]
        if cls_name.strip() == 'benign':
            return 0
        else:
            return 1

    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap


class CRAGTrain(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


         for img in glob.iglob(os.path.join(self.path, '**', '*.png'), recursive=True):

            basename = os.path.basename(img)
            if 'train' not in basename:
                continue

            if 'Annotation' in img:
                continue

            if 'Overlay' in img:
                continue

            img_info = {}
            assert os.path.isfile(img)
            segmap = img.replace('Images', 'Annotation')
            assert os.path.isfile(segmap)

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        raise ValueError('no img level label available')


    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap

class CRAGTest(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


         for img in glob.iglob(os.path.join(self.path, '**', '*.png'), recursive=True):

            basename = os.path.basename(img)
            if 'test' not in basename:
                continue

            if 'Annotation' in img:
                continue

            if 'Overlay' in img:
                continue

            img_info = {}
            assert os.path.isfile(img)
            segmap = img.replace('Images', 'Annotation')
            assert os.path.isfile(segmap)

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        raise ValueError('no img level label available')


    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap

class EBHI(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
         """ return two lists: img_names,  seg_maps """
         img_infos = []


         for img in glob.iglob(os.path.join(self.path, '**', '*.png'), recursive=True):

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

            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

         return img_infos

    def read_label(self, idx):
        raise ValueError('no img level label available')


    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap



class ProstateSin(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
        """ return two lists: img_names,  seg_maps """
        img_infos = []

        for img in glob.iglob(os.path.join(self.path, '**', 'img', '*.png'), recursive=True):
            img_info = {}

            assert os.path.isfile(img)

            segmap = img.replace('.png', '_mask.png')
            #img =
            #print(img)
            segmap_dir = os.path.dirname(segmap)
            segmap_dir = os.path.dirname(segmap_dir)
            segmap_dir = os.path.join(segmap_dir, 'mask')
            segmap = os.path.join(segmap_dir, os.path.basename(segmap))
            #print(img)
            #print(seg_map)
            # imgs.append(img)
            # seg_maps.append(segmap)
            assert os.path.isfile(segmap)


            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)

        return img_infos

    def read_label(self, idx):
        raise ValueError('no img level label available')


    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap


class RINGS(BaseDataset):

    def __init__(self, path, repeat_factor=1, return_label=None, binary=True):
        super().__init__(path, repeat_factor=repeat_factor, return_label=return_label, binary=binary)

    def read_img_infos(self):
        """ return two lists: img_names,  seg_maps """
        img_infos = []

        #path = '/data/smb/syh/gland_segmentation/RINGS'
        #imgs = []
        #seg_maps = []
        for img in glob.iglob(os.path.join(self.path, '**', 'IMAGES', '*.png'), recursive=True):
            img_info = {}
            # imgs.append(img)
            segmap = img.replace('IMAGES', 'MANUAL GLANDS')
            # seg_maps.append(seg_map)
            assert os.path.isfile(segmap)
            img_info['img_filename'] = img
            img_info['label'] = segmap
            img_infos.append(img_info)
            assert os.path.isfile(segmap)

        #path = '/data/smb/syh/gland_segmentation/Singapore/gland_seg/gland_segmentation_dataset/cropped_patches__complete_and_partial_glands_50_50_512_tars/imgs'
        #path = '/data/smb/syh/gland_segmentation/RINGS'
        #for img in glob.iglob(os.path.join(path, '**', 'img', '*.png'), recursive=True):
        #    img_info = {}

        #    assert os.path.isfile(img)

        #    segmap = img.replace('.png', '_mask.png')
        #    #img =
        #    #print(img)
        #    segmap_dir = os.path.dirname(segmap)
        #    segmap_dir = os.path.dirname(segmap_dir)
        #    segmap_dir = os.path.join(segmap_dir, 'mask')
        #    segmap = os.path.join(segmap_dir, os.path.basename(segmap))
        #    #print(img)
        #    #print(seg_map)
        #    # imgs.append(img)
        #    # seg_maps.append(segmap)
        #    assert os.path.isfile(segmap)


        #    img_info['img_filename'] = img
        #    img_info['label'] = segmap
        #    img_infos.append(img_info)

        return img_infos

    def read_label(self, idx):
        raise ValueError('no img level label available')


    def read_img(self, img_filename):
        img = cv2.imread(img_filename)
        return img

    def read_segmap(self, segmap_filename):
        segmap = cv2.imread(segmap_filename, -1)
        return segmap


    def get_prostate_rings(self):
        path = '/data/smb/syh/gland_segmentation/RINGS'
        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', 'IMAGES', '*.png'), recursive=True):
            imgs.append(img)
            seg_map = img.replace('IMAGES', 'MANUAL GLANDS')
            seg_maps.append(seg_map)

        return imgs, seg_maps











# #cocahis = ProstateSin(datasets.get('ProstateSin'), repeat_factor=1, return_label='pixel', binary=True)
# cocahis = RINGS(datasets.get('RINGS'), repeat_factor=1, return_label='pixel', binary=True)
# #cocahis = CRAGTrain(datasets.get('CRAGTrain'), repeat_factor=1, return_label='pixel', binary=True)
cocahis = EBHI(datasets.get('EBHI'), repeat_factor=1, return_label='pixel', binary=True)

for i in cocahis:
    pass
# #
# #
# #
# ## test code
# #
# import random
# img, segmap = random.choice(cocahis)
# print(len(cocahis), np.unique(segmap))
# #
# segmap[segmap > 0] = 255
# #h, w = img.shape[:2]
# #h = int(h / 4)
# #w = int(w / 4)

# img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
# segmap = cv2.resize(segmap, (0, 0), fx=0.25, fy=0.25)
# #
# cv2.imwrite('test.jpg', img)
# cv2.imwrite('test.png', segmap)
