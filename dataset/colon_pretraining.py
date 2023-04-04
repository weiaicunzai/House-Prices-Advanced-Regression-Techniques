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

import scipy.io as sio




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

        # img, seg_map = self.get_lizeard()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)

        # img, seg_map = self.get_monusac()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)

        # img, seg_map = self.get_tnbc()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)

        # img, seg_map = self.get_cpm17()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)

        # img, seg_map = self.get_kumar()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)

        # img, seg_map = self.get_consep()
        # imgs.extend(img)
        # seg_maps.extend(seg_map)


        #img, seg_map = self.get_prostate_sin()
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))


        #img, seg_map = self.get_prostate_rings()
        ##num_imgs_rings = len(img)
        ##img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))
        #print(len(imgs))

        #img, seg_map = self.get_glas()
        #num_imgs_glas = len(img)
        ##img, seg_map = self.over_sampling(img, seg_map, num_imgs_rings)
        ##print(len(img))
        ##img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        ##print(len(img))
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))

        #img, seg_map = self.get_crag()
        ##img, seg_map = self.over_sampling(img, seg_map, num_imgs_rings)
        ##img, seg_map = self.over_sampling(img, seg_map, num_imgs_prostate_sin)
        ##print(len(img))
        ##print(len(img))
        ##import sys; sys.exit()
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))

        #img, seg_map = self.get_crag_val()
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))

        #for i in range(len(img))
        #for img, seg_map in zip(imgs, seg_maps):
        #    print(img)
        #    print(seg_map)
        #img, seg_map = self.get_crops()
        #imgs.extend(img)
        #seg_maps.extend(seg_map)
        #print(len(imgs))
        img, seg_map = self.get_ebhi()
        imgs.extend(img)
        seg_maps.extend(seg_map)
        print(len(imgs))


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
            #img_info['img_filename'] = img
            #img_info['label'] = segmap
            #img_infos.append(img_info)
            seg_maps.append(segmap)

        #path = '/data/hdd1/by/datasets/original/CRAG/train'
        #path = '/data/hdd1/by/datasets/original/CRAGV2/CRAG/train'
        #path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/train/'

        #imgs = []
        #seg_maps = []
        #for img in glob.iglob(os.path.join(path, 'Images',  '**', '*.png'), recursive=True):
        #    #print(img)
        #    imgs.append(img)
        #    seg_map = img.replace('Images', 'Annotation')
        #    #print(seg_map)
        #    seg_maps.append(seg_map)

        return imgs, seg_maps

    def get_crag(self):
        #path = '/data/hdd1/by/datasets/original/CRAG/train'
        #path = '/data/hdd1/by/datasets/original/CRAGV2/CRAG/train'
        path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/train/'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, 'Images',  '**', '*.png'), recursive=True):
            #print(img)
            imgs.append(img)
            seg_map = img.replace('Images', 'Annotation')
            #print(seg_map)
            seg_maps.append(seg_map)

        return imgs, seg_maps

    def get_crag_val(self):
        #path = '/data/hdd1/by/datasets/original/CRAG/train'
        #path = '/data/hdd1/by/datasets/original/CRAGV2/CRAG/train'
        path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/valid'

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

    def get_monusac(self):
        path = '/data/smb/syh/colon_dataset/MoNuSAC/MoNuSAC_images_and_annotations'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.tif'), recursive=True):
            if 'Overlay' in img:
                continue

            seg_map = img.replace('.tif', '.npy')

            imgs.append(img)
            seg_maps.append(seg_map)



        return imgs, seg_maps

    def get_tnbc(self):
        path = '/data/smb/syh/colon_dataset/TNBC/tnbc'

        imgs = []
        seg_maps = []
        label_path = '/data/smb/syh/colon_dataset/TNBC/tnbc/Labels/'
        for img in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            if 'Overlay' in img:
                continue

            bs_name = os.path.basename(img).replace('.png', '.mat')
            seg_map = os.path.join(label_path, bs_name)

            #print(img)
            #print(seg_map)
            imgs.append(img)
            seg_maps.append(seg_map)



        return imgs, seg_maps

    def get_kumar(self):
        path = '/data/smb/syh/colon_dataset/Kumar/kumar'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.tif'), recursive=True):
            if 'Overlay' in img:
                continue

            label_path = img.replace('Images', 'Labels').replace('.tif', '.mat')

            imgs.append(img)
            seg_maps.append(label_path)



        return imgs, seg_maps

    def get_kumar(self):
        path = '/data/smb/syh/colon_dataset/Kumar/kumar'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.tif'), recursive=True):
            if 'Overlay' in img:
                continue

            label_path = img.replace('Images', 'Labels').replace('.tif', '.mat')

            imgs.append(img)
            seg_maps.append(label_path)



        return imgs, seg_maps


    def get_cpm17(self):
        path = '/data/smb/syh/colon_dataset/CPM17/cpm17/'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            if 'Overlay' in img:
                continue

            label_path = img.replace('Images', 'Labels').replace('.png', '.mat')

            imgs.append(img)
            seg_maps.append(label_path)



        return imgs, seg_maps

    def get_consep(self):
        path = '/data/smb/syh/colon_dataset/CoNSeP/'

        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            if 'Overlay' in img:
                continue

            label_path = img.replace('Images', 'Labels').replace('.png', '.mat')

            #print(img)
            #print(label_path)
            imgs.append(img)
            seg_maps.append(label_path)



        #return img, label_path
        return imgs, seg_maps

    def get_crops(self):
        imgs = []
        seg_maps = []

        glas_train = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/'
        for img in glob.iglob(os.path.join(glas_train, '**', '*.jpg'), recursive=True):

           if 'Images' not in img:
               continue

           label_path = img.replace('Images', 'Annotation').replace('.jpg', '.png')

           imgs.append(img)
           seg_maps.append(label_path)


        #glas_valid = '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/'
        #for img in glob.iglob(os.path.join(glas_valid, '**', '*.jpg'), recursive=True):

        #    if 'Images' not in img:
        #        continue

        #    label_path = img.replace('Images', 'Annotation').replace('.jpg', '.png')

        #    imgs.append(img)
        #    seg_maps.append(label_path)

        crag_train = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/'
        for img in glob.iglob(os.path.join(crag_train, '**', '*.jpg'), recursive=True):

            if 'Images' not in img:
                continue

            label_path = img.replace('Images', 'Annotation').replace('.jpg', '.png')

            imgs.append(img)
            seg_maps.append(label_path)

        crag_valid = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/'
        for img in glob.iglob(os.path.join(crag_valid, '**', '*.jpg'), recursive=True):

            if 'Images' not in img:
                continue

            label_path = img.replace('Images', 'Annotation').replace('.jpg', '.png')

            imgs.append(img)
            seg_maps.append(label_path)


        return imgs, seg_maps


    def get_lizeard(self):
        path = '/data/smb/syh/gland_segmentation/Lizard/'
        label = os.path.join(path, 'Lizard_Labels', 'Labels')



        imgs = []
        seg_maps = []
        for img in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            if 'Lizard_Images' not in img:
                continue

            imgs.append(img)

            bs_name = os.path.basename(img)
            bs_name = bs_name.replace('.png', '.mat')

            label_map = os.path.join(label, bs_name)
            seg_maps.append(label_map)

            #seg_map = sio.loadmat(label_map)
            #seg_map = seg_map['inst_map']

            #seg_map[seg_map > 0] = 1
            #seg_maps.append(seg_map)

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

        if 'Lizard' in seg_map:

            label= sio.loadmat(seg_map)
            label = label['inst_map']

        elif 'CoNSeP' in seg_map:
            label = sio.loadmat(seg_map)
            label = label['inst_map']
        elif 'CPM17' in seg_map:
            label = sio.loadmat(seg_map)
            label = label['inst_map']
        elif 'Kumar' in seg_map:
            label = sio.loadmat(seg_map)
            label = label['inst_map']
        elif 'TNBC' in seg_map:
            label = sio.loadmat(seg_map)
            label = label['inst_map']
        elif 'MoNuSAC' in seg_map:
            label = np.load(seg_map)
        else:

            label = cv2.imread(seg_map, -1)
        if image is None:
            print(img)
        if label is None:
            print(seg_map)

        assert os.path.isfile(seg_map)
        #if label
        #print(seg_map)
        #print(np.unique(label))
        if label is None:
            print(seg_map, 'NOOOOOOOOOOOOOONE')
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