import sys
import os
import random
import shutil
import argparse
sys.path.append(os.getcwd())

import numpy as np
import cv2
from albumentations.augmentations.geometric.transforms import ElasticTransform

import transforms
import utils

def test_data_trans(img, anno, weight_map):

    crop_size = (480, 480)
    trans = transforms.Compose(
        [
            transforms.RandomChoice
                (
                    [
                        # nothing:
                        transforms.Compose([]),

                        # h:
                        transforms.RandomHorizontalFlip(p=1),

                        # v:
                        transforms.RandomVerticalFlip(p=1),

                        # hv:
                        transforms.Compose([
                               transforms.RandomVerticalFlip(p=1),
                               transforms.RandomHorizontalFlip(p=1),
                        ]),

                         #r90:
                        transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),

                        # #r90h:
                        transforms.Compose([
                            transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            transforms.RandomHorizontalFlip(p=1),
                        ]),

                        # #r90v:
                        transforms.Compose([
                            transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            transforms.RandomVerticalFlip(p=1),
                        ]),

                        # #r90hv:
                        transforms.Compose([
                            transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.RandomVerticalFlip(p=1),
                        ]),
                    ]
                ),

            # transforms.ElasticTransformWrapper(),
            transforms.MyElasticTransform(),
            transforms.Resize(range=[0.5, 1.5]),
            # transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=0.5),
            transforms.RandomRotation(degrees=(0, 90), expand=True),
            # transforms.RandomApply([
            #transforms=transforms.PhotoMetricDistortion(),
            # ]),
            transforms.PhotoMetricDistortion(),
            transforms.RandomCrop(crop_size=crop_size, cat_max_ratio=0.85, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD)

        ]

    )


    #img = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/aa.png', -1)
    # idx = random.randint(1, 85)
    # idx = 14
    # img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}.bmp'.format(idx)
    # anno_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}_anno.bmp'.format(idx)
    # weight_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/weight_maps/train/train_{}_anno_weight.png'.format(idx)
    # #path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/after_final.jpg'
    # # path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/aa.png'
    # print(img_path)
    # print(anno_path)
    # img = cv2.imread(img_path, -1)
    # weight_map = cv2.imread(weight_path, -1)
    #cv2.imwrite('tmp/ori.jpg', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

    # cv2.imwrite('tmp/ori.jpg', img)
    # shutil.copy(img_path, 'tmp/ori.jpg')
    # anno = cv2.imread(anno_path, -1)
    # print(np.unique(anno))
    #img, anno, weight_map = trans(img, anno, weight_map)
    trans = trans.transforms
    #print(trans)
    store_rotate = 0
    store_elastic = 0
    store_resize = 0
    store_random = 0
    for t in trans:
        try:
            img, anno, weight_map = t(img, anno, weight_map)
            if isinstance(t, transforms.RandomRotation):
                store_rotate = img
            if isinstance(t, transforms.ElasticTransform):
                store_elastic = img
            if isinstance(t, transforms.Resize):
                store_elastic = img
            if isinstance(t, transforms.RandomChoice):
                store_random = img

        except:
            cv2.imwrite('tmp/rotate.jpg', store_rotate)
            cv2.imwrite('tmp/elastic.jpg', store_elastic)
            cv2.imwrite('tmp/resize.jpg', store_resize)
            cv2.imwrite('tmp/random.jpg', store_random)






    # alb_trans = ElasticTransform(alpha=20, sigma=3, alpha_affine=30, p=1)

    #elb_img = alb_trans(image=img)['image']


    # print(np.unique(anno))
    # anno[anno == 255] = 0

    #cv2.imwrite('tmp/img.jpg', img)
    #cv2.imwrite('tmp/anno.png', anno / anno.max() * 255)
    #cv2.imwrite('tmp/weight_map.jpg', weight_map)
    return img, anno, weight_map

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


count = 0
# for i in range(15):
    # for imgs, masks, weight_maps in train_loader:
    #     # print(image.shape)
    #     count += 1
    #     #print(count)
    #     if count % 10 == 0:
    #         print(count)

#for img, mask, weight_map in train_loader.dataset:
for img, mask, weight_map in train_loader.dataset:
    count += 1
    print(img.shape)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('tmp1/{}.jpg'.format(count), img)
    if count > 50:
        import sys; sys.exit()



# idx = random.randint(1, 85)
# idx = 14
# img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}.bmp'.format(idx)
# anno_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}_anno.bmp'.format(idx)
# weight_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/weight_maps/train/train_{}_anno_weight.png'.format(idx)

# print(img_path)
# print(anno_path)
# img = cv2.imread(img_path, -1)
# weight_map = cv2.imread(weight_path, -1)
# anno = cv2.imread(anno_path, -1)


# inputs = []
# imgs = []
# annos = []
# weight_maps = []
# for i in range(1000):
#     imgs.append(img)
#     annos.append(anno)
#     weight_maps.append(weight_map)
# from multiprocessing import Pool
# #with Pool(6) as p:
#     #p.map(test_data_trans, zip(imgs, annos, weight_maps))

# for inputs in zip(imgs, annos, weight_maps):
#     test_data_trans(inputs)
#    if i % 1000 == 0:
#        print('i')