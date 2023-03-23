import os
import sys
import random
sys.path.append(os.getcwd())

import cv2

from dataset import CRAG


import transforms

def test_crag():

#
    #crop_size=(768, 768)
    crop_size=(1024, 1024)
    trans = transforms.Compose([
                transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=0.5),
                transforms.Resize(range=[0.5, 1.5]),
                # transforms.RandomRotation(degrees=90, expand=False),
                #transforms.RandomRotation(degrees=90, expand=True),
                transforms.RandomCrop(crop_size=crop_size, cat_max_ratio=0.9, pad_if_needed=True),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    transforms=[transforms.PhotoMetricDistortion()]
                ),
                #transforms.ToTensor(),
                #transforms.Normalize(settings.MEAN, settings.STD)
            ])

    dataset = CRAG('/data/smb/syh/gland_segmentation/CRAGV2/CRAG/', 'train', transforms=trans)

    image, label, weight_map = random.choice(dataset)
    label[label > 0] = 255

    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    label = cv2.resize(label, (0, 0), fx=0.3, fy=0.3)
    weight_map = cv2.resize(weight_map, (0, 0), fx=0.3, fy=0.3)

    cv2.imwrite('img.jpg', image)
    cv2.imwrite('label.png', label)
    cv2.imwrite('weight_map.png', weight_map)



test_crag()


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