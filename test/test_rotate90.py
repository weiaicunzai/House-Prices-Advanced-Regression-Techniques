import os
import sys
sys.path.append(os.getcwd())
import argparse

import cv2
import numpy as np
import random

import transforms
import utils


# trans = transforms.Resize(
#     range=[0.5, 1.5],
#     # size=(300, 400)
# )

from transforms import MyRotate90
# from albumentations.augmentations.geometric.rotate import RandomRotate90
# class MyRotate90(RandomRotate90):

#     @property
#     def targets(self):
#         return {
#             "image": self.apply,
#             "mask": self.apply_to_mask,
#             "masks": self.apply_to_masks,
#             "bboxes": self.apply_to_bboxes,
#             "keypoints": self.apply_to_keypoints,
#             "weightmap": self.apply_to_weightmap,
#         }

#     # def get_params(self):
#         # self.alpha = random.randint(*self.alpha_range)
#         # self.sigma = random.uniform(*self.sigma_range)
#         # return {"random_state": random.randint(0, 10000)}

#     def apply(self, img, factor=0, **params):
#         """
#         Args:
#             factor (int): number of times the input will be rotated by 90 degrees.
#         """
#         return np.ascontiguousarray(np.rot90(img, factor))


#     def apply_to_weightmap(
#        self, img, factor=0, **params
#     ):
#         return self.apply(
#             img,
#             factor,
#             **params
#         )

#     def get_params(self):
#         # Random int in the range [0, 3]
#         return {"factor": 1}


#     def trans(self, *args, force_apply: bool = False, **kwargs):
#         if args:
#             raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
#         if self.replay_mode:
#             if self.applied_in_replay:
#                 return self.apply_with_params(self.params, **kwargs)

#             return kwargs

#         if (random.random() < self.p) or self.always_apply or force_apply:
#             params = self.get_params()

#             if self.targets_as_params:
#                 assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
#                     self.__class__.__name__, self.targets_as_params
#                 )
#                 targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
#                 params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
#                 params.update(params_dependent_on_targets)
#             if self.deterministic:
#                 if self.targets_as_params:
#                     warn(
#                         self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
#                         " because its' params depend on targets."
#                     )
#                 kwargs[self.save_key][id(self)] = deepcopy(params)
#             return self.apply_with_params(params, **kwargs)

#         return kwargs

#     def __call__(self, img, mask, weight_map=None):
#         """
#             img (np.ndarray): Image to be rotated.
#         Returns:
#             np.ndarray: Rotated image.
#         """

#         if weight_map is not None:
#             output = self.trans(image=img, mask=mask, weightmap=weight_map)
#             img = output.get('image')
#             mask = output.get('mask')
#             weight_map = output.get('weightmap')

#             return img, mask, weight_map

#         else:

#             output = self.trans(image=img, mask=mask)
#             img = output.get('image')
#             mask = output.get('mask')
#             return img, mask



def gen_trans():

    crop_size = (480, 480)
    trans = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(0, 90), expand=True, p=1),
            ]
        )

    return trans
def gen_trans1():
    from albumentations.augmentations.geometric.rotate import Rotate
    trans = Rotate(limit=(0, 90))
    return trans


def gen_mytrans():
    trans = MyRotate90(p=1)
    return trans

def test_mytrans(trans, img, anno, weight_map):

    print(np.unique(anno))
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite('tmp/orig.jpg', img)
    img, anno, weight_map = trans(img, anno, weight_map)
    # img = output['image']
    # anno = output['mask']
    cv2.imwrite('tmp/img.jpg', img)

    print(np.unique(anno))

    anno = anno / anno.max() * 255

    cv2.imwrite('tmp/mask.png', anno)
    cv2.imwrite('tmp/weight_map.png', weight_map)

def test_trans1(trans, img, anno, weight_map):

    print(np.unique(anno))
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite('tmp/orig.jpg', img)
    output = trans(image=img, mask=anno)
    img = output['image']
    anno = output['mask']
    cv2.imwrite('tmp/img.jpg', img)

    print(np.unique(anno))

    anno = anno / anno.max() * 255

    cv2.imwrite('tmp/mask.png', anno)



def gen_random_inputs():
    idx = random.randint(1, 85)
    img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}.bmp'.format(idx)
    anno_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/train_{}_anno.bmp'.format(idx)
    weight_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/weight_maps/train/train_{}_anno_weight.png'.format(idx)
    #path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/after_final.jpg'
    # path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/aa.png'
    print(img_path)
    print(anno_path)
    img = cv2.imread(img_path, -1)
    weight_map = cv2.imread(weight_path, -1)
    anno = cv2.imread(anno_path, -1)

    return img, anno, weight_map


# img, anno, weight_map = gen_random_inputs()

# test_trans1(gen_trans1(), *gen_random_inputs())
test_mytrans(gen_mytrans(), *gen_random_inputs())
