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

from albumentations.augmentations.geometric.rotate import Rotate
class MyRotate(Rotate):

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "weightmap": self.apply_to_weightmap,
        }

    # def get_params(self):
        # self.alpha = random.randint(*self.alpha_range)
        # self.sigma = random.uniform(*self.sigma_range)
        # return {"random_state": random.randint(0, 10000)}

    def apply_to_weightmap(
        self, img, angle=0, interpolation=cv2.INTER_LINEAR, x_min=None, x_max=None, y_min=None, y_max=None, **params
    ):
        return self.apply(
            img,
            angle,
            interpolation,
            x_min,
            x_max,
            y_min,
            y_max,
            **params
        )
        #img_out = F.rotate(img, angle, interpolation, self.border_mode, self.value)
        #if self.crop_border:
        #    img_out = FCrops.crop(img_out, x_min, y_min, x_max, y_max)
        #return img_out

    # def apply_to_weightmap(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
    #     return self.apply(
    #         img,
    #         random_state,
    #         interpolation,
    #         **params,
    #     )

    # def get_transform_init_args_names(self):
    #     return (
    #         "alpha",
    #         "alpha_range",
    #         "sigma",
    #         "sigma_range",
    #         "alpha_affine",
    #         "interpolation",
    #         "border_mode",
    #         "value",
    #         "mask_value",
    #         "approximate",
    #         "same_dxdy",
    #     )


    def trans(self, *args, force_apply: bool = False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def __call__(self, img, mask, weight_map=None):
        """
            img (np.ndarray): Image to be rotated.
        Returns:
            np.ndarray: Rotated image.
        """

        # if random.random() > self.p:
        #     if weight_map is not None:
        #         return img, mask, weight_map
        #     else:
        #         return img, mask

        if weight_map is not None:
            output = self.trans(image=img, mask=mask, weightmap=weight_map)
            img = output.get('image')
            mask = output.get('mask')
            weight_map = output.get('weightmap')

            return img, mask, weight_map

        else:

            output = self.trans(image=img, mask=mask)
            img = output.get('image')
            mask = output.get('mask')
            # weight_map = output.get('weightmap')
            return img, mask



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
    #trans = MyRotate(limit=(0, 90), crop_border=False, p=1)
    trans = MyRotate(limit=90, crop_border=False, p=1)
    return trans

def test_mytrans(trans, img, anno, weight_map):

    print(np.unique(anno), trans)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite('tmp/orig.jpg', img)
    print(img.shape)
    img, anno, weight_map = trans(img, anno, weight_map)
    print(img.shape)
    # img = output['image']
    # anno = output['mask']
    cv2.imwrite('tmp/img.jpg', img)

    print(np.unique(anno))

    anno = anno / anno.max() * 255

    cv2.imwrite('tmp/mask.png', anno)
    cv2.imwrite('tmp/weight_map.png', weight_map)


def test_trans1(trans, img, anno, weight_map):

    print(np.unique(anno))
    # img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite('tmp/orig.jpg', img)
    img, anno, weight_map = trans(img, anno, weight_map)
    # img = output['image']
    # anno = output['mask']
    cv2.imwrite('tmp/img.jpg', img)

    print(np.unique(anno), trans)

    anno = anno / anno.max() * 255

    cv2.imwrite('tmp/mask.png', anno)

    cv2.imwrite('tmp/weight_map.png', weight_map)


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

test_trans1(gen_trans(), *gen_random_inputs())
#test_mytrans(gen_mytrans(), *gen_random_inputs())





# class MyRotate(Rotate):
    # def


#def __init__(self, size=None, range=None, keep_ratio=True, min_size=None):

# segmap = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_22_anno.bmp'
# img = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_22.bmp'

# img = cv2.imread(img, -1)
# segmap = cv2.imread(segmap, -1)
# h, w = img.shape[:2]
# h, w = (300, 400)
# print(np.unique(segmap), img.shape)

# img, segmap = trans(img, segmap)

# print(np.unique(segmap))
# print(img.shape, segmap.shape, img.shape[0] / h,  img.shape[1] / w)

# cv2.imwrite('tmp/img.jpg', img)
# cv2.imwrite('tmp/segmap.png', segmap / segmap.max() * 255)


# parser = argparse.ArgumentParser()
# parser.add_argument('-b', type=int, default=64 * 4,
#                     help='batch size for dataloader')
# parser.add_argument('-net', type=str, help='if resume training')
# parser.add_argument('-dataset', type=str, default='Glas', help='dataset name')
# parser.add_argument('-download', action='store_true', default=False,
#     help='whether to download camvid dataset')
# parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
# args = parser.parse_args()
# print(args)

# # args.


# train_loader = utils.data_loader(args, 'train')
# # train_loader.dataset.transforms = test_data_trans
# dataset = train_loader.dataset
# dataset.transforms = trans


# count = 0
# for img, anno, weight_map in dataset:
#     #print(img.shape, anno.shape, weight_map.shape)
#     count += 1
#     img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
#     anno = cv2.resize(anno, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)

#     anno_mask = anno == 255
#     anno[anno_mask] = 0
#     anno = anno / anno.max() * 200
#     anno[anno_mask] = 255
#     cv2.imwrite('tmp1/{}.jpg'.format(count), img)
#     cv2.imwrite('tmp1/{}.png'.format(count), anno)

#     if count > 100:
#         break
