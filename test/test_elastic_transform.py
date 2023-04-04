import sys
import os
import random
import shutil
sys.path.append(os.getcwd())

import numpy as np
import cv2
import transforms

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from albumentations.augmentations.geometric.transforms import ElasticTransform


# class MyElasticTransform(ElasticTransform):

class MyElasticTransform(ElasticTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
        same_dxdy (boolean): Whether to use same random generated shift for x and y.
                             Enabling this option gives ~2X speedup.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha=1,
        alpha_range=(80, 120),
        sigma=50,
        sigma_range=(9, 11),
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        approximate=False,
        same_dxdy=False,
        p=0.5,
    ):
        super(ElasticTransform, self).__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range

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

    def get_params(self):
        self.alpha = random.randint(*self.alpha_range)
        self.sigma = random.uniform(*self.sigma_range)
        return {"random_state": random.randint(0, 10000)}

    def apply_to_weightmap(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return self.apply(
            img,
            random_state,
            interpolation,
            **params,
        )

    def get_transform_init_args_names(self):
        return (
            "alpha",
            "alpha_range",
            "sigma",
            "sigma_range",
            "alpha_affine",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "approximate",
            "same_dxdy",
        )


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


# class ElasticTransformWrapper:
#     def __init__(
#         self,
#         alpha=1,
#         alpha_range=(80, 120),
#         sigma=50,
#         sigma_range=(9, 11),
#         alpha_affine=50,
#         interpolation=cv2.INTER_LINEAR,
#         border_mode=cv2.BORDER_REFLECT_101,
#         value=None,
#         mask_value=None,
#         always_apply=False,
#         approximate=False,
#         same_dxdy=False,
#         p=0.5,
#     ):
#         self.trans = MyElasticTransform(
#             alpha=alpha,
#             alpha_range=alpha_range,
#             sigma=sigma,
#             sigma_range=sigma_range,
#             alpha_affine=alpha_affine,
#             interpolation=interpolation,
#             border_mode=border_mode,
#             value=value,
#             mask_value=mask_value,
#             always_apply=always_apply,
#             approximate=approximate,
#             same_dxdy=same_dxdy,
#             p=p,
#         )

#     def __call__(self, img, mask, weight_map=None):
#         """
#             img (np.ndarray): Image to be rotated.
#         Returns:
#             np.ndarray: Rotated image.
#         """

#         # if random.random() > self.p:
#         #     if weight_map is not None:
#         #         return img, mask, weight_map
#         #     else:
#         #         return img, mask

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
#             # weight_map = output.get('weightmap')
#             return img, mask

def test_data_trans():

    crop_size = (480, 480)
    trans = transforms.Compose(
        [
            transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=90, p=0.5),
            # transforms.RandomRotation(degrees=(0, 90), expand=True),
            # transforms.Resize(range=[0.5, 1.5]),
            # transforms.RandomApply(
            #     transforms=[transforms.PhotoMetricDistortion()]
            # ),
            # transforms.RandomCrop(crop_size=crop_size, cat_max_ratio=0.99, pad_if_needed=True),
            #transforms.ToTensor(),
            #transforms.Normalize(settings.MEAN, settings.STD)
        ]
    )

    #img = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/aa.png', -1)
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
    cv2.imwrite('tmp/ori.jpg', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

    #cv2.imwrite('tmp/ori.jpg', img)
    # shutil.copy(img_path, 'tmp/ori.jpg')
    anno = cv2.imread(anno_path, -1)
    print(np.unique(anno))

    #alb_trans = MyElasticTransform(alpha=20, sigma=3, alpha_affine=30, p=1)
    alb_trans = MyElasticTransform(alpha=100, sigma=10, alpha_affine=50, p=1)
    # alb_trans = ElasticTransformWrapper()

    img = cv2.imread(img_path, -1)
    # output = alb_trans(image=img, mask=anno, weightmap=weight_map)



    # img = output.get('image')
    # anno = output.get('mask')
    # weight_map = output.get('weightmap')
    img, anno, weight_map = alb_trans(img, anno, weight_map)


    print(np.unique(anno))
    anno[anno == 255] = 0

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('tmp/img.jpg', img)
    print(anno.max())
    cv2.imwrite('tmp/anno.png', anno / anno.max() * 255)
    cv2.imwrite('tmp/weight_map.jpg', weight_map)

    #cv2.imwrite('tmp/ic.jpg', img)
    #cv2.imwrite('tmp/alb_elastic.jpg', elb_img)


test_data_trans()

    # if i % 1000 == 0:
        # print('i')