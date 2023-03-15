import random
from PIL import Image
import math
import numbers
from collections.abc import Iterable
import warnings
import types
import collections
from typing import List, Optional, Tuple, Union, no_type_check
from itertools import repeat


import cv2
# import mmcv
import numpy as np

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance

_cv2_pad_to_str = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}
INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


class Compose:
    """Composes several transforms together.
    Args:
        transforms(list of 'Transform' object): list of transforms to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, weight_map=None):
        #count = 0
        for trans in self.transforms:
                if weight_map is not None:
                    img, mask, weight_map = trans(img, mask, weight_map)
                else:
                    img, mask = trans(img, mask)


        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize:
    """Resize an image and an mask to given size
    Args:
        size: expected output size of each edge, can be int or iterable with (h, w)
        if range is given:
            resize image to size  [h * s, w * s],  s is sampled from range .
    """

    def __init__(self, size=None, range=None, keep_ratio=True, min_size=None):

        if size is not None:
            if isinstance(size, int):
                self.size = (size, size)
            elif isinstance(size, Iterable) and len(size) == 2:
                self.size = size
            else:
                raise TypeError('size should be iterable with size 2 or int')

        elif range is not None:
            if isinstance(range, Iterable) and len(range) == 2:
                self.range = range
            else:
                raise TypeError('size should be iterable with size 2 or int')

        # elif range:
            # raise ValueError(' size and range should be set least one')

        # print(range, size)
        # assert  range is not None and size is None
        #if not (range is None) ^ (size is None):
            #raise ValueError('can not both be set or not set')

        self.size = size
        self.range = range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    # def random

    def __call__(self, img, mask, weight_map=None):

        # size = self.size



        if self.range:
            ratio = random.uniform(*self.range)
            resized_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
            resized_mask = cv2.resize(mask, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            if weight_map is not None:
                weight_map = cv2.resize(weight_map, (0, 0), fx=ratio, fy=ratio)

        if self.size:
            h, w = self.size
            size = (w, h)
            resized_img = cv2.resize(img, size)
            resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            if weight_map is not None:
                weight_map = cv2.resize(weight_map, size)

        if self.keep_ratio:
            #print(self.min_size)
            if self.min_size is not None:
                h, w = img.shape[:2]
                #print(h, w)
                if h < w:
                    new_h = self.min_size
                    new_w = w / h * new_h
                else:
                    new_w = self.min_size
                    new_h =  h / w * new_w

                #print(new_w, new_h)
                resized_img = cv2.resize(img, (int(new_w), int(new_h)))
                resized_mask = cv2.resize(mask, (int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)
                if weight_map is not None:
                    weight_map = cv2.resize(weight_map, (int(new_w), int(new_h)))
        # print(np.unique(resized_mask))
        if weight_map is not None:
            return resized_img, resized_mask, weight_map
        else:
            return resized_img, resized_mask

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    # print(type(img))
    # mg[i:i + h, j: j+ w, ...]
    return img[i:i + h, j: j+ w, ...]

    # print(type(tmp), 'cccc')

    # return tmp
    # print(img.shape, type(img))
    # import sys; sys.exit()
    #if len(img.shape) == 3:
        #return img[i:i + h, j:j + w, :]
        #return img[i:i + h, j:j + w, :]
    #if len(img.shape) == 2:
        #return img[i:i + h, j:j + w, ...]

    # print(img.shape, type(img), '------------------------------')

    # return img
    # return
    #print()

def center_crop(img, output_size, fill=0):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    h, w = img.shape[0:2]
    th, tw = output_size
    pad_left = max(int((tw - w) / 2), 0)
    # tw - w - pad_left  >= pad_left if tw - w > 0
    pad_right = max(tw - w - pad_left, pad_left)
    pad_top = max(int((th - h) / 2), 0)
    pad_bot = max(th - h - pad_top, pad_top)
    img = pad(img, (pad_left, pad_top, pad_right, pad_bot), fill=fill)
    h, w = img.shape[0:2]

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value io only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(
            type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding,
                  collections.abc.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.abc.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.abc.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    #if len(img.shape) == 2:
    #    return cv2.copyMakeBorder(img,
    #                               top=pad_top,
    #                               bottom=pad_bottom,
    #                               left=pad_left,
    #                               right=pad_right,
    #                               borderType=_cv2_pad_to_str[padding_mode],
    #                               value=fill)[:, :, np.newaxis]
    #else:
    return cv2.copyMakeBorder(img,
                              top=pad_top,
                              bottom=pad_bottom,
                              left=pad_left,
                              right=pad_right,
                              borderType=_cv2_pad_to_str[padding_mode],
                              value=fill)

class RandomCrop(object):
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self,
                #  size,
                crop_size,
                #  padding=None,
                pad_if_needed=False,
                pad_value=0,
                seg_pad_value=255,
                cat_max_ratio=1.):
                #  padding_mode='constant'):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        # self.padding = padding
        self.pad_if_needed = pad_if_needed
        # self.fill = fill
        # self.padding_mode = padding_mode
        self.pad_value= pad_value
        self.seg_pad_value = seg_pad_value
        #self.ignore_value = ignore_value,
        self.cat_max_ratio = cat_max_ratio

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        # print(img.shape[:2], output_size)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, max(h - th, 0))
        j = random.randint(0, max(w - tw, 0))
        return i, j, th, tw

    def __call__(self, img, mask, weight_map=None):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        #if self.padding is not None:
        #    img = pad(img, self.padding, self.pad_value, self.padding_mode)
        #    mask = pad(mask, self.padding, self.seg_pad_value, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.crop_size[1]:
        # if self.pad_if_needed and img.shape[1] < self.crop_size[1]:
            left_pad = int((self.crop_size[1] - img.shape[1]) / 2)
            right_pad = self.crop_size[1] - img.shape[1] - left_pad
            #img = pad(img, (self.size[1] - img.shape[1], 0), self.fill,
            #            self.padding_mode)
            img = pad(img, (left_pad, 0, right_pad, 0), self.pad_value,
                        # self.padding_mode)
                        'constant')

            if weight_map is not None:
                weight_map = pad(weight_map, (left_pad, 0, right_pad, 0), self.pad_value,
                        # self.padding_mode)
                        'constant')
            #mask = pad(mask, (self.size[1] - mask.shape[1], 0), self.fill,
            #            self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.seg_pad_value,
                        # self.padding_mode)
                        'constant')
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.crop_size[0]:
            top_pad = int((self.crop_size[0] - img.shape[0]) / 2)
            bot_pad = self.crop_size[0] - img.shape[0] - top_pad
            #img = pad(img, (0, self.size[0] - img.shape[0]), self.fill,
            #            self.padding_mode)
            #mask = pad(mask, (0, self.size[0] - mask.shape[0]), self.fill,
            #            self.padding_mode)
            img = pad(img, (0, top_pad, 0, bot_pad), self.pad_value,
                        'constant')
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.seg_pad_value,
                        # self.padding_mode)
                        'constant')

            if weight_map is not None:
                weight_map = pad(weight_map, (0, top_pad, 0, bot_pad), self.pad_value,
                        'constant')

        #print(self.pad_if_needed)
        i, j, h, w = self.get_params(img, self.crop_size)





        # print(i,j,h,w)

        # for self.
        # mask = crop(mask, i, j, h, w)

        # consume time
        #if self.cat_max_ratio < 1:
        #    for _ in range(10):
        #        # print(_)
        #        mask_temp = crop(mask, i, j, h, w)
        #        labels, cnt = np.unique(mask_temp, return_counts=True)
        #        cnt = cnt[labels != self.seg_pad_value]

        #        # thresh = np.sum(cnt) / (mask_temp.shape[0] * mask_temp.shape[1])
        #        # print(thresh)
        #        # if thresh < 0.75:
        #            # continue

        #        if len(cnt) > 1 and np.max(cnt) / np.sum(
        #            cnt) < self.cat_max_ratio:
        #            break

        #        i, j, h, w = self.get_params(img, self.crop_size)


        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for iidx in range(1000):
                bbox = crop(mask, i, j, h, w)
                labels, cnt = np.unique(bbox, return_counts=True)
                cnt = cnt[labels != self.seg_pad_value]
                #print(iidx, cnt, np.max(cnt) / np.sum(cnt), i, j)
                if len(cnt) > 1 and 0.1 < np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                i, j, h, w = self.get_params(img, self.crop_size)

                if len(cnt) == 1 and iidx == 999:
                    cv2.imwrite('img.jpg', img)
                    cv2.imwrite('mask.png', mask)
                    raise ValueError('still no pixels of class???')


        img = crop(img, i, j, h, w)

        if weight_map is not None:
            weight_map = crop(weight_map, i, j, h, w)

        if weight_map is not None:
            return img, bbox, weight_map
        else:
            return img, bbox




        #return crop(img, i, j, h, w), crop(mask, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(
            self.crop_size)

#class RandomScale:
#    """Randomly scaling an image (from 0.5 to 2.0]), the output image and mask
#    shape will be the same as the input image and mask shape. If the
#    scaled image is larger than the input image, randomly crop the scaled
#    image.If the scaled image is smaller than the input image, pad the scaled
#    image.
#
#    Args:
#        size: expected output size of each edge
#        scale: range of size of the origin size cropped
#        value: value to fill the mask when resizing,
#               should use ignore class index
#    """
#
#    def __init__(self, scale=(0.5, 2.0), value=0):
#
#        if not isinstance(scale, Iterable) and len(scale) == 2:
#            raise TypeError('scale should be iterable with size 2 or int')
#
#        self.value = value
#        self.scale = scale
#
#    def __call__(self, img, mask):
#        oh, ow = img.shape[:2]
#
#        # scale image
#        scale = random.uniform(*self.scale)
#        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale,
#                          interpolation=cv2.INTER_NEAREST)
#
#        h, w = img.shape[:2]
#
#        # pad image and mask
#        diff_h = max(0, oh - h)
#        diff_w = max(0, ow - w)
#
#        img = cv2.copyMakeBorder(
#            img,
#            diff_h // 2,
#            diff_h - diff_h // 2,
#            diff_w // 2,
#            diff_w - diff_w // 2,
#            cv2.BORDER_CONSTANT,
#            value=[0, 0, 0]
#        )
#        mask = cv2.copyMakeBorder(
#            mask,

#            diff_h // 2,
#            diff_h - diff_h // 2,
#            diff_w // 2,
#            diff_w - diff_w // 2,
#            cv2.BORDER_CONSTANT,
#            value=self.value
#        )
#
#        h, w = img.shape[:2]
#
#        # crop image and mask
#        y1 = random.randint(0, h - oh)
#        x1 = random.randint(0, w - ow)
#        img = img[y1: y1 + oh, x1: x1 + ow]
#        mask = mask[y1: y1 + oh, x1: x1 + ow]
#
#        return img, mask


def rotate(img, angle, resample='BILINEAR', expand=False, center=None, value=0):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees clockwise order.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    imgtype = img.dtype
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    #h, w, _ = img.shape
    h, w = img.shape[:2]
    point = center or (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH), flags=INTER_MODE[resample], borderValue=value)
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample], borderValue=value)
    else:
        # print('cccccc')
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample], borderValue=value)
    return dst.astype(imgtype)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        resample ({CV.Image.NEAREST, CV.Image.BILINEAR, CV.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, p=0.5, resample='BILINEAR', expand=False, center=None, pad_value=0, seg_pad_value=255):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value
        self.p = p

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask, weight_map=None):
        """
            img (np.ndarray): Image to be rotated.
        Returns:
            np.ndarray: Rotated image.
        """

        if random.random() > self.p:
            if weight_map is not None:
                return img, mask, weight_map
            else:
                return img, mask



        angle = self.get_params(self.degrees)

        img = rotate(img, angle, self.resample, self.expand, self.center, value=self.pad_value)
        mask = rotate(mask, angle, 'NEAREST', self.expand, self.center, value=self.seg_pad_value)

        if weight_map is not None:
            weight_map = rotate(weight_map, angle, self.resample, self.expand, self.center, value=self.pad_value)

        if weight_map is not None:
            return img, mask, weight_map

        else:
            return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.transforms = transforms
        self.p = p

    def forward(self, img, mask, weight_map=None):
        if random.random() <= self.p:
        # if self.p < torch.rand(1):
            if weight_map is not None:
                return img, mask, weight_map
            else:
                return img, mask

        for t in self.transforms:
            if weight_map is not None:
                img, mask, weight_map = t(img, mask, weight_map)
            else:
                img, mask = t(img, mask)


        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask


    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


#class RandomRotation:
#    """Rotate the image by angle
#
#    Args:
#        angle: rotated angle
#        value: value used for filling the empty pixel after rotating,
#               should use ignore class index
#
#    """
#
#    def __init__(self, p=0.5, angle=10, fill=0):
#
#        if not (isinstance(angle, numbers.Number) and angle > 0):
#            raise ValueError('angle must be a positive number.')
#
#        self.angle = angle
#        self.value = fill
#        self.p = p
#
#    def __call__(self, image, mask):
#        if random.random() > self.p:
#            return image, mask
#
#        angle = random.uniform(-self.angle, self.angle)
#        image_center = tuple(np.array(image.shape[1::-1]) / 2)
#        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#        image = cv2.warpAffine(
#            image, rot_mat, image.shape[1::-1])
#        mask = cv2.warpAffine(
#            mask, rot_mat, mask.shape[1::-1],
#            flags=cv2.INTER_NEAREST,
#            borderMode=cv2.BORDER_CONSTANT,
#            borderValue=self.value
#        )
#
#        return image, mask

class RandomVerticalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, weight_map=None):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() <= self.p:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
            if weight_map is not None:
                weight_map = cv2.flip(weight_map, 0)

        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask

class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, weight_map=None):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() <= self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            if weight_map is not None:
                weight_map = cv2.flip(weight_map, 1)

        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask

class RandomGaussianBlur:
    """Blur an image using gaussian blurring.

    Args:
       sigma: Standard deviation of the gaussian kernel.
       Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are
       common. Kernel size will automatically be derived from sigma
       p: probability of applying gaussian blur to image

       https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/blur.html#GaussianBlur
    """

    def __init__(self, p=0.5, sigma=(0.0, 3.0)):

        if not isinstance(sigma, Iterable) and len(sigma) == 2:
            raise TypeError('sigma should be iterable with length 2')

        if not sigma[1] >= sigma[0] >= 0:
            raise ValueError(
                'sigma shoule be an iterval of nonegative real number')

        self.sigma = sigma
        self.p = p

    def __call__(self, img, mask):

        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            k_size = self._compute_gaussian_blur_ksize(sigma)
            img = cv2.GaussianBlur(img, (k_size, k_size),
                                   sigmaX=sigma, sigmaY=sigma)

        return img, mask

    @staticmethod
    def _compute_gaussian_blur_ksize(sigma):
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        ksize = int(max(ksize, 3))

        # kernel size needs to be an odd number
        if not ksize % 2:
            ksize += 1

        return ksize

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([i * brightness_factor
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([(i - 74) * contrast_factor + 74
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, mask):
        return self.lambd(img), mask

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn(
                'Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn(
                'Hue jitter enabled. Will slow down loading immensely.')
        self.p = p

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(
                    name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(
                    lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(
                    lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        if random.random() < self.p:
            return img, mask

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch
    float tensor (c, h, w) ranged from 0 to 1, and convert mask to torch tensor
    """

    def __call__(self, img, mask, weight_map=None):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]

        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        mask = torch.from_numpy(mask).long()
        if weight_map is not None:
            weight_map = torch.from_numpy(weight_map).long()

        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask


class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    and does nothing to mask tensor

    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, mask, weight_map=None):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if img.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(img.size()))

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)

        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(std.dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        img.sub_(mean).div_(std)

        # img.sub_(mean[:, None, None]).div_(std[:, None, None])

        if weight_map is not None:
            return img, mask, weight_map
        else:
            return img, mask


#class Normalize:
#    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
#    and does nothing to mask tensor
#
#    for each channel in torch tensor:
#        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#    Args:
#        mean: sequence of means for each channel
#        std: sequence of stds for each channel
#    """
#
#    def __init__(self, mean, std, inplace=False):
#        self.mean = mean
#        self.std = std
#        self.inplace = inplace
#
#    def __call__(self, img, mask):
#        """
#        Args:
#            (H W C) format numpy array range from [0, 255]
#        Returns:
#            (H W C) format numpy array in float32 range from [0, 1]
#        """
#        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'
#
#        if not self.inplace:
#            img = img.clone()
#
#        mean = torch.tensor(self.mean, dtype=torch.float32)
#        std = torch.tensor(self.std, dtype=torch.float32)
#        img.sub_(mean[:, None, None]).div_(std[:, None, None])
#
#        return img, mask


class EncodingLable:
    def __init__(self):
        pass

    def __call__(self, image, label):
        label[label != 0] = 1
        return image, label

class RandomScaleCrop:
    """Randomly scaling an image (from 0.5 to 2.0]), the output image and mask
    shape will be the same as the input image and mask shape. If the
    scaled image is larger than the input image, randomly crop the scaled
    image.If the scaled image is smaller than the input image, pad the scaled
    image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        value: value to fill the mask when resizing,
               should use ignore class index
    """

    def __init__(self, crop_size, scale=(0.5, 2.0), value=0, padding_mode='constant'):

        if not isinstance(scale, Iterable) and len(scale) == 2:
            raise TypeError('scale should be iterable with size 2 or int')

        self.fill = value
        self.scale = scale
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask):

        scale = random.uniform(self.scale[0], self.scale[1])

        crop_size = int(self.crop_size / scale)

        if img.shape[1] < crop_size:
            left_pad = int((crop_size - img.shape[1]) / 2)
            right_pad = crop_size - img.shape[1] - left_pad
            img = pad(img, (left_pad, 0, right_pad, 0), 0,
                        self.padding_mode)
            mask = pad(mask, (left_pad, 0, right_pad, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if img.shape[0] < crop_size:
            top_pad = int((crop_size - img.shape[0]) / 2)
            bot_pad = crop_size - img.shape[0] - top_pad
            img = pad(img, (0, top_pad, 0, bot_pad), 0,
                        self.padding_mode)
            mask = pad(mask, (0, top_pad, 0, bot_pad), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(img, (crop_size, crop_size))
        img = crop(img, i, j, h, w)
        mask = crop(mask, i, j, h, w)

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)


        return img, mask


class MultiScaleFlipAug(object):
    """Return a set of MultiScale fliped Images"""


    def __init__(self,
                #  transforms,
                #  img_scale,
                # img_ratios=None,
                 img_ratios,
                 mean,
                 std,
                 transforms=None,
                 flip=False,
                 min_size=None,
                 flip_direction='horizontal',
                 resize_to_multiple=True):

        img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]

        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]


        # normalize and to_tensor
        self.transforms = transforms

        #param_dict = {
        #    'scale': 208,
        #    'to_tensor': 1,
        #    'normalize': np.array([[0.78780321, 0.5120167 , 0.78493782],
        #[0.16766301, 0.24838048, 0.13225162]])}
        #self.my_trans = my_transforms.get_transforms(param_dict)

        for flip_direction in self.flip_direction:
            assert flip_direction in ['v', 'h', 'hv', 'r90']

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        # print(self.flip_direction, 'cccccccccccccccccccccccccc')
        if resize_to_multiple:
            self.resize_to_multiple = ResizeToMultiple(
                interpolation=cv2.INTER_LINEAR,
                size_divisor=32,
            )

        self.if_resize_to_multiple = resize_to_multiple
        self.min_size = min_size

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(flip={self.flip}, '
                     f'img_ratios={self.img_ratios}), '
                     f'resize_to_multiple={self.if_resize_to_multiple}), '
                     f'min_size={self.min_size}), '
                     f'flip_direction={self.flip_direction})')
        return repr_str

    def construct_flip_param(self):

        #flip_aug = [False, True] if self.flip else [False]
        flip_aug = [False]
        flip_direction = []
        flip_direction.append(self.flip_direction[0])
        #if len(self.flip_direction) == 2:
            #flip_aug.append(True)


        #flip_direction = self.flip_direction.copy()
        if self.flip:
            #flip_direction.append(flip_direction[0])
            for flip_direct in self.flip_direction:
                flip_aug.append(True)
                flip_direction.append(flip_direct)


        # print(flip_aug, flip_direction)
        assert len(flip_aug) == len(flip_direction)
        #print(flip_aug, flip_direction)

        return list(zip(flip_aug, flip_direction))


    def norm(self, img):

        std = self.std
        mean = self.mean

        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(std.dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return img.sub_(mean).div_(std)

    def mask_to_tensor(self, mask):
        mask = torch.from_numpy(mask).long()
        return mask


    def img_to_tensor(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img


    def __call__(self, img, gt_seg):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        img_meta = {
            "seg_map": None,
            "imgs" : [],
            "flip" : []
        }

        flip_param = self.construct_flip_param()

        if self.min_size is not None:
            h, w = img.shape[:2]
            #print(h, w)
            if h < w:
                new_h = self.min_size
                new_w = w / h * new_h
            else:
                new_w = self.min_size
                new_h =  h / w * new_w

            #print(new_w, new_h)
            img = cv2.resize(img, (int(new_w), int(new_h)))
        # print(np.unique(resized_mask))
        #return resized_img, resized_mask



        for ratio in self.img_ratios:

            resized_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

            if self.if_resize_to_multiple:
                resized_img = self.resize_to_multiple(resized_img)

            for flip, direction in flip_param:
                #print(flip, direction)

                if flip:

                    #if direction == 'horizontal':
                    if direction == 'h':
                        flipped_img = cv2.flip(resized_img, 1)
                        img_meta['flip'].append(direction)

                    #if direction == 'vertical':
                    if direction == 'v':
                        flipped_img = cv2.flip(resized_img, 0)
                        img_meta['flip'].append(direction)

                    if direction == 'hv':
                        flipped_img = cv2.flip(resized_img, 1)
                        flipped_img = cv2.flip(flipped_img, 0)
                        img_meta['flip'].append(direction)

                    if direction == 'r90':
                        flipped_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)
                        img_meta['flip'].append(direction)
                else:
                    img_meta['flip'].append('none')
                    flipped_img = resized_img

                img_tensor = self.img_to_tensor(flipped_img)
                norm_img = self.norm(img_tensor)

                #####################################
                #flipped_img = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
                #flipped_img = Image.fromarray(flipped_img)
                #norm_img = self.my_trans((flipped_img,))[0]
                #####################################

                # normalize + to_tensor
                # if self.transforms is not None:
                    # for trans in self.transforms:
                        # print(type(gt_seg), trans)
                        # flipped_img, gt_seg = trans(flipped_img, gt_seg)
                img_meta['imgs'].append(norm_img)
                #img_meta['imgs'].append(flipped_img)

        #img_meta['seg_map'] = self.mask_to_tensor(gt_seg)

        img_meta['seg_map'] = gt_seg

        return img_meta


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.
    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.
    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.
    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.
    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

def imresize_to_multiple(
    img: np.ndarray,
    divisor: Union[int, Tuple[int, int]],
    size: Union[int, Tuple[int, int], None] = None,
    scale_factor: Union[float, Tuple[float, float], None] = None,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    #out: Optional[np.ndarray] = None,
    #backend: Optional[str] = None
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image according to a given size or scale factor and then rounds
    up the the resized or rescaled image size to the nearest value that can be
    divided by the divisor.

    Args:
        img (ndarray): The input image.
        divisor (int | tuple): Resized image size will be a multiple of
            divisor. If divisor is a tuple, divisor should be
            (w_divisor, h_divisor).
        size (None | int | tuple[int]): Target size (w, h). Default: None.
        scale_factor (None | float | tuple[float]): Multiplier for spatial
            size. Should match input size if it is a tuple and the 2D style is
            (w_scale_factor, h_scale_factor). Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: False.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if size is not None and scale_factor is not None:
        raise ValueError('only one of size or scale_factor should be defined')
    elif size is None and scale_factor is None:
        raise ValueError('one of size or scale_factor should be defined')
    elif size is not None:
        size = to_2tuple(size)
        if keep_ratio:
            size = rescale_size((w, h), size, return_scale=False)
    else:
        size = _scale_size((w, h), scale_factor)

    divisor = to_2tuple(divisor)
    size = tuple(int(np.ceil(s / d)) * d for s, d in zip(size, divisor))

    resized_img = cv2.resize(
        img,
        size,
        interpolation=interpolation
    )
    return resized_img
    #resized_img, w_scale, h_scale = imresize(
    #    img,
    #    size,
    #    return_scale=True,
    #    interpolation=interpolation,
    #    out=out,
    #    backend=backend)
    #if return_scale:
    #    return resized_img, w_scale, h_scale
    #else:
    #    return resized_img

class ResizeToMultiple(object):
    """Resize images & seg to multiple of divisor.
    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self, size_divisor=32, interpolation=None):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    #def __call__(self, results):
    def __call__(self, img):
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        #img = results['img']
        #img = mmcv.imresize_to_multiple(
        #    img,
        #    self.size_divisor,
        #    scale_factor=1,
        #    interpolation=self.interpolation
        #    if self.interpolation else 'bilinear')
        img = imresize_to_multiple(
            img=img,
            divisor=self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation,
            keep_ratio=True
        )

        #seg_map = imresize_to_multiple(
        #    img,
        #    self.size_divisor,
        #    scale_factor=1,
        #    interpolation=cv2.INTER_NEAREST

        #)
        #results['img'] = img
        #results['img_shape'] = img.shape
        #results['pad_shape'] = img.shape

        # Align segmentation map to multiple of size divisor.
        #for key in results.get('seg_fields', []):
        #    gt_seg = results[key]
        #    gt_seg = mmcv.imresize_to_multiple(
        #        gt_seg,
        #        self.size_divisor,
        #        scale_factor=1,
        #        interpolation='nearest')
        #    results[key] = gt_seg

        return img
        #return results
        #return img, seg_map

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size_divisor={self.size_divisor}, '
                     f'interpolation={self.interpolation})')
        return repr_str


#from dataset.camvid import CamVid
#
#
#train_dataset = CamVid(
#        'data',
#        image_set='train'
#)
#
#transform = RandomScaleCrop(473)
#
#train_dataset.transforms = transform
#
#import cv2
#import time
#start = time.time()
#for i in range(100):
#    img, mask = train_dataset[i]
#
#finish = time.time()
#print(100 // (finish - start))
#
#    #cv2.imwrite('test/img{}.png'.format(i), img)
#    #cv2.imwrite('test/mask{}.png'.format(i), mask / mask.max() * 255)
#
#
#

#img = cv2.imread('/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)/testB_17.bmp')
##img = cv2.imread('/data/by/pytorch-camvid/data/camvid/images/0001TP_006870.png')
##img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
#print(img.shape)
#mask = np.random.randint(0, 2, size=img.shape[:2])
#print(mask.shape)
#trans = RandomScaleCrop(473)
#trans = RandomRotation(p=1)
#trans = Resize(100)



# Glas transforms
#train_transforms = Compose([
#            #transforms.Resize(settings.IMAGE_SIZE),
#            RandomVerticalFlip(),
#            RandomHorizontalFlip(),
#            RandomRotation(45, fill=11),
#            RandomScaleCrop(473),
#            RandomGaussianBlur(),
#            RandomHorizontalFlip(),
#            ColorJitter(0.4, 0.4),
#            #ToTensor(),
#            #Normalize(settings.MEAN, settings.STD),
#])


#import time
#start = time.time()
#for i in range(10):
#    ig, m = trans(img, mask)
#    #cv2.imwrite('test/{}.jpg'.format(i), ig)
#
#finish = time.time()
#print(finish - start)
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        #if random.randint(2):
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            #img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1):
            # img = mmcv.bgr2hsv(img)
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            # img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    #def __call__(self, results):
    #def __call__(self, img, gt_seg, **kwargs):
    def __call__(self, img, gt_seg, weight_map=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        # img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        #mode = random.randint(2)
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        #results['img'] = img
        #return results
        if weight_map is not None:
            return img, gt_seg, weight_map
        else:
            return img, gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.fill = fill

    def __call__(self, img, mask):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return center_crop(img, self.size, fill=self.fill), center_crop(mask, self.size, fill=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def elastic_transform(
    #img: np.ndarray,
    #alpha: float,
    #sigma: float,
    #alpha_affine: float,
    #interpolation: int = cv2.INTER_LINEAR,
    #border_mode: int = cv2.BORDER_REFLECT_101,
    img,
    alpha,
    sigma,
    alpha_affine,
    pts1,
    pts2,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT,
    value=None,
    # value: Optional[ImageColorType] = None,
    # random_state: Optional[np.random.RandomState] = None,
    # approximate: bool = False,
    same_dxdy: bool = False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    height, width = img.shape[:2]

    # Random affine
    # center_square = np.array((height, width), dtype=np.float32) // 2
    # square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    # alpha_affine = float(alpha_affine)

    # pts1 = np.array(
    #     [
    #         center_square + square_size,
    #         [center_square[0] + square_size, center_square[1] - square_size],
    #         center_square - square_size,
    #     ],
    #     dtype=np.float32,
    # )

    #

    #pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
    #    np.float32
    #)
    # np.random.uniform: Samples are uniformly distributed over the half-open interval [low, high)
    # (includes low, but excludes high).In other words, any value within the given interval is equally
    # likely to be drawn by uniform.
    # pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    matrix = cv2.getAffineTransform(pts1, pts2)

    #warp_fn = _maybe_process_in_chunks(
    #    cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    #)
    img = cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
    # img = warp_fn(img)

    # if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        #dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
    dx =  np.random.rand(height, width).astype(np.float32) * 2 - 1
    cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
    dx *= alpha
    if same_dxdy:
        # Speed up even more
        dy = dx
    else:
        # dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
        dy =  np.random.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        #cv2.GaussianBlur(dy, (3, 3), sigma, dst=dy)
        dy *= alpha
    #else:
    #    dx = np.float32(
    #        gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
    #    )
    #    if same_dxdy:
    #        # Speed up
    #        dy = dx
    #    else:
    #        dy = np.float32(
    #            gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
    #        )
    # print(dx.mean(), dy.mean(), dx.std(), dy.std())

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # print(dx, dy)
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    #map_x = np.float32(x)
    #map_y = np.float32(y)

    img = cv2.remap(img, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value)

    #if interpolation == cv2.INTER_NEAREST:
        #kernel = np.ones((5,5),np.uint8)
        #Wimg = cv2.erode(img, kernel, iterations=3)
        #img = cv2.dilate(img, kernel, iterations=3)

    return img


class ElasticTransform:
    def __init__(
        self,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        # border_mode=cv2.BORDER_REFLECT_101,
        same_dxdy=False,
        pad_value=0,
        seg_pad_value=255,
        p=0.5,
    ):

        self.alpha = alpha
        self.p = p
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.seg_pad_value = seg_pad_value
        self.pad_value = pad_value
        self.same_dxdy = same_dxdy
        # interpolation=cv2.INTER_LINEAR,

    def __repr__(self):
        return self.__class__.__name__ + \
            '(alpha={})'.format(self.alpha) + \
            '(sigma={})'.format(self.sigma) + \
            '(alpha_affine={})'.format(self.alpha_affine) + \
            '(p={})'.format(self.p)


    def __call__(self, img, seg_map, weight_map=None):

        #if weight_map is not None:
        if random.random() > self.p:
            if weight_map is not None:
                return img, seg_map, weight_map
            else:
                return img, seg_map

        #else:
        #    if random.random() > self.p:
        #        return img, seg_map

        height, width = img.shape[:2]

        # Random affine
        center_square = np.array((height, width), dtype=np.float32) // 2
        square_size = min((height, width)) // 3
        pts1 = np.array(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ],
            dtype=np.float32,
        )
        alpha_affine = float(self.alpha_affine)
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

        img = elastic_transform(
            img=img,
            alpha=self.alpha,
            sigma=self.sigma,
            alpha_affine=self.alpha_affine,
            pts1=pts1,
            pts2=pts2,
            interpolation=cv2.INTER_LINEAR,
            #interpolation=cv2.INTER_NEAREST,
            value=self.pad_value,
            same_dxdy=self.same_dxdy)

        seg_map = elastic_transform(
            img=seg_map,
            alpha=self.alpha,
            sigma=self.sigma,
            alpha_affine=self.alpha_affine,
            pts1=pts1,
            pts2=pts2,
            interpolation=cv2.INTER_NEAREST,
            value=self.seg_pad_value,
            same_dxdy=self.same_dxdy
        )

        if weight_map is not None:
            weight_map = elastic_transform(
            img=weight_map,
            alpha=self.alpha,
            sigma=self.sigma,
            alpha_affine=self.alpha_affine,
            pts1=pts1,
            pts2=pts2,
            interpolation=cv2.INTER_LINEAR,
            #interpolation=cv2.INTER_NEAREST,
            value=self.pad_value,
            same_dxdy=self.same_dxdy)

        if weight_map is not None:
            return img, seg_map, weight_map
        else:
            return img, seg_map

class RandomChoice():
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, *args):
        t = random.choice(self.transforms)
        return t(*args)

    #def __repr__(self) -> str:
        #return f"{super().__repr__()}(p={self.p})"

#img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/img.jpg'
#img = cv2.imread(img_path)
#trans = ResizeToMultiple()
#img = img[:244, :300]
#cv2.imwrite('resG5.png', img)
#print(img.shape, img.shape[0] / img.shape[1])
#img = trans(img)
#cv2.imwrite('resP4.png', img)
#print(img.shape, img.shape[0] / img.shape[1])
#
#img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_16.bmp'
#segmap_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_16_anno.bmp'
#img = cv2.imread(img_path)
#seg_map = cv2.imread(segmap_path, -1)
#trans = MultiScaleFlipAug(
#    #img_ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0],
#    img_ratios=[1],
#    flip=True,
#    #flip=False,
#    flip_direction=['horizontal', 'vertical'],
#    #flip_direction=['horizontal'],
#    # transforms=[
#        # transforms.ToTensor(),
#        # transforms.Normalize(settings.MEAN, settings.STD),
#    # ]
#    min_size=208,
#    mean=(0.5, 0.5, 0.5),
#    std = (1, 1, 1)
#    #std=settings.STD
#)

#trans = Resize(min_size=230)
#print(img.shape)
#imgs = trans(img, seg_map)
#for img in imgs:
    #print(img)
    #print(imgs['flip'])
    #rint(imga
    #print(imgs['seg_map'].shape)
    #print(len(imgs['imgs']))
    #print(imgs['imgs'][0])
    #for im in imgs['imgs']:
        #print(im.shape)
        #print
#img, seg_map = trans(img, seg_map)
#print(img.shape, seg_map.shape)
#seg_map[seg_map != 0] = 255
#cv2.imwrite('test1.jpg', img)
#cv2.imwrite('test2.jpg', seg_map)
#
## print(img)
#
#cv2.imwrite('res1.jpg', seg_map)
##cv2.imwrite('res1.jpg', img)
#elastic_transforms = ElasticTransform(alpha=10, sigma=3, alpha_affine=90, p=1)
#img1, seg_map = elastic_transforms(img, seg_map)
##cv2.imwrite('res.jpg', img1)
#cv2.imwrite('res.jpg', seg_map[0])
#cv2.imwrite('seg_map.jpg', seg_map[1])
#

# print(np.unique(seg_map))

# print((seg_map - img1).mean())


##
#crop_size=(480, 480)
#trans = Resize(range=[0.5, 1.5])  # 0.0004977783894538879
#trans1 = RandomRotation(degrees=10, expand=True)  # 0.0017581235980987549
#trans2 = RandomCrop(crop_size=crop_size, cat_max_ratio=0.75, pad_if_needed=True) # 0.007527546215057373
## trans
##trans3 = RandomVerticalFlip()
##trans4 = RandomHorizontalFlip()
##
#
#
#
###trans = Compose([
###    Resize(range=[0.5, 1.5]),
###    RandomRotation(degrees=90, expand=True, pad_value=0, seg_pad_value=255),
###])
### trans5 = ColorJitter(p=0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
#trans5 = PhotoMetricDistortion() # 0.0068617223739624025
#
#img = cv2.imread(img_path)
##
##print(img.shape)
##
#import time
#start = time.time()
#times = 10000
#for _ in range(times):
#    _, _ = trans5(img, img[:, :, 0])
##    # _ = trans5(img, img[:, :, 0])
#finish = time.time()
#print((finish - start) / times)
##img, mask = trans5(img, img[:, :, 0])
##img, mask = trans(img, mask)
##img, mask = trans1(img, mask)
##img, mask = trans2(img, mask)
##img, mask = trans3(img, mask)
##img, mask = trans4(img, mask)
### print(finish - start)
### print(img.shape, mask.shape)
##
##print(img.shape, mask.shape)
##cv2.imwrite('src.jpg', img)
##cv2.imwrite('src1.jpg', mask)

#crop_size=(208, 208)
#trans = Compose([
#            ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=0.5),
#            RandomRotation(degrees=90, expand=True),
#            Resize(min_size=208 + 30),
#            RandomCrop(crop_size=crop_size, cat_max_ratio=0.75, pad_if_needed=True),
#            RandomVerticalFlip(),
#            RandomHorizontalFlip(),
#            RandomApply(
#                transforms=[PhotoMetricDistortion()]
#            ),
#            #transforms.ToTensor(),
#            #transforms.Normalize(settings.MEAN, settings.STD)
#        ])
#
#
#from dataset.glas import Glas
##img_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_16.bmp'
##segmap_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_16_anno.bmp'
#dataset = Glas('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data', image_set='train', transforms=trans)
#
#import random
#img, segmap, weight_map = random.choice(dataset)
#
#cv2.imwrite('test.jpg', img)
#cv2.imwrite('test1.jpg', segmap * 50)
#cv2.imwrite('test2.jpg', weight_map)