#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import cv2
import numpy as np
from utils import image


def filter_bounds(transformed_poly, size):
    # group the boundary
    filtered_polygon = []
    pre_pt = None
    pre_pt_flag = -1
    bound_flags = [False, False, False, False, False]
    for point in transformed_poly:
        if point[0] == 0:
            pt_flag = 0
            bound_flags[0] = True
        elif point[1] == 0:
            pt_flag = 1
            bound_flags[1] = True
        elif point[0] == size[0] - 1:
            pt_flag = 2
            bound_flags[2] = True
        elif point[1] == size[1] - 1:
            pt_flag = 3
            bound_flags[3] = True
        else:
            pt_flag = -1
            bound_flags[4] = True

        if pre_pt_flag != pt_flag or pt_flag == -1:
            if pre_pt is not None:
                filtered_polygon.append(pre_pt)
            filtered_polygon.append(point)
            pre_pt = None
        else:
            pre_pt = point

        pre_pt_flag = pt_flag
    keep = bound_flags[4] or (bound_flags[0] and bound_flags[1] and bound_flags[2] and bound_flags[3])
    return keep, np.vstack(filtered_polygon)


def transform_label(label, transform_matrix, target_size):
    cls_ids, polygons = label
    new_cls_ids = []
    new_polygons = []
    for index, poly in enumerate(polygons):
        transformed_poly = image.apply_affine_transform(poly, transform_matrix, target_size)
        keep, filtered_poly = filter_bounds(transformed_poly, target_size)
        if keep:
            new_cls_ids.append(cls_ids[index])
            new_polygons.append(filtered_poly)
    return new_cls_ids, new_polygons


def crop_label(label, lt_pt, size):
    cls_ids, polygons = label
    new_cls_ids = []
    new_polygons = []
    for index, poly in enumerate(polygons):
        cropped_poly = poly.copy() - np.array(lt_pt)
        cropped_poly[:, 0] = cropped_poly[:, 0].clip(min=0, max=size[0] - 1)
        cropped_poly[:, 1] = cropped_poly[:, 1].clip(min=0, max=size[1] - 1)

        keep, filtered_poly = filter_bounds(cropped_poly, size)
        if keep:
            new_cls_ids.append(cls_ids[index])
            new_polygons.append(filtered_poly)
    return new_cls_ids, new_polygons


class Padding(object):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """

    def __init__(self, pad=None, pad_ratio=0.5, mean=(104, 117, 123), allow_outside_center=True):
        self.pad = pad
        self.ratio = pad_ratio
        self.mean = mean
        self.allow_outside_center = allow_outside_center

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        height, width, channels = img.shape
        left_pad, up_pad, right_pad, down_pad = self.pad

        target_size = [width + left_pad + right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        expand_image = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                 max(width, target_size[0]) + abs(offset_left), channels), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
        abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img
        img = expand_image[max(offset_up, 0):max(offset_up, 0) + target_size[1],
              max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        if label is not None:
            cls_ids, polygons = label
            for poly in polygons:
                poly[:, 0] += abs(min(offset_up, 0))
                poly[:, 1] += abs(min(offset_left, 0))

        return img, label


class RandomHFlip(object):
    def __init__(self, swap_pair=None, flip_ratio=0.5):
        self.swap_pair = swap_pair
        self.ratio = flip_ratio

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        height, width, _ = img.shape
        img = cv2.flip(img, 1)
        if label is not None:
            cls_ids, polygons = label
            for poly in polygons:
                poly[:, 0] = width - poly[:, 0] - 1

        return img, label


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, saturation_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = saturation_ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, label


class RandomHue(object):
    def __init__(self, delta=18, hue_ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = hue_ratio

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, label


class RandomPerm(object):
    def __init__(self, perm_ratio=0.5):
        self.ratio = perm_ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        swap = self.perms[random.randint(0, len(self.perms) - 1)]
        img = img[:, :, swap].astype(np.uint8)
        return img, label


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, contrast_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = contrast_ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        img = img.astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img, label


class RandomBrightness(object):
    def __init__(self, shift_value=30, brightness_ratio=0.5):
        self.shift_value = shift_value
        self.ratio = brightness_ratio

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img, label


class RandomResize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', max_side_bound=None, scale_list=None, resize_ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.max_side_bound = max_side_bound
        self.scale_list = scale_list
        self.method = method
        self.ratio = resize_ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError('Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            print('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img, label=None):
        """
        Args:
            img     (Image):   Image to be resized.
            label    (tuple):   label to be resized.

        Returns:
            Image:  Randomly resize image.
            tuple:  Randomly resize label.
            list:   Randomly resize center points.
        """
        assert isinstance(img, np.ndarray)

        height, width, _ = img.shape
        if random.random() < self.ratio:
            if self.scale_list is None:
                scale_ratio = self.get_scale([width, height])
            else:
                scale_ratio = self.scale_list[random.randint(0, len(self.scale_list)-1)]

            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
            if self.max_side_bound is not None and max(height*h_scale_ratio, width*w_scale_ratio) > self.max_side_bound:
                d_ratio = self.max_side_bound / max(height * h_scale_ratio, width * w_scale_ratio)
                w_scale_ratio *= d_ratio
                h_scale_ratio *= d_ratio

        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
        transform_matrix = image.get_affine_transform(img.shape[:2][::-1], converted_size)
        img = cv2.warpAffine(img, transform_matrix, converted_size)
        if label is not None:
            label = transform_label(label, transform_matrix, converted_size)

        return img, label


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, rotate_ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = rotate_ratio
        self.mean = mean

    def __call__(self, img, label=None):
        """
        Args:
            img    (Image):     Image to be rotated.
            maskmap   (Image):     Mask to be rotated.
            kpt    (list):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        assert isinstance(img, np.ndarray)

        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img, label

        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean).astype(np.uint8)
        if label is not None:
            label = transform_label(label, rotate_mat, (new_height, new_width))

        return img, label


class RandomCrop(object):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, crop_ratio=0.5, method='random', grid=None, allow_outside_center=True):
        self.ratio = crop_ratio
        self.method = method
        self.grid = grid
        self.allow_outside_center = allow_outside_center

        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            print('Crop method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img, label=None):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        assert isinstance(img, np.ndarray)

        if random.random() > self.ratio:
            return img, label

        height, width, _ = img.shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]

        offset_left, offset_up = self.get_lefttop(target_size, [width, height])

        img = img[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]
        if label is not None:
            label = crop_label(label, (offset_left, offset_up), target_size)

        return img, label


class Resize(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.
    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img, label=None):
        assert isinstance(img, np.ndarray)

        height, width, _ = img.shape
        scale = 1/self.target_size
        resized_height = int(height*scale)
        resized_width = int(width * scale)

        image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            cls_ids, polygons = label
            label = (cls_ids, [polygon*scale for polygon in polygons])

        return image, label


class CV2AugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> CV2AugCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, configer, split='train'):
        self.configer = configer
        self.split = split

        self.transforms = dict()
        if self.split == 'train':
            shuffle_train_trans = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    train_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    for train_trans_seq in train_trans_seq_list:
                        shuffle_train_trans += train_trans_seq

                else:
                    shuffle_train_trans = self.configer.get('train_trans', 'shuffle_trans_seq')

            if 'random_saturation' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('train_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('train_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('train_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('train_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('train_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('train_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('train_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('train_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('train_trans', 'random_contrast')['ratio']
                )

            if 'padding' in self.configer.get('train_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('train_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('train_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('train_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('train_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('train_trans', 'random_brightness')['ratio']
                )

            if 'random_hflip' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('train_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('train_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_resize')['method'] == 'random':
                    if 'scale_list' not in self.configer.get('train_trans', 'random_resize'):
                        if 'max_side_bound' in self.configer.get('train_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('train_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                    else:
                        if 'max_side_bound' in self.configer.get('train_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('train_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('train_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('train_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('train_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('train_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                else:
                    print('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        grid=self.configer.get('train_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    print('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('train_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('train_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if 'target_size' in self.configer.get('train_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('train_trans', 'resize')['target_size']
                    )

        else:
            if 'random_saturation' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('val_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('val_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('val_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('val_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('val_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('val_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('val_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('val_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('val_trans', 'random_contrast')['ratio']
                )

            if 'padding' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('val_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('val_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('val_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('val_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('val_trans', 'random_brightness')['ratio']
                )

            if 'random_hflip' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('val_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('val_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('train_trans', 'random_resize')['method'] == 'random':
                    if 'scale_list' not in self.configer.get('val_trans', 'random_resize'):
                        if 'max_side_bound' in self.configer.get('val_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('val_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                    else:
                        if 'max_side_bound' in self.configer.get('val_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('val_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('val_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('val_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('val_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('val_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                else:
                    print('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('val_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        grid=self.configer.get('val_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    print('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('val_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('val_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('val_trans', 'trans_seq'):
                if 'target_size' in self.configer.get('val_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('val_trans', 'resize')['target_size']
                    )

    def __call__(self, img, label=None):

        if self.split == 'train':
            shuffle_trans_seq = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    shuffle_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
                else:
                    shuffle_trans_seq = self.configer.get('train_trans', 'shuffle_trans_seq')
                    random.shuffle(shuffle_trans_seq)

            for trans_key in (shuffle_trans_seq + self.configer.get('train_trans', 'trans_seq')):
                img, label = self.transforms[trans_key](img, label)

        else:
            for trans_key in self.configer.get('val_trans', 'trans_seq'):
                img, label = self.transforms[trans_key](img, label)

        return img, label
