__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"
import cv2
from collections import namedtuple
import torch
import numpy as np
from PIL import Image

TransInfo = namedtuple('TransInfo', ['img_path', 'img_size', 'scale', 'scaled_size', 'pad_size'])


class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value)
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs


class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(0)):
            result[i, :, :] = result[i, :, :] * self.std[i] + self.mean[i]

        return result.mul_(self.div_value)


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            channels = len(inputs.mode)
            inputs = np.array(inputs)
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], channels)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
        else:
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=(512, 1024)):
        self.img_size = img_size
        self.color = (114, 114, 114)

    def __call__(self, image, label=None):
        if self.img_size[0] == -1:
            return image, label, 1.0, image.shape[1::-1], (0, 0)

        height, width, _ = image.shape
        scale = min(self.img_size[0]/height, self.img_size[1]/width)

        resized_height = int(height * scale)
        resized_width = int(width * scale)

        pad_left = (self.img_size[1]-resized_width)//2
        pad_right = self.img_size[1]-resized_width-pad_left
        pad_top = self.img_size[0]-resized_height
        pad_bottom = self.img_size[0]-resized_height-pad_top

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        if label is not None:
            class_ids, instance_map = label

            instance_map = cv2.resize(instance_map, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            new_instance_map = cv2.copyMakeBorder(instance_map, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)  # add border

            class_ids[:, :4] = class_ids[:, :4]*scale+np.array(((pad_left, pad_top, pad_left, pad_top)), dtype=np.float32)
            label = (class_ids, new_instance_map)

        return new_image, label, scale, (resized_width, resized_height), (pad_left, pad_top)


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image, label, flip_x=0.5):
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :].copy()
            if label is not None:
                class_ids, instance_map = label
                instance_map = instance_map[:, ::-1].copy()

                height, width = instance_map.shape
                class_ids[:, 0] = width - class_ids[:, 0]
                class_ids[:, 2] = width - class_ids[:, 2]

                label = (class_ids, instance_map)

        return image, label


class CommonTransforms(object):
    def __init__(self, trans_cfg, phase="train", device=None):
        self.configer = trans_cfg
        self.normalizer = Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std'))
        self.to_tensor = ToTensor()
        self.resizer = Resizer(self.configer.get('resize_tar'))
        self.augmenter = Augmenter()
        self.device = device
        self.phase = phase

    def __call__(self, img, label=None, img_path=None, img_size=None):
        """
        compose transform the all the transform
        :param img:  rgb and the shape is h*w*c
        :param label: cls_ids, polygons, the pixel of polygons format as (w,h)
        :return:
        """
        img, label, scale, scaled_size, pad_size = self.resizer(img, label)
        if self.phase == 'train':
            img, label = self.augmenter(img, label)
        input_tensor = self.to_tensor(img)
        if self.device is not None:
            input_tensor = input_tensor.to(self.device)
        input_tensor = self.normalizer(input_tensor)
        return input_tensor, label, TransInfo(img_path, img_size, scale, scaled_size, pad_size)