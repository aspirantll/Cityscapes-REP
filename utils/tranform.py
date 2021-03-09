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

TransInfo = namedtuple('TransInfo', ['img_path', 'img_size'])

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


class CoordinateReverser(object):
    def __call__(self, label):
        cls_ids, polygons = label
        polygons = [poly[:, ::-1].astype(np.int32) for poly in polygons]
        return cls_ids, polygons


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, image, label=None):
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        if label is not None:
            class_map, instance_map = label

            class_map = cv2.resize(class_map, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            new_class_map = np.zeros((self.img_size, self.img_size))
            new_class_map[0:resized_height, 0:resized_width] = class_map

            instance_map = cv2.resize(instance_map, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            new_instance_map = np.zeros((self.img_size, self.img_size))
            new_instance_map[0:resized_height, 0:resized_width] = instance_map

            label = (new_class_map, new_instance_map)

        return new_image, label, scale


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, label, flip_x=0.5):
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :].copy()
            if label is not None:
                class_map, instance_map = label
                class_map = class_map[:, ::-1].copy()
                instance_map = instance_map[:, ::-1].copy()
                label = (class_map, instance_map)

        return image, label


class Normalizer(object):

    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return ((image.astype(np.float32)/self.div_value - self.mean) / self.std).astype(np.float32)


class CommonTransforms(object):
    def __init__(self, trans_cfg, phase="train"):
        self.configer = trans_cfg
        self.normalizer = Normalizer(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std'))
        self.aug = Augmenter()
        # self.resizer = Resizer(input_size)
        self.to_tensor = ToTensor()
        self.phase = phase

    def __call__(self, img, label=None, img_path=None):
        """
        compose transform the all the transform
        :param img:  rgb and the shape is h*w*c
        :param label: cls_ids, polygons, the pixel of polygons format as (w,h)
        :param img_path: as the key
        :return:
        """
        img_size = img.shape[:2]
        img = self.normalizer(img)
        if self.phase == "train":
            img, label = self.aug(img, label)
        input_tensor = self.to_tensor(img)
        return input_tensor, label, TransInfo(img_path, img_size)