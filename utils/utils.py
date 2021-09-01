# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union

import cv2
import numpy as np
import torch
import webcolors
import itertools
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torchvision.ops.boxes import batched_nms

from utils.sync_batchnorm import SynchronizedBatchNorm2d


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def to_numpy(tensor):
    return tensor.cpu().numpy()


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path)[..., ::-1] for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


color_list = standard_to_bgr(STANDARD_COLORS)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float32:
            dtype = np.float32
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes


def generate_coordinates():
    xm = torch.linspace(0, 2, 2048).view(
        1, 1, -1).expand(1, 1024, 2048)
    ym = torch.linspace(0, 1, 1024).view(
        1, -1, 1).expand(1, 1024, 2048)
    return torch.cat((ym, xm), 0)


def generate_corner(center, wh, max_h, max_w, delta_factor):
    delta = (wh / 2)*delta_factor
    lt = np.clip((center - delta).astype(np.int32), a_min=0, a_max=max(max_h, max_w))
    rb = (center + delta).astype(np.int32)
    rb[0] = np.clip(rb[0], a_min=0, a_max=max_h)
    rb[1] = np.clip(rb[1], a_min=0, a_max=max_w)
    return lt, rb


def convert_corner_to_corner(o_lt, o_rb, max_h, max_w, delta_factor):
    center = (o_lt+o_rb)/2
    wh = o_rb - o_lt
    return generate_corner(center, wh, max_h, max_w, delta_factor)