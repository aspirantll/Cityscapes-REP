__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.
    
    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import torch
import torch.nn as nn
import numpy as np

from models.lovasz_losses import lovasz_hinge
from utils.target_generator import generate_all_annotations, generate_fcos_annotations
from utils.utils import generate_coordinates, convert_corner_to_corner


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


class DetFocalLoss(nn.Module):
    def __init__(self):
        super(DetFocalLoss, self).__init__()

    def forward(self, classifications, box_regressions, center_regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        embedding_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        center_embeddings = []
        for j in range(batch_size):

            classification = classifications[j, :, :]
            box_regression = box_regressions[j, :, :]
            center_regression = center_regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            box_num = bbox_annotation.shape[0]
            if box_num == 0:
                center_embeddings.append([])
                if torch.cuda.is_available():
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    embedding_losses.append(torch.tensor(0).to(dtype).cuda())
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    embedding_losses.append(torch.tensor(0).to(dtype))
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            embeddings = []
            if positive_indices.sum() > 0:
                positive_argmax = IoU_argmax[positive_indices]
                positive_regressions = center_regression[positive_indices, :]
                positive_ann_indices = positive_argmax.unique()

                embedding_loss = torch.tensor(0).to(dtype)
                if torch.cuda.is_available():
                    embedding_loss = embedding_loss.cuda()

                for ann_ind in range(box_num):
                    if ann_ind not in positive_ann_indices:
                        target_embedding = torch.zeros((3), dtype=dtype)
                    else:
                        anchor_indices = positive_argmax == ann_ind
                        target_embedding = positive_regressions[anchor_indices].mean(dim=0)
                        embedding_loss = embedding_loss + torch.mean(torch.pow(positive_regressions[anchor_indices] - target_embedding.detach(), 2))
                    embeddings.append(target_embedding)

                embedding_losses.append(embedding_loss/max(box_num, 1))

                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - box_regression[positive_indices, :4])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    embedding_losses.append(torch.tensor(0).to(dtype).cuda())
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    embedding_losses.append(torch.tensor(0).to(dtype))
                    regression_losses.append(torch.tensor(0).to(dtype))

                for ann_ind in range(box_num):
                    target_embedding = torch.zeros((3), dtype=dtype)
                    embeddings.append(target_embedding)

            center_embeddings.append(embeddings)

        return [torch.stack(classification_losses).mean(dim=0),
                torch.stack(regression_losses).mean(dim=0) * 50,
                torch.stack(embedding_losses).mean(dim=0)], center_embeddings  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


class AELoss(object):
    def __init__(self, device, weight=1):
        self._device = device
        self._weight = weight
        self._xym = generate_coordinates().to(device)

    def __call__(self, ae, targets, center_embeddings):
        """
        :param ae:
        :param targets: (instance_map_list)
        :return:
        """
        # prepare step
        det_annotations, instance_ids_list, instance_map_list = targets
        b, c, h, w = ae.shape

        xym_s = self._xym[:, 0:h, 0:w].contiguous()  # 2 x h x w

        ae_loss = zero_tensor(self._device)
        for b_i in range(b):
            instance_ids = instance_ids_list[b_i]
            instance_map = instance_map_list[b_i]

            n = len(instance_ids)
            if n <= 0:
                continue

            spatial_emb = torch.tanh(ae[b_i, 0:2]) + xym_s  # 2 x h x w

            instance_loss = zero_tensor(self._device)

            for o_j, instance_id in enumerate(instance_ids):
                in_mask = instance_map.eq(instance_id).view(1, h, w) # 1 x h x w

                # calculate center of attraction
                o_lt = det_annotations[b_i, o_j, 0:2][::-1].astype(np.int32)
                o_rb = det_annotations[b_i, o_j, 2:4][::-1].astype(np.int32)

                center_embedding = center_embeddings[b_i][o_j]
                if center_embedding.eq(0).all():
                    continue

                s = torch.exp(center_embedding[2])

                # limit 2*box_size mask
                lt, rb = convert_corner_to_corner(o_lt, o_rb, h, w, 1.5)
                selected_spatial_emb = spatial_emb[:, lt[0]:rb[0], lt[1]:rb[1]]
                label_mask = in_mask[:, lt[0]:rb[0], lt[1]:rb[1]].float()
                center = torch.tanh(center_embeddings[b_i][o_j][:2]).view(2,1,1)
                # calculate gaussian
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(selected_spatial_emb - center, 2) * s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, label_mask)

            ae_loss += instance_loss / max(n, 1)
        # compute mean loss
        return ae_loss / b


class EffLoss(nn.Module):
    def __init__(self, device):
        super(EffLoss, self).__init__()
        self._device = device
        self._loss_names = ["cls_loss", "wh_loss", "center_loss", "ae_loss", "total_loss"]
        self.det_focal_loss = DetFocalLoss()
        self.ae_loss = AELoss(device)

    def forward(self, model, outputs, targets):
        # unpack the output
        spatial_out, box_regression, center_regression, classification, anchors = outputs
        det_annotations, instance_ids_list, instance_map_list = generate_all_annotations(spatial_out.shape, targets, self._device)

        losses = []
        det_losses, center_embeddings = self.det_focal_loss(classification, box_regression, center_regression, anchors,
                                          torch.from_numpy(det_annotations).to(self._device))
        losses.extend(det_losses)
        losses.append(self.ae_loss(spatial_out, (det_annotations, instance_ids_list, instance_map_list), center_embeddings))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names


class FCOSLoss(nn.Module):
    def __init__(self, device):
        super(FCOSLoss, self).__init__()
        self._device = device
        self._loss_names = ["cls_loss", "wh_loss", "center_loss", "embedding_loss", "ae_loss", "total_loss"]
        self.ae_loss = AELoss(device)

    def forward(self, model, outputs, targets):
        # unpack the output
        spatial_out, cls_scores, bbox_preds, centernesses, center_embeddings = outputs
        gt_boxes, gt_labels, det_annotations, instance_ids_list, instance_map_list = generate_fcos_annotations(spatial_out.shape, targets, self._device)
        det_losses, pred_embeddings = model.bbox_head.loss(cls_scores, bbox_preds, centernesses, center_embeddings, gt_boxes, gt_labels, None)

        losses = []
        losses.extend(det_losses)
        losses.append(self.ae_loss(spatial_out, (det_annotations, instance_ids_list, instance_map_list), pred_embeddings))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names






