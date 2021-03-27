__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import os

from utils.utils import BBoxTransform, ClipBoxes, generate_coordinates, generate_corner
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms
import numpy as np
from utils import parell_util

base_dir = r""
target_size = 1

xym = generate_coordinates()


def to_numpy(tensor):
    return tensor.cpu().numpy()


def smooth_dist(dist):
    weights = torch.tensor([1 / 9] * 9).view((1, 1, 3, 3)).to(dist.device)
    return nn.functional.conv2d(dist.unsqueeze(0).unsqueeze(0), weights, padding=1).squeeze(0).squeeze(0)


def group_instance_map(ae_mat, boxes_cls, boxes_confs, boxes, center_embeddings, device):
    """
    group the bounds key points
    :param hm_kp: heat map for key point, 0-1 mask, 2-dims:h*w
    :param hm_ae: heat map for associate embedding, 2-dims:h*w
    :param transforms: transforms for task
    :param center_indexes: the object centers
    :return: the groups
    """
    objs_num = len(boxes_cls)
    h, w = ae_mat.shape[1:]
    xym_s = xym[:, 0:h, 0:w].contiguous().to(device)
    spatial_emb = torch.tanh(ae_mat[0:2, :, :]) + xym_s

    boxes_lt = boxes[:, :2][:, ::-1]
    boxes_rb = boxes[:, 2:][:, ::-1]
    center_indexes = ((boxes_lt + boxes_rb) / 2).astype(np.int32)
    boxes_wh = (boxes_rb - boxes_lt).astype(np.int32)
    
    n_boxes_cls, n_boxes_confs, n_boxes, instance_ids = [], [], [], []
    instance_map = torch.zeros(h, w, dtype=torch.uint8, device=device)
    conf_map = torch.zeros(h, w, dtype=torch.float32, device=device)
    for i in range(objs_num):
        center_index = center_indexes[i]
        box_wh = boxes_wh[i]

        if box_wh[0] < 2 or box_wh[1] < 2:
            continue

        lt, rb = generate_corner(center_index, box_wh, h, w, 1.0)
        selected_spatial_emb = spatial_emb[:, lt[0]:rb[0], lt[1]:rb[1]]

        if center_embeddings is not None:
            center = torch.tanh(center_embeddings[i][:2]).view(2, 1, 1)
            s = torch.exp(center_embeddings[i][2])
        else:
            center = xym_s[:, center_index[0], center_index[1]].view(2, 1, 1)
            s = torch.exp(ae_mat[2:3, center_index[0], center_index[1]])

        dist = torch.exp(-1 * torch.sum(torch.pow(selected_spatial_emb -
                                                  center, 2) * s, 0, keepdim=True)).squeeze()

        # dist = smooth_dist(dist)
        proposal = (dist > 0.5)
        # resolve the conflicts
        box_h, box_w = proposal.shape
        area = box_h * box_w
        if proposal.sum().item() < 128 or proposal.sum().item() / area < 0.3:
            continue

        # nms
        instance_map_cut = instance_map[lt[0]:rb[0], lt[1]:rb[1]]
        occupied_ids = instance_map_cut.unique().cpu().numpy()
        skip = False
        for occupied_id in occupied_ids:
            if occupied_id == 0:
                continue
            other_proposal = instance_map_cut.eq(occupied_id)
            overlapped_area = other_proposal * proposal
            if overlapped_area.sum().item() / proposal.sum().item() >= 0.5:
                skip = True
                break

        if skip or proposal.sum().item() < 128 or proposal.sum().item() / area < 0.3:
            continue

        cls_id = boxes_cls[i]
        conf = boxes_confs[i]
        instance_id = i + 1

        instance_map[lt[0]:rb[0], lt[1]:rb[1]][proposal] = instance_id
        conf_map[lt[0]:rb[0], lt[1]:rb[1]][proposal] = dist[proposal]

        n_boxes_cls.append(cls_id)
        n_boxes_confs.append(conf)
        n_boxes.append(boxes[i])
        instance_ids.append(instance_id)

    return n_boxes_cls, n_boxes_confs, n_boxes, instance_ids, instance_map.cpu().numpy()


def decode_boxes(x, anchors, box_regression, center_regression, classification, threshold, iou_threshold):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    transformed_anchors = regressBoxes(anchors, box_regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]

    dets = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            dets.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'embeddings': np.array(())
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        embedding_per = center_regression[i, scores_over_thresh[i, :], :]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            embedding_per = embedding_per[anchors_nms_idx, :]

            dets.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
                'embeddings': embedding_per
            })
        else:
            dets.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'embeddings': np.array(())
            })

    return dets


def decode_single(ae_mat, dets, device):
    cls_ids, boxes, confs, center_embeddings = dets["class_ids"], dets["rois"], dets["scores"], dets["embeddings"]
    if len(cls_ids) == 0:
        return ([], [])

    cls_ids, confs, boxes, instance_ids, instance_map = group_instance_map(ae_mat, cls_ids, confs, boxes,
                                                                    center_embeddings, device)

    return ((np.array(cls_ids, dtype=np.uint8), np.array(confs, dtype=np.float), np.vstack(boxes), np.array(instance_ids, dtype=np.uint16)), instance_map)


def covert_boxlist_to_det_boxes(det_results):
    b = len(det_results)
    det_boxes = []
    for b_i in range(b):
        if det_results[b_i][0].shape[0] == 0:
            det_boxes.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'embeddings': None
            })
        else:
            det_boxes.append({
                'rois': det_results[b_i][0][:, :4].cpu().numpy(),
                'class_ids': det_results[b_i][1][:].cpu().numpy(),
                'scores': det_results[b_i][0][:, 4].cpu().numpy(),
                'embeddings': None
            })
    return det_boxes


def decode_output(inputs, model, outs, decode_cfg, device):
    """
    decode the model output
    :param outs:
    :param infos:
    :param transforms:
    :param decode_cfg:
    :param device:
    :return:
    """
    # get output
    if decode_cfg.model_type == "eff":
        spatial_out, box_regression, center_regression, classification, anchors = outs
        det_boxes = decode_boxes(inputs, anchors, box_regression, center_regression, classification, decode_cfg.cls_th,
                                 decode_cfg.iou_th)
    elif decode_cfg.model_type == "fcos":
        spatial_out, cls_scores, bbox_preds, centernesses = outs
        det_results = model.bbox_head.get_bboxes(cls_scores, bbox_preds, centernesses, [{"img_shape": spatial_out.shape[2:], "scale_factor": 1} for i in range(spatial_out.shape[0])])
        det_boxes = covert_boxlist_to_det_boxes(det_results)
    else:
        raise RuntimeError("no support for model type:%s"%decode_cfg.model_type)

    dets = parell_util.multi_apply(decode_single, spatial_out, det_boxes, device=device)

    return dets[0], dets[1]