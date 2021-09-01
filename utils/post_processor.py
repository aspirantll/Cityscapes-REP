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

from utils.utils import BBoxTransform, ClipBoxes, generate_coordinates, generate_corner
import torch
from torchvision.ops.boxes import batched_nms
import numpy as np
from utils import parell_util

xym = generate_coordinates()
mask_cls_ids = [8, 9, 10]
nms_mutex = [[4, 5, 6, 7], [8, 9]]


def to_numpy(tensor):
    return tensor.cpu().numpy()


def nms(transformed_anchors_per, scores_per, classes_, iou_threshold):
    indexes = batched_nms(transformed_anchors_per, scores_per, classes_, iou_threshold=iou_threshold)

    boxes, scores, classes = transformed_anchors_per[indexes], scores_per[indexes], classes_[indexes].clone()

    # replace class id
    for group in nms_mutex:
        for i in range(1, len(group)):
            classes[classes==group[i]] = group[0]
    inner_indexes = batched_nms(boxes, scores, classes, 0.8)

    return indexes[inner_indexes]


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]

    return res[0].reshape(-1, 2) if len(res)>0 else np.array(())


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

    n_boxes_cls, n_boxes_confs, n_boxes, polygons = [], [], [], []
    for i in range(objs_num):
        center_index = center_indexes[i]
        box_wh = boxes_wh[i]

        if box_wh[0] < 2 or box_wh[1] < 2:
            continue

        cls_id = boxes_cls[i]
        conf = boxes_confs[i]

        polygon = np.empty(0)
        if cls_id in mask_cls_ids:

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

            mask = to_numpy(proposal)
            polygon = mask_to_polygons(mask)
            if not isinstance(polygon, np.ndarray) or len(polygon) == 0:
                continue
            polygon = polygon + lt[::-1]

        polygons.append(polygon)
        n_boxes_cls.append(cls_id)
        n_boxes_confs.append(conf)
        n_boxes.append(boxes[i])

    return n_boxes_cls, n_boxes_confs, n_boxes, polygons


def decode_boxes(x, anchors, box_regression, center_regression, classification, threshold, iou_threshold):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    transformed_anchors = regressBoxes(anchors, box_regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]

    top_scores, top_inds = scores.topk(1000, 1)
    scores_over_thresh = (top_scores > threshold)[:, :, 0]

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

        classification_per = classification[i, top_inds[i, :, 0], ...][scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, top_inds[i, :, 0], ...][scores_over_thresh[i, :], ...]
        embedding_per = center_regression[i, top_inds[i, :, 0], ...][scores_over_thresh[i, :], ...]
        scores_per = top_scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
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


def decode_single(ae_mat, dets, info, device):
    cls_ids, boxes, confs, center_embeddings = dets["class_ids"], dets["rois"], dets["scores"], dets["embeddings"]
    if len(cls_ids) == 0:
        return np.array(()), np.array(()), np.array(()), np.array(())

    cls_ids, confs, boxes, polygons = group_instance_map(ae_mat, cls_ids, confs, boxes,
                                                                    center_embeddings, device)

    if len(cls_ids) == 0:
        return np.array(()), np.array(()), np.array(()), np.array(())
    else:
        np_pad = np.array(info.pad_size, dtype=np.float32).reshape(-1, 2)
        return np.array(cls_ids, dtype=np.uint8), np.array(confs, dtype=np.float), (np.vstack(boxes)-np.hstack((np_pad, np_pad)))/info.scale, [(polygon-np_pad)/info.scale if len(polygon)>0 else polygon for polygon in polygons]


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


def decode_output(inputs, model, outs, decode_cfg, infos, device):
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
    else:
        raise RuntimeError("no support for model type:%s"%decode_cfg.model_type)

    return [decode_single(spatial_out[i], det_boxes[i], infos[i], device=device) for i in range(len(det_boxes))]