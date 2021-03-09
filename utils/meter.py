import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


class APMeter(object):
    def __init__(self, num_scales):
        self.num_scales = num_scales
        self.reset()

    def reset(self):
        self.num_dets = 0
        self.num_gts = np.zeros(self.num_scales, dtype=int)
        self.tps = []
        self.fps = []
        self.confs = []

    def update(self, tp, fp, conf, num_dets, num_gts):
        self.__extend_tp(tp)
        self.__extend_fp(fp)
        self.__extend_conf(conf)
        self.__add_num_dets(num_dets)
        self.__add_num_gts(num_gts)

    def __extend_tp(self, tp):
        self.tps.extend(tp)

    def __extend_fp(self, fp):
        self.fps.extend(fp)

    def __extend_conf(self, conf):
        self.confs.extend(conf)

    def __add_num_dets(self, num):
        self.num_dets += num

    def __add_num_gts(self, num):
        self.num_gts += num

    def eval(self, scale_ranges=None, dataset=None):
        conf = np.hstack(self.confs)
        tp = np.hstack(self.tps)
        fp = np.hstack(self.fps)
        num_gts = self.num_gts
        num_dets = self.num_dets

        sort_inds = np.argsort(- conf)
        tp = tp[:, sort_inds]
        fp = fp[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)

        return {
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        }

