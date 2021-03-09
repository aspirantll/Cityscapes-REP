__copyright__ = \
    """
    This code is based on
        # CenterNet (https://github.com/xingyizhou/CenterNet)
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import numpy as np
import cv2

from scipy.sparse import lil_matrix, issparse
from skimage.measure import find_contours


def random_crop(img_size):
    """
    random crop the image.
    :param img_size: (height, width)
    :return: center point and size
    """
    # random size
    out_size = img_size * np.random.choice(np.arange(0.6, 1.4, 0.1))
    # determine the center range
    def get_border(border, size):
        while size - border <= border:
            border = border//2
        return border
    default_border = 128
    w_border = get_border(default_border, img_size[1])
    h_border = get_border(default_border, img_size[0])
    # random center, the form is (h,w), and center = size/2
    center = np.zeros(2)
    center[0] = np.random.randint(low=h_border, high=img_size[0]-h_border)//2 * 2
    center[1] = np.random.randint(low=w_border, high=img_size[1]-w_border)//2 * 2
    # clamp
    out_size[0] = int(min(out_size[0], 2*center[0], 2*(img_size[0]-center[0])))
    out_size[1] = int(min(out_size[1], 2*center[1], 2*(img_size[1]-center[1])))

    return center, out_size


def get_affine_transform(in_size, out_size, inv=False):
    """
    get transform matrix from in to out
    :param in_size: h * w
    :param out_size: h * w
    :return:
    """
    # create the src points
    src = np.array([[0, 0], [0, in_size[1]-1], [in_size[0]-1, in_size[1]-1]], dtype=np.float32)
    # create the dst points
    dst = np.array([[0, 0], [0, out_size[1]-1], [out_size[0]-1, out_size[1]-1]], dtype=np.float32)
    # get affine transforms
    if not inv:
        return cv2.getAffineTransform(src, dst)
    else:
        return cv2.getAffineTransform(dst, src)


def apply_affine_transform(pts, t, size):
    """
    apply the affine transform for points
    :param pts: the size is n*2
    :param t: the size is 2 * 3
    :param size: width, height
    :return:
    """
    # convert to homogeneous coordinates
    pts = pts.reshape(-1, 2)
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    # transform
    t = t.astype(np.float32)
    out_pts_h = np.dot(t, pts_h.T).T
    out_pts_h[:, 0] = out_pts_h[:, 0].clip(min=0, max=size[0]-1)
    out_pts_h[:, 1] = out_pts_h[:, 1].clip(min=0, max=size[1]-1)
    return out_pts_h[:, :2]


def clamp_pixel(pixel, size):
    """
    clamp the pixel
    :param pixel:
    :param size: img size
    :return:
    """
    pixel[0] = np.clip(pixel[0], 0, size[0] - 1)
    pixel[1] = np.clip(pixel[1], 0, size[1] - 1)
    return pixel[:2]


def load_rgb_image(img_path):
    input_img = cv2.imread(img_path)
    if input_img is None:
        raise ValueError("the img load error:{}".format(img_path))
    else:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    return input_img


def load_instance_image(img_path):
    return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)


def mask2poly(mask, threshold=0.5):
    """Takes a mask and returns a MultiPolygon
    Parameters
    ----------
    mask : array
        Sparse or dense array to identify polygon contours within.
    threshold : float, optional
        Threshold value used to separate points in and out of resulting
        polygons. 0.5 will partition a boolean mask, for an arbitrary value
        binary mask choose the midpoint of the low and high values.
    Output
    ------
    MultiPolygon
        Returns a MultiPolygon of all masked regions.
    """
    m = _reformat_mask(mask)[0]

    if issparse(m):
        m = np.array(m.astype('byte').todense())

    if (m != 0).sum() == 0:
        # If the plane is empty, just skip it
        return []

    # Add an empty row and column around the mask to make sure edge masks
    # are correctly determined
    expanded_dims = (m.shape[0] + 2, m.shape[1] + 2)
    expanded_mask = np.zeros(expanded_dims, dtype=float)
    expanded_mask[1:m.shape[0] + 1, 1:m.shape[1] + 1] = m

    verts = find_contours(expanded_mask.T, threshold)

    # Subtract off 1 to shift coords back to their real space,
    # but also add 0.5 to move the coordinates back to the corners,
    # so net subtract 0.5 from every coordinate
    # verts = [np.subtract(x, 0.5) for x in verts]

    return verts


def _reformat_mask(mask):
    """Convert mask to a list of sparse matrices (scipy.sparse.lil_matrix)
    Accepts a 2 or 3D array, a list of 2D arrays, or a sequence of sparse
    matrices.
    Parameters
    ----------
    mask : a 2 or 3 dimensional numpy array, a list of 2D numpy arrays, or a
        sequence of sparse matrices.  Masks are assumed to follow a (z, y, x)
        convention.  If mask is a list of 2D arrays or of sparse matrices, each
        element is assumed to correspond to the mask for a single plane (and is
        assumed to follow a (y, x) convention)
    """
    if isinstance(mask, np.ndarray):
        # user passed in a 2D or 3D np.array
        if mask.ndim == 2:
            mask = [lil_matrix(mask, dtype=mask.dtype)]
        elif mask.ndim == 3:
            new_mask = []
            for s in range(mask.shape[0]):
                new_mask.append(lil_matrix(mask[s, :, :], dtype=mask.dtype))
            mask = new_mask
        else:
            raise ValueError('numpy ndarray must be either 2 or 3 dimensions')
    elif issparse(mask):
        # user passed in a single lil_matrix
        mask = [lil_matrix(mask)]
    else:
        new_mask = []
        for plane in mask:
            new_mask.append(lil_matrix(plane, dtype=plane.dtype))
        mask = new_mask
    return mask


def poly_to_mask(poly, img_size=None):
    poly = poly.astype(np.int32)
    if img_size is None:
        img_size = (poly.max(0) + 1)[::-1]
    mask = np.zeros(img_size, dtype=np.uint8)
    return cv2.fillPoly(mask, [poly], 1)


def compute_iou_for_mask(mask1, mask2):
    overlap = mask1 & mask2
    union = mask1 | mask2
    return float(overlap.sum() + 1) / float(union.sum() + 1)


def compute_iou_for_poly(poly1, poly2, img_size=None):
    if img_size is None:
        img_size = (np.max(np.vstack((poly1.max(0), poly2.max(0)))
                          , axis=0).astype(np.int32) + 1)[::-1]

    # generate mask
    mask1 = poly_to_mask(poly1, img_size)
    mask2 = poly_to_mask(poly2, img_size)
    return compute_iou_for_mask(mask1, mask2)


def is_cover(mask1, mask2):
    mask_intersect = (mask1 * mask2).sum()
    return mask1.sum() == mask_intersect or mask2.sum() == mask_intersect