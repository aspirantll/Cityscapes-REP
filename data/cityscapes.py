import os
import numpy as np
from torch.utils.data import Dataset

from utils.tranform import TransInfo
from .dataset import DatasetBuilder
from utils.image import load_rgb_image, load_instance_image

label_names = ['background', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
label_ids = [-10, 24, 25, 26, 27, 28, 31, 32, 33]

num_cls = len(label_names)
IMAGE_EXTENSIONS = ['.jpg', '.png']


def is_image(filename):
    return any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_label(filename):
    return "gtFine_instanceIds" in filename


def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    m = label_field.max()
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(int(m))
        label_field = label_field.astype(new_type)
        m = m.astype(new_type)  # Ensures m is an integer
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    required_type = np.min_scalar_type(offset + len(labels0))
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        label_field = label_field.astype(required_type)
    new_labels0 = np.arange(offset, offset + len(labels0))
    if np.all(labels0 == new_labels0):
        return label_field, labels, labels
    forward_map = np.zeros(int(m + 1), dtype=label_field.dtype)
    forward_map[labels0] = new_labels0
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = np.zeros(offset - 1 + len(labels), dtype=label_field.dtype)
    inverse_map[(offset - 1):] = labels
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def decode_instance(pic):
    instance_map = np.zeros(
        (pic.shape[0], pic.shape[1]), dtype=np.uint8)

    # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
    class_ids = np.empty((0, 5), dtype=np.float32)

    for i, c in enumerate(label_ids):
        mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids + np.amax(instance_map)

            pos_y, pos_x = mask.nonzero()
            class_ids = np.append(class_ids,
                                  np.array([[pos_x.min(), pos_y.min(),
                                            pos_x.max(), pos_y.max(), i+1]], dtype=np.float32), axis=0)
    return class_ids, instance_map


class CityscapesDataset(Dataset):

    def __init__(self, root, transforms=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f
                          in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if is_label(f)]
        self.filenamesGt.sort()

        self._transforms = transforms  # ADDED THIS

        print("dataset size: {}".format(len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_path = filename
        input_img = load_rgb_image(img_path)

        filenameGt = self.filenamesGt[index]
        instance_img = load_instance_image(filenameGt)
        label = decode_instance(instance_img)
        img_size = input_img.shape[1::-1]

        if self._transforms is not None:
            return self._transforms(input_img, label, img_path=img_path, img_size=img_size)

        return input_img, label, TransInfo(img_path, img_size, 1,0, img_size)

    def __len__(self):
        return len(self.filenames)


class CityscapesDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, phase):
        super().__init__(data_dir, phase)

    def get_dataset(self, **kwargs):
        return CityscapesDataset(self._data_dir, subset=self._phase, **kwargs)
