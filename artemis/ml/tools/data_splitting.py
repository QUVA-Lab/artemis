import numpy as np
from six.moves import xrange

from artemis.general.nested_structures import get_meta_object, fill_meta_object, get_leaf_values
from artemis.general.should_be_builtins import izip_equal

__author__ = 'peter'


def split_data_by_label(data, labels, frac_training = 0.5):
    """
    Split the data so that each label gets approximately the correct proportions between the training and test sets
    :param data: An (n_samples, ...) array of data
    :param labels: An (n_samples, ) array of labels
    :param frac_training: The fraction of data to put in the training set vs the test.
    :return: (x_tr, x_ts, y_tr, y_ts)
    """
    # TODO: Write test - it's important that this be correct

    assert len(data)==len(labels)
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    unique_labels, inverse_ixs = np.unique(labels, return_inverse = True)
    ixs = np.arange(len(data))
    label_indices = [ixs[inverse_ixs==l] for l in xrange(len(unique_labels))]
    cutoffs = [int(np.round(frac_training*len(ixs))) for ixs in label_indices]
    training_indices = np.sort(np.concatenate([ixs[:c] for ixs, c in zip(label_indices, cutoffs)]))
    test_indices = np.sort(np.concatenate([ixs[c:] for ixs, c in zip(label_indices, cutoffs)]))
    return data[training_indices], labels[training_indices], data[test_indices], labels[test_indices]


class ArrayStructRebuilder(object):
    """
    A parameterized function which rebuilds a data structure given a flattened array containing the values.
    Suggest using it through join_arrays_and_get_rebuild_func
    """

    def __init__(self, split_shapes, meta_object):
        """
        :param Sequence[Tuple[int]] split_shapes: The shapes
        :param Any meta_object: A nested object defining the structure in which to rebuild (see get_meta_object)
        """
        self.split_shapes = split_shapes
        self.meta_object = meta_object

    def __call__(self, joined_array, share_data = True, transform_func = None, check_types=True):
        axis = joined_array.ndim-1
        pre_join_shapes = [list(s[:axis]) + [np.prod(list(s[axis:]), dtype=int)] for s in self.split_shapes]
        split_axis_ixs = np.cumsum([0]+[s_[-1] for s_ in pre_join_shapes], axis=0)
        if share_data:
            x_split = [joined_array[..., start:end].reshape(shape) for (start, end, shape) in izip_equal(split_axis_ixs[:-1], split_axis_ixs[1:], self.split_shapes)]
        else:  # Note: this will raise an Error if the self.dim != 0, because the data is no longer contigious in memory.
            x_split = [joined_array[..., start:end].copy().reshape(shape) for (start, end, shape) in izip_equal(split_axis_ixs[:-1], split_axis_ixs[1:], self.split_shapes)]
        if transform_func is not None:
            x_split = [transform_func(xs) for xs in x_split]
        x_reassembled = fill_meta_object(self.meta_object, (x for x in x_split), check_types=check_types)
        return x_reassembled


def join_arrays_and_get_rebuild_func(arrays, axis = 0, transform_func = None):
    """
    Given a nested structure of arrays, join them into a single array by flattening dimensions from axis on
    concatenating them.  Return the joined array and a function which can take the joined array and reproduce the
    original structure.

    :param arrays: A possibly nested structure containing arrays which you want to join into a single array.
    :param axis: Axis after which to flatten and join all arrays.  The resulting array will be (dim+1) dimensional.
    :param transform_func: Optionally, a function which you apply to every element in the nested struture of arrays first.
    :return ndarray, ArrayStructRebuilder: The joined array, and the function which can be called to reconstruct
        the structure from the joined array.
    """
    meta_object = get_meta_object(arrays)
    data_list = get_leaf_values(arrays)
    if transform_func is not None:
        data_list = [transform_func(d) for d in data_list]
    split_shapes = [x_.shape for x_ in data_list]
    pre_join_shapes = [list(s[:axis]) + [np.prod(list(s[axis:]), dtype=int)] for s in split_shapes]
    joined_arr = np.concatenate(list(x_.reshape(s_) for x_, s_ in izip_equal(data_list, pre_join_shapes)), axis=axis)
    rebuild_function = ArrayStructRebuilder(split_shapes=split_shapes, meta_object=meta_object)
    return joined_arr, rebuild_function
