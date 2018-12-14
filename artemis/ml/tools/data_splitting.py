import numpy as np
from six.moves import xrange

from artemis.general.nested_structures import NestedType
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


def join_arrays_and_get_rebuild_func(arrays, axis = 0):
    """
    Given a nested structure of arrays, join them into a single array by flattening dimensions from axis on
    concatenating them.  Return the joined array and a function which can take the joined array and reproduce the
    original structure.

    :param arrays: A possibly nested structure containing arrays which you want to join into a single array.
    :param axis: Axis after which to flatten and join all arrays.  The resulting array will be (dim+1) dimensional.
    :return ndarray, Callable[[ndarray], [Any]]: The joined array, and the function which can be called to reconstruct
        the structure from the joined array.
    """
    nested_type = NestedType.from_data(arrays)
    data_list = nested_type.get_leaves(arrays)
    split_shapes = [x_.shape for x_ in data_list]
    pre_join_shapes = [list(x_.shape[:axis]) + [np.prod(list(x_.shape[axis:]), dtype=int)] for x_ in data_list]
    split_axis_ixs = np.cumsum([0]+[s_[-1] for s_ in pre_join_shapes], axis=0)
    joined_arr = np.concatenate(list(x_.reshape(s_) for x_, s_ in izip_equal(data_list, pre_join_shapes)), axis=axis)

    def rebuild_function(joined_array, share_data = True):
        if share_data:
            x_split = [joined_array[..., start:end].reshape(shape) for (start, end, shape) in izip_equal(split_axis_ixs[:-1], split_axis_ixs[1:], split_shapes)]
        else:  # Note: this will raise an Error if the self.dim != 0, because the data is no longer contigious in memory.
            x_split = [joined_array[..., start:end].copy().reshape(shape) for (start, end, shape) in izip_equal(split_axis_ixs[:-1], split_axis_ixs[1:], split_shapes)]
        x_reassembled = nested_type.expand_from_leaves(x_split, check_types=False)
        return x_reassembled

    return joined_arr, rebuild_function
