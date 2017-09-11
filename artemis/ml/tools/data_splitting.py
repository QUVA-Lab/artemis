import numpy as np
from six.moves import xrange

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
