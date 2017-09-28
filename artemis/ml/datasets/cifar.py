import os
import pickle
import numpy as np
from artemis.fileman.file_getter import get_file, get_archive
from artemis.ml.datasets.datasets import DataSet, DataCollection
from six.moves import xrange, input

__author__ = 'peter'


def get_cifar_100_dataset(n_training_samples=None, n_test_samples=None, whiten_individual_images=False, fine_labels=True):
    '''
    :param n_training_samples: Number of training samples, or None to leave it at 50000
    :param n_test_samples: Number of test samples, or None to leave it at 10000
    :param whiten_individual_images: True to normalize each individual image, and turn them from uint8 to double
    :param fine_labels: True to use the fine categorization (100 classes) or False to use the coarse categorization (20 classes)
    :return: The CIFAR-100 dataset, which consists of 50000 training and 10000 test images.
        Images are 32x32 uint8 (or double) RGB images (n_samples, 3, 32, 32) of 100 (or 20) categories of objects.
        Targets are integer labels in the range [0, 100] (or [0,20])

    '''

    directory = get_archive(relative_path='data/cifar-100', url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
    with open(os.path.join(directory, "cifar-100-python", "train"), 'rb') as fo:
        dict = pickle.load(fo)
    x_tr = dict["data"].reshape(-1, 3, 32, 32)
    y_tr = dict["fine_labels"] if fine_labels else dict["coarse_labels"]
    y_tr = np.array(y_tr)

    with open(os.path.join(directory, "cifar-100-python", "test"), 'rb') as fo:
        dict = pickle.load(fo)
    x_ts = dict["data"].reshape(-1, 3, 32, 32)
    y_ts = dict["fine_labels"] if fine_labels else dict["coarse_labels"]
    y_ts = np.array(y_ts)

    if whiten_individual_images:
        mean_tr = x_tr.mean(axis=(1,2,3), keepdims=True)
        std_tr = x_tr.std(axis=(1,2,3), keepdims=True)
        x_tr = (x_tr - mean_tr)/std_tr
        mean_ts = x_ts.mean(axis=(1,2,3), keepdims=True)
        std_ts = x_ts.std(axis=(1,2,3), keepdims=True)
        x_ts = (x_ts - mean_ts)/std_ts

    if n_training_samples is not None:
        x_tr = x_tr[:n_training_samples]
        y_tr = y_tr[:n_training_samples]
    if n_test_samples is not None:
        x_ts = x_ts[:n_test_samples]
        y_ts = y_ts[:n_test_samples]


    return DataSet(training_set=DataCollection(x_tr, y_tr), test_set=DataCollection(x_ts, y_ts), name='CIFAR-100-%s'%("fine" if fine_labels else "coarse"))






def get_cifar_10_dataset(n_training_samples = None, n_test_samples = None, normalize_inputs = False):
    """
    :param n_training_samples: Number of training samples, or None to leave it at 50000
    :param n_test_samples: Number of test samples, or None to leave it at 10000
    :param normalize_inputs: True to normalize inputs, and turn them from uint8 to double
    :param swap_axes: True to arrange images as (n_samples, n_colors, n_rows, n_cols) instead of (n_samples, n_rows, n_cols, n_colors)

    :return: The CIFAR-10 dataset, which consists of 50000 training and 10000 test images.
        Images are 32x32 uint8 RGB images (n_samples, 3, 32, 32) of 10 categories of objects.
        Targets are integer labels in the range [0, 9]
    """
    # TODO: Make method for downloading/unpacking data (from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # We have this for single files already, but in this case the gz contains a folder with the files in it.

    directory = get_archive(relative_path='data/cifar-10', url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')

    n_batches_to_read = 5 if n_training_samples is None else int(np.ceil(n_training_samples/10000.))

    file_paths = [get_file(os.path.join(directory, 'cifar-10-batches-py', 'data_batch_%s' % (i, ))) for i in xrange(1, n_batches_to_read+1)] \
        + [get_file(os.path.join(directory, 'cifar-10-batches-py', 'test_batch'))]

    data = []
    for file_path in file_paths:
        with open(file_path) as f:
            batch_data = pickle.load(f)
            data.append(batch_data)

    x_tr = np.concatenate([d['data'] for d in data[:-1]], axis = 0).reshape(-1, 3, 32, 32)
    y_tr = np.concatenate([d['labels'] for d in data[:-1]], axis = 0)
    x_ts = data[-1]['data'].reshape(-1, 3, 32, 32)
    y_ts = np.array(data[-1]['labels'])

    if normalize_inputs:
        mean = x_tr.mean(axis=0, keepdims=True)
        std = x_tr.std(axis=0, keepdims=True)
        x_tr = (x_tr - mean)/std
        x_ts = (x_ts - mean)/std

    if n_training_samples is not None:
        x_tr = x_tr[:n_training_samples]
        y_tr = y_tr[:n_training_samples]
    if n_test_samples is not None:
        x_ts = x_ts[:n_test_samples]
        y_ts = y_ts[:n_test_samples]

    return DataSet(training_set=DataCollection(x_tr, y_tr), test_set=DataCollection(x_ts, y_ts), name = 'CIFAR-10')


def get_cifar_10_labels():
    return [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']


if __name__ == '__main__':

    from artemis.plotting.easy_plotting import ezplot

    dataset = get_cifar_10_dataset()
    dataset = get_cifar_100_dataset()
    n_samples = 100

    ezplot({
        'sampled training images': np.rollaxis(dataset.training_set.input[:n_samples], 1, 4),# np.swapaxes(dataset.training_set.input[:n_samples], 1, 3).reshape(10, 10, 32, 32, 3),
        'sampled training labels': dataset.training_set.target[:n_samples].reshape(10, 10)
        }, cmap = 'jet')
    input("Enter to exit")
