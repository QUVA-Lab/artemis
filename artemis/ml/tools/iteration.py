from collections import namedtuple
import itertools
from artemis.general.should_be_builtins import bad_value
import numpy as np
import time

__author__ = 'peter'


def minibatch_index_generator(n_samples, minibatch_size, n_epochs = 1, final_treatment = 'stop', slice_when_possible = True):
    """
    Generates the indices for minibatch-iteration.

    :param n_samples: Number of samples in the data you want to iterate through
    :param minibatch_size: Number of samples in the minibatch
    :param n_epochs: Number of epochs to iterate for
    :param final_treatment: How to terminate.  Options are:
        'stop': Stop when you can no longer get a complete minibatch
        'truncate': Produce a runt-minibatch at the end.
    :param slice_when_possible: Return slices, instead of indices, as long as the indexing does not wrap around.  This
        can be more efficient, since it avoids array copying, but you have to be careful not to modify your source array.
    :yield: IIndices that you can use to slice arrays for minibatch iteration.
    """

    true_minibatch_size = n_samples if minibatch_size == 'full' else \
        minibatch_size if isinstance(minibatch_size, int) else \
        bad_value(minibatch_size)
    remaining_samples = int(n_epochs * n_samples) if not np.isinf(n_epochs) else np.inf
    base_indices = np.arange(minibatch_size)
    standard_indices = (lambda: slice(i, i+minibatch_size)) if slice_when_possible else (lambda: base_indices+i)
    i = 0
    while True:
        next_i = i + true_minibatch_size
        if remaining_samples < minibatch_size:  # Final minibatch case
            if final_treatment == 'stop':
                break
            elif final_treatment == 'truncate':
                yield np.arange(i, i+remaining_samples) % n_samples
                break
            else:
                raise Exception('Unknown final treatment: %s' % final_treatment)
        elif next_i < n_samples:  # Standard case
            segment = standard_indices()
        else:  # Wraparound case
            segment = np.arange(i, next_i) % n_samples
            next_i = next_i % n_samples

        yield segment
        i = next_i
        remaining_samples -= minibatch_size


def checkpoint_minibatch_index_generator(n_samples, checkpoints, slice_when_possible = True):
    """
    Generates minibatch indices that fill the space between checkpoints.  This is useful, for instance, when you want to test
    at certain points, and your predictor internally iterates through the minibatch sample by sample between those.
    :param n_samples: Number of samples in the data you want to iterate through
    :param checkpoints: An array of indices at which the "checkpoints" happen.  Minibatches will be sliced up by these
        indices.
    :param slice_when_possible: Return slices, instead of indices, as long as the indexing does not wrap around.  This
        can be more efficient, since it avoids array copying, but you have to be careful not to modify your source array.
    :yield: Indices that you can use to slice arrays for minibatch iteration.
    """
    checkpoints = np.array(checkpoints, dtype = int)
    assert len(checkpoints) > 1 and checkpoints[0] >= 0 and np.all(np.diff(checkpoints) > 0)
    checkpoint_divs = zip(checkpoints[:-1], checkpoints[1:])
    if checkpoints[0] > 0:
        checkpoint_divs.insert(0, (0, checkpoints[0]))
    for start, stop in checkpoint_divs:
        if start/n_samples == stop/n_samples:  # No wrap
            if slice_when_possible:
                yield slice(start % n_samples, stop % n_samples)
            else:
                yield np.arange(start % n_samples, stop % n_samples)
        else:
            yield np.arange(start, stop) % n_samples


def zip_minibatch_iterate(arrays, minibatch_size, n_epochs=1):
    """
    Yields minibatches from each array in arrays in sequence.
    :param arrays: A collection of arrays, all of which must have the same shape[0]
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :yield: len(arrays) arrays, each of shape: (minibatch_size, )+arr.shape[1:]
    """
    assert isinstance(arrays, (list, tuple)), 'You need to provide an array or collection of arrays.'
    assert len(arrays)>0, 'Need at least one array'
    total_size = arrays[0].shape[0]
    assert all(a.shape[0] == total_size for a in arrays), 'All arrays must have the same length!  Lengths are: %s' % ([len(arr) for arr in arrays])
    end = total_size*n_epochs
    ixs = np.arange(minibatch_size)
    while ixs[0] <= end:
        yield tuple(a[ixs % total_size] for a in arrays)
        ixs+=minibatch_size


IterationInfo = namedtuple('IterationInfo', ['iteration', 'epoch', 'sample', 'time', 'test_now', 'done'])


def iteration_info(n_samples, minibatch_size, test_epochs = None, n_epochs = None):
    """
    Create an iterator that keeps track of the state of minibatch iteration, and simplifies the scheduling of tests.
    You can izip this iterator with one that returns your data.

    :param n_samples: Number of samples in the dataset.
    :param minibatch_size: Size of minibatches
    :param test_epochs: Epochs on which you'd like to run tests.  You can also enter
        'every', which will test once-per-epoch,
        'always', which will test on every iteration
        'never', which will never test.
    :yield: IterationInfo objects which contain info about the state of iteration.
    """
    # next_text_point = 0 if test_epochs is not None and len(test_epochs)>0 else None
    start_time = time.time()
    n_samples = float(n_samples)
    if minibatch_size=='full':
        minibatch_size = n_samples
    if isinstance(test_epochs, str):
        assert test_epochs in ('always', 'never', 'every')
    elif isinstance(test_epochs, tuple):
        assert len(test_epochs)==2, "If you pass in a tuple for test epochs, it should be in the form ('every', period).  Get {}".format(test_epochs)
        name, period = test_epochs
        assert period > 0, 'Period must be a positive number, not {}'.format(period)
        assert name == 'every'
    elif n_epochs is None and isinstance(test_epochs, (list, tuple, np.ndarray)):
        n_epochs = test_epochs[-1]

    last_epoch = -float('inf')
    for i in itertools.count(0):
        epoch = i*minibatch_size/n_samples
        test_now = (
                True if test_epochs=='always' else
                False if test_epochs=='never' else
                np.floor(epoch)>np.floor(last_epoch) if test_epochs == 'every' else
                bad_value(test_epochs)
            ) if isinstance(test_epochs, basestring) else \
            np.floor(epoch/period) > np.floor(last_epoch/period) if isinstance(test_epochs, tuple) else \
            False if test_epochs is None else \
            np.searchsorted(test_epochs, epoch, side='right') > np.searchsorted(test_epochs, last_epoch, side='right')
        info = IterationInfo(
            iteration = i,
            epoch = epoch,
            sample = i*minibatch_size,
            time = time.time()-start_time,
            test_now = test_now,
            done = epoch >= n_epochs
            )
        yield info
        last_epoch = epoch


def zip_minibatch_iterate_info(arrays, minibatch_size, n_epochs=None, test_epochs = None):
    """
    Iterate through minibatches of arrays and yield info about the state of iteration though training.

    :param arrays:
    :param arrays: A collection of arrays, all of which must have the same shape[0]
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :param test_epochs: A list of epochs to test at.
    :return: (array_minibatches, info)
        arrays is a tuple of minibatches from arrays
        info is an IterationInfo object returning information about the state of iteration.
    """
    if n_epochs is None:
        assert isinstance(test_epochs, (list, tuple, np.ndarray)), "If you don't specify n_epochs, you need to specify an array of test epochs."
        n_epochs = test_epochs[-1]
    for arrays, info in itertools.izip(
            zip_minibatch_iterate(arrays, minibatch_size=minibatch_size, n_epochs=n_epochs),
            iteration_info(n_samples=arrays[0].shape[0], minibatch_size=minibatch_size, test_epochs=test_epochs, n_epochs=n_epochs)
            ):
        yield arrays, info


def minibatch_iterate(data, minibatch_size, n_epochs=1):
    """
    Yields minibatches in sequence.
    :param data: A (n_samples, ...) data array
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :yield: (minibatch_size, ...) data arrays.
    """
    if minibatch_size == 'full':
        minibatch_size = len(data)
    end = len(data)*n_epochs
    ixs = np.arange(minibatch_size)
    while ixs[0] < end:
        yield data[ixs % len(data)]
        ixs+=minibatch_size


def minibatch_iterate_info(data, minibatch_size, n_epochs, test_epochs = None):
    """
    Iterate through data and yield info about the iteration
    :param data: An (n_samples, ...) array of data
    :param minibatch_size: Size of the minibatches
    :param n_epochs: Number of epochs to iterate for
    :param test_epochs: A list of epochs to test at.
    :return:(minibatch, info)
        minibatch is a minibatch of data (minibatch_size, ...)
        info is an IterationInfo object returning information about the state of iteration.
    """
    for arrays, info in itertools.izip(
            minibatch_iterate(data, minibatch_size=minibatch_size, n_epochs=n_epochs),
            iteration_info(n_samples=data.shape[0], minibatch_size=minibatch_size, test_epochs=test_epochs)
            ):
        yield arrays, info
