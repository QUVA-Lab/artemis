from collections import namedtuple
import itertools

from six import string_types

from artemis.general.numpy_helpers import fast_array
from artemis.general.should_be_builtins import bad_value, izip_equal
import numpy as np
from six.moves import xrange, zip
import time

__author__ = 'peter'


SINGLE_MINIBATCH_SIZE = 'single'
FULL_MINIBATCH_SIZE = 'full'


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

    if minibatch_size==SINGLE_MINIBATCH_SIZE:
        for i in (xrange(n_samples*n_epochs) if not np.isinf(n_epochs) else itertools.count(0)):
            yield i % n_samples
        return

    true_minibatch_size = n_samples if minibatch_size == FULL_MINIBATCH_SIZE else \
        minibatch_size if isinstance(minibatch_size, int) else \
        bad_value(minibatch_size)
    remaining_samples = int(n_epochs * n_samples) if not np.isinf(n_epochs) else np.inf

    base_indices = np.arange(minibatch_size)
    standard_indices = (lambda: slice(i, i+minibatch_size)) if slice_when_possible else (lambda: base_indices+i)
    i = 0
    while remaining_samples>0:
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
    checkpoint_divs = list(zip(checkpoints[:-1], checkpoints[1:]))
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


def zip_minibatch_iterate(arrays, minibatch_size, n_epochs=1, final_treatment = 'truncate'):
    """
    Yields minibatches from each array in arrays in sequence.
    :param arrays: A collection of arrays, all of which must have the same shape[0]
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for (or 'inf' for an infinite iterator)
    :yield: len(arrays) arrays, each of shape: (minibatch_size, )+arr.shape[1:]
    """
    assert isinstance(arrays, (list, tuple)), 'You need to provide an array or collection of arrays.'
    assert len(arrays)>0, 'Need at least one array'
    total_size = arrays[0].shape[0]
    if minibatch_size==FULL_MINIBATCH_SIZE:
        minibatch_size=total_size
    if n_epochs=='inf':
        n_epochs=float('inf')
    assert isinstance(n_epochs, (int, float))
    assert all(a.shape[0] == total_size for a in arrays), 'All arrays must have the same length!  Lengths are: %s' % ([len(arr) for arr in arrays])

    for ixs in minibatch_index_generator(n_samples=total_size, minibatch_size=minibatch_size, n_epochs=n_epochs, final_treatment=final_treatment):
        yield tuple(a[ixs] for a in arrays)


IterationInfo = namedtuple('IterationInfo', ['iteration', 'epoch', 'sample', 'time', 'test_now', 'done'])


def iteration_info(n_samples, minibatch_size, test_epochs = None, n_epochs = 5):
    """
    Create an iterator that keeps track of the state of minibatch iteration, and simplifies the scheduling of tests.
    You can izip this iterator with one that returns your data.

    :param n_samples: Number of samples in the dataset.
    :param minibatch_size: Size of minibatches
    :param test_epochs: Epochs on which you'd like to run tests.  You can also enter
        'every', which will test once-per-epoch,
        'always', which will test on every iteration
        'never', which will never test.
        ('every', 0.2), which will test at every 0.2 epochs (for example)
    :yield: IterationInfo objects which contain info about the state of iteration.
    """
    # next_text_point = 0 if test_epochs is not None and len(test_epochs)>0 else None
    start_time = time.time()
    n_samples = float(n_samples)
    if minibatch_size==FULL_MINIBATCH_SIZE:
        minibatch_size = n_samples
    elif minibatch_size == SINGLE_MINIBATCH_SIZE:
        minibatch_size = 1
    elif not isinstance(minibatch_size, int):
        raise Exception('Unexpected value for minibatch_size: {}'.format(minibatch_size))
    if isinstance(test_epochs, str):
        assert test_epochs in ('always', 'never', 'every'), "test_epochs={} is not valid".format(test_epochs)
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
            ) if isinstance(test_epochs, string_types) else \
            np.floor(epoch/period) > np.floor(last_epoch/period) if isinstance(test_epochs, tuple) else \
            False if test_epochs is None else \
            np.searchsorted(test_epochs, epoch, side='right') > np.searchsorted(test_epochs, last_epoch, side='right')
        info = IterationInfo(
            iteration = i,
            epoch = epoch,
            sample = i*minibatch_size,
            time = time.time()-start_time,
            test_now = test_now,
            done = epoch >= n_epochs if n_epochs is not None else False
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
    for arrays, info in zip(
            zip_minibatch_iterate(arrays, minibatch_size=minibatch_size, n_epochs='inf'),
            iteration_info(n_samples=arrays[0].shape[0], minibatch_size=minibatch_size, test_epochs=test_epochs, n_epochs=n_epochs)
            ):
        yield arrays, info
        if info.done:
            break


def minibatch_index_info_generator(n_samples, minibatch_size, n_epochs, test_epochs = None, slice_when_possible=False):
    for ixs, info in zip(
            minibatch_index_generator(n_samples=n_samples, minibatch_size=minibatch_size, n_epochs=n_epochs, slice_when_possible=slice_when_possible),
            iteration_info(n_samples=n_samples, minibatch_size=minibatch_size, test_epochs=test_epochs, n_epochs=n_epochs)
            ):
        yield ixs, info


def minibatch_iterate(data, minibatch_size, n_epochs=1):
    """
    Yields minibatches in sequence.
    :param data: A (n_samples, ...) data array
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :yield: (minibatch_size, ...) data arrays.
    """
    if minibatch_size == FULL_MINIBATCH_SIZE:
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
    for arrays, info in zip(
            minibatch_iterate(data, minibatch_size=minibatch_size, n_epochs=n_epochs),
            iteration_info(n_samples=data.shape[0], minibatch_size=minibatch_size, test_epochs=test_epochs)
            ):
        yield arrays, info


def minibatch_process(f, minibatch_size, mb_args = (), mb_kwargs = {}, fixed_kwargs={}):
    """
    Process inputs through a function f in minibatches, and stitch together the outputs.

    :param f: A function of the form y=f(x1, x2, ...)  where x1.shape[0] == x2.shape[0] == y.shape[0].
        We assume that outputs of f are equivariant to permuations along the 0th axis of inputs of f.
        i.e. each row is processed independently.
    :param minibatch_size: The size of the minibatches in which to feed the date through y
    :param mb_args: Arguments that will be fed through in minibatches
    :param mb_kwargs: Keyword-Arguments that will be fed through in minibatches
    :param fixed_kwargs: Keyword arguments that will remain fixed.
    :return: The output y
    """
    all_mb_args = list(mb_args) + list(mb_kwargs.values())
    assert len(all_mb_args)>0, 'Need some input.'
    assert callable(f), 'f must be a function'
    n_samples = len(mb_args[0])
    assert all(len(arg) == n_samples for arg in all_mb_args)
    mb_kwarg_list = mb_kwargs.items()
    fixed_kwarg_list = list(fixed_kwargs.items())
    index_generator = minibatch_index_generator(n_samples = n_samples, n_epochs=1, minibatch_size=minibatch_size, final_treatment='truncate')
    ix = next(index_generator)
    first_output = f(*(a[ix] for a in mb_args), **dict([(k, v[ix]) for k, v in mb_kwarg_list]+fixed_kwarg_list))
    if first_output is None:
        for ix in index_generator:
            f(*(a[ix] for a in mb_args), **dict([(k, v[ix]) for k, v in mb_kwarg_list]+fixed_kwarg_list))
    else:
        output_shape = first_output.shape if minibatch_size==SINGLE_MINIBATCH_SIZE else first_output.shape[1:]
        results = np.empty((n_samples, )+output_shape, dtype=first_output.dtype)
        results[:len(first_output)] = first_output
        for ix in index_generator:
            results[ix] = f(*(a[ix] for a in mb_args), **dict([(k, v[ix]) for k, v in mb_kwarg_list]+fixed_kwarg_list))
        return results


def generator_pool(generator_generator):
    for generator in generator_generator:
        yield generator


def batchify_generator(generator_generator, batch_size, receive_input=False, out_format ='array'):
    """
    Best understood by example:

    Suppose we want to get batches of frames from video data.  Where the batch[t][i] is the frame after batch[t-1][i].

    e.g. Suppose we have 7 videos.  In the following, each column represents a batch of data, and rows represent the
    index within a batch.

        -------vid-1---------|-------vid-5-------|--vid-7--
        -----------vid-2-----------|--------vid-6---------|
        -----vid-3-------|----------vid-4------------------

    generator_genererator yields 7 generators, corresponding to each of the movies.
    Each of those generators is a frame-generator, which produces the frames in a given video.
    Here, we generate frames from each movie, and start a new movies whenever an old one stops, until there are no
    new movies to start.

    :param generator_generator: An generator which generates generators
    :param batch_size: The size if the batch you want to yield
    :param receive_input: Expect a "send" to this generatoer AFTER it yields.  (see Python coroutines)
    :param out_format: 'array' or 'tuple_of_arrays' currently supported.
    :yield: An array consisting of batch_size of the outputs of the subgenerator, batched together.
    """
    assert receive_input in (False, 'post'), 'pre-receive not yet implemented'

    total = batch_size

    assert out_format in ('array', 'tuple_of_arrays')
    generators = [next(generator_generator) for _ in range(batch_size)]
    while True:
        items = []
        for i in range(batch_size):
            while True:
                try:
                    items.append(next(generators[i]))
                    break
                except StopIteration:
                    total+=1
                    generators[i] = next(generator_generator)  # This will rais StopIteration when we're out of generators

        if out_format=='array':
            output= np.array(items)
        else:
            output = tuple(np.array(x) for x in zip(*items))

        if not receive_input:
            yield output
        elif receive_input=='post':
            received_signal = (yield output)  # Assume in_format=='array'
            for gen, sig in izip_equal(generators, received_signal):
                gen.send(sig)
            yield None
        else:
            raise Exception(receive_input)