from numpy.random.mtrand import RandomState
import numpy as np
__author__ = 'peter'


def get_rng(initializer):
    """
    Creates a random number generator from either a seed, the time, or another random number generator.
    This is useful when you need to initialize random states on many levels, and you want the choice between
    specifying a seed directly or passing in an RNG from somewhere else.

    :param initializer: Can be:
        None, in which case the RNG is seeded on the time
        an int, in which case this is used as the seed
        a numpy RandomState object, in which case it's just passed through.
    :return: A numpy RandomState object.
    """
    if initializer is None or isinstance(initializer, int):
        return np.random.RandomState(initializer)
    elif isinstance(initializer, RandomState):
        return initializer


def array_info(arr):
    """
    :param arr: Return info about an array
    :return:
    """
    return '<{dtype} array of shape {shape}: mean: {mean:.3g}, std: {std:.3g}, min: {min:.3g}, max: {max:.3g} at {addr:s}>'.format(
        dtype = arr.dtype,
        shape = arr.shape,
        mean = arr.mean(),
        std = arr.std(),
        min = arr.min(),
        max = arr.max(),
        addr = hex(id(arr))
        )


def argtopk(x, k, axis=-1):
    return np.take(np.argpartition(-x, axis=axis, kth=k), np.arange(k), axis=axis)


def fast_array(objects):
    """
    np.ndarray does some checking on the list of objects first.  This just trusts that they're all similat
    :param objects:
    :return:
    """
    arr = np.empty((len(objects), )+objects[0].shape, dtype=objects[0].dtype)
    for i, ob in enumerate(objects):
        arr[i] = ob
    return arr