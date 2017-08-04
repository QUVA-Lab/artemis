import numpy as np

__author__ = 'peter'


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


