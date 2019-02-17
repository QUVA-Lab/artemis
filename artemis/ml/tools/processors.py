from abc import abstractmethod
import numpy as np
from six.moves import xrange

__author__ = 'peter'


class OneHotEncoding(object):

    def __init__(self, n_classes = None, form = 'bin', dtype = None):
        assert form in ('bin', 'sign')
        if dtype is None:
            dtype = np.int32 if form == 'sign' else bool
        self._n_classes = n_classes
        self._dtype = dtype
        self.form = form

    def __call__(self, data):
        if self._n_classes is None:
            self._n_classes = np.max(data)+1
        out = np.zeros((data.size, self._n_classes, ), dtype = self._dtype)
        if self.form == 'sign':
            out[:] = -1
        if data.size > 0:  # Silly numpy
            out[np.arange(data.size), data.flatten()] = 1
        out = out.reshape(data.shape+(self._n_classes, ))
        return out

    def inverse(self, data):
        return np.argmax(data, axis = 1)


class IDifferentiableFunction(object):

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def backprop_delta(self, delta_y):
        pass

    def batch_call(self, *args, **kwargs):
        return single_to_batch(self, *args, **kwargs)

    def batch_backprop_delta(self, *args, **kwargs):
        return single_to_batch(self.backprop_delta, *args, **kwargs)


class NonNormalize(IDifferentiableFunction):

    def __call__(self, x):
        return x

    def backprop_delta(self, delta_y):
        return delta_y


def single_to_batch(fcn, *batch_inputs, **batch_kwargs):
    """
    :param fcn: A function
    :param batch_inputs: A collection of batch-form (n_samples, input_dims_i) inputs
    :return: batch_outputs, an (n_samples, output_dims) array
    """
    n_samples = len(batch_inputs[0])
    assert all(len(b) == n_samples for b in batch_inputs)
    first_out = fcn(*[b[0] for b in batch_inputs], **{k: b[0] for k, b in batch_kwargs.items()})
    if n_samples==1:
        return first_out[None]
    out = np.empty((n_samples, )+first_out.shape)
    out[0] = n_samples
    for i in xrange(1, n_samples):
        out[i] = fcn(*[b[i] for b in batch_inputs], **{k: b[i] for k, b in batch_kwargs.items()})
    return out
