from abc import abstractmethod
import numpy as np
from artemis.general.mymath import recent_moving_average
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


class RunningAverage(object):

    def __init__(self):
        self._n_samples_seen = 0
        self._average = 0

    def __call__(self, data):
        self._n_samples_seen+=1
        frac = 1./self._n_samples_seen
        self._average = (1-frac)*self._average + frac*data
        return self._average

    @classmethod
    def batch(cls, x):
        return np.cumsum(x, axis=0)/np.arange(1, len(x)+1).astype(np.float)[(slice(None), )+(None, )*(x.ndim-1)]


class RecentRunningAverage(object):

    def __init__(self):
        self._n_samples_seen = 0
        self._average = 0

    def __call__(self, data):
        self._n_samples_seen+=1
        frac = 1/self._n_samples_seen**.5
        self._average = (1-frac)*self._average + frac*data
        return self._average

    @classmethod
    def batch(cls, x):
        # return recent_moving_average(x, axis=0)  # Works only for python 2.X, with weave
        ra = cls()
        return np.array([ra(x_) for x_ in x])


class RunningAverageWithBurnin(object):

    def __init__(self, burn_in_steps):
        self._burn_in_step_remaining = burn_in_steps
        self.averager = RunningAverage()

    def __call__(self, x):

        if self._burn_in_step_remaining > 0:
            self._burn_in_step_remaining-=1
            return x
        else:
            return self.averager(x)


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


class RunningCenter(IDifferentiableFunction):
    """
    Keep an exponentially decaying running mean, subtract this from the value.
    """
    def __init__(self, half_life):
        self.decay_constant = np.exp(-np.log(2)/half_life)
        self.one_minus_decay_constant = 1-self.decay_constant
        self.running_mean = None

    def __call__(self, x):
        if self.running_mean is None:
            self.running_mean = np.zeros_like(x)
        self.running_mean[:] = self.decay_constant * self.running_mean + self.one_minus_decay_constant * x
        return x - self.running_mean

    def backprop_delta(self, delta_y):
        return self.decay_constant * delta_y



class RunningNormalize(IDifferentiableFunction):

    def __init__(self, half_life, eps = 1e-7, initial_std=1):
        self.decay_constant = np.exp(-np.log(2)/half_life)
        self.one_minus_decay_constant = 1-self.decay_constant
        self.running_mean = None
        self.eps = eps
        self.initial_std = initial_std

    def __call__(self, x):
        if self.running_mean is None:
            self.running_mean = np.zeros_like(x)
            self.running_mean_sq = np.zeros_like(x) + self.initial_std**2
        self.running_mean[:] = self.decay_constant * self.running_mean + self.one_minus_decay_constant * x
        self.running_mean_sq[:] = self.decay_constant * self.running_mean_sq + self.one_minus_decay_constant * x**2
        std = np.sqrt(self.running_mean_sq - self.running_mean**2)
        return (x - self.running_mean) / (std+self.eps)

    def backprop_delta(self, delta_y):
        """
        Ok, we're not doing this right at all, but lets just ignore the contribution of the current
        sample to the mean/std.  This makes the gradient waaaaaay simpler.  If you want to see the real thing, put

        (x-(a*u+(1-a)*x))/sqrt((a*s+(1-a)*x^2 - (a*u+(1-a)*x)^2))
        into http://www.derivative-calculator.net/
        (a stands for lambda here)

        :param delta_y: The derivative of the cost wrt the output of this normalizer
        :return: delta_x: The derivative of the cost wrt the input of this normalizer
        """
        std = np.sqrt(self.running_mean_sq - self.running_mean**2)
        return delta_y/std


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
