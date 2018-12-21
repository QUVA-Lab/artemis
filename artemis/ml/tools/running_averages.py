import numpy as np

from artemis.general.global_rates import is_elapsed
from artemis.general.global_vars import get_global, has_global, set_global
from artemis.general.mymath import recent_moving_average
from artemis.ml.tools.processors import IDifferentiableFunction


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
        try:
            return recent_moving_average(x, axis=0)  # Works only for python 2.X, with weave
        except ModuleNotFoundError:
            rma = RecentRunningAverage()
            return np.array([rma(xt) for xt in x])


class OptimalStepSizeAverage(object):

    def __init__(self, error_stepsize_target=0.01, initial_stepsize = 1., epsilon=1--7):

        self.error_stepsize_target = error_stepsize_target
        self.error_stepsize = initial_stepsize  # (nu)
        self.error_stepsize_target = 0.001  # (nu-bar)
        self.step_size = 1.  # (a)
        self.avg = 0  # (theta)
        self.beta = 0.
        self.delta = 0.
        self.lambdaa = 0.
        self.epsilon = epsilon
        self.first_iter = True

    def __call__(self, x):
        error = x-self.avg
        error_stepsize = self.error_stepsize / (1 + self.error_stepsize - self.error_stepsize_target)
        self.beta = (1-error_stepsize) * self.beta + error_stepsize * error
        self.delta = (1-error_stepsize) * self.delta + error_stepsize * error**2
        sigma_sq = (self.delta-self.beta**2)/(1+self.lambdaa)
        self.step_size = np.array(1.) if self.first_iter else 1 - (sigma_sq+self.epsilon) / (self.delta+self.epsilon)
        # step_size = 1 - (sigma_sq+self.epsilon) / (delta+self.epsilon)
        self.lambdaa = (1-self.step_size)**2* self.lambdaa + self.step_size**2  # TODO: Test: Should it be (1-step_size**2) ??
        avg = (1-self.step_size) * self.avg + self.step_size * x
        # new_obj = OptimalStepSizer(error_stepsize=error_stepsize, error_stepsize_target=self.error_stepsize_target,
        #     step_size=step_size, avg=avg, beta=beta, delta = delta, lambdaa=lambdaa, epsilon=self.epsilon, first_iter=False)

        if np.any(np.isnan(avg)):
            raise Exception()
        return avg


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


class RunningCenter(object):
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


class ExponentialRunningVariance(object):

    def __init__(self, decay):
        self.decay = decay
        self.running_mean = 0
        self.running_mean_sq = 1

    def __call__(self, x, decay = None):

        decay = self.decay if decay is None else decay
        self.running_mean = (1-decay) * self.running_mean + decay * x
        self.running_mean_sq = (1-decay) * self.running_mean_sq + decay * x**2
        var = self.running_mean_sq - self.running_mean**2
        return np.maximum(0, var)  # TODO: VERIFY THIS... Due to numerical issues, small negative values are possible...


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


_running_averages = {}


def construct_running_averager(ra_type):
    if callable(ra_type):
        return ra_type()
    else:
        return {'simple': RunningAverage, 'recent': RecentRunningAverage, 'osa': OptimalStepSizeAverage}[ra_type]()


def get_global_running_average(value, identifier, ra_type='simple', reset=False):
    """
    Get the running average of a variable.
    :param value: The latest value of the variable
    :param identifier: An identifier (to store the state of the running averager)
    :param ra_type: The type of running averge.  Options are 'simple', 'recent', 'osa'
    :return: The running average
    """

    if not has_global(identifier):
        set_global(identifier, construct_running_averager(ra_type))
    running_averager = get_global(identifier=identifier)
    avg = running_averager(value)
    if reset:
        set_global(identifier, construct_running_averager(ra_type))
    return avg


def periodically_report_running_average(identifier, time, period, value, ra_type = 'simple', format_str = '{identifier}: Average at t={time:.3g}: {avg:.3g} ', reset_between = False):

    report_time = is_elapsed(identifier, period=period, current=time, count_initial=False)
    avg = get_global_running_average(value=value, identifier=identifier, ra_type=ra_type, reset=reset_between and report_time)
    if report_time:
        print(format_str.format(identifier=identifier, time=time, avg=avg))
