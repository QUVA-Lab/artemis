import itertools
import time
import types

__author__ = 'peter'
import numpy as np


class CheckPointCounter(object):
    """
    Scenerio: You have a loop, and want to do something periodically within that loop, but not
    on every iteration.  Use this object to tell you if you want to do that thing or not.

    You give it a series of checkpoints, and each time it is called, it will return the number
    of checkpoints it has passed since it was last called.  If you only want to know IF it's passed
    a checkpoint, you can treat the return as a boolean.
    """

    def __init__(self, checkpoints):
        self._checkpoints = checkpoints
        self._index = 0

    def check(self, progress):
        """
        :param progress: Indicator of the progress - on the same scale of the checkpoints.
        :return: (points_passed, done), where:
            points_passed is the number of checkpoints passed since the last call to this method,
            done is a boolean indicating whether the last checkpoint has been passed.

        Note that done will only be True of points_passed>0
        """
        counter = 0
        done = False

        while True:
            if self._index == len(self._checkpoints):
                done = True
                break
            elif progress < self._checkpoints[self._index]:
                break
            else:
                counter += 1
                self._index += 1

        return counter, done


class Checkpoints(object):
    """
    An object where you specify checkpoints and return true every time one of those checkpoints is passed.
    """

    def __init__(self, checkpoint_generator, default_units = None, skip_first = False):
        """
        :param checkpoint_generator: Can be:
            A generator object returning checkpoints
            A list/tuple/array of checkpoints
            ('even', interval)
            ('exp', first, growth)
            None
        """

        if isinstance(checkpoint_generator, str):
            assert default_units in ('sec', None)
            assert checkpoint_generator.endswith('s')
            checkpoint_generator = int(checkpoint_generator[:-1])
            default_units = 'sec'
        elif default_units is None:
            default_units = 'iter'

        assert default_units in ('iter', 'sec')
        self.default_units = default_units
        if isinstance(checkpoint_generator, tuple):
            distribution = checkpoint_generator[0]
            if distribution == 'even':
                interval, = checkpoint_generator[1:]
                checkpoint_generator = (interval*i for i in itertools.count(1))
            elif distribution == 'exp':
                first, growth = checkpoint_generator[1:]
                checkpoint_generator = (first*i*(1+growth)**(i-1) for i in itertools.count(0))
            else:
                raise Exception("Can't make a checkpoint generator {}".format(checkpoint_generator))
        elif isinstance(checkpoint_generator, (list, tuple, np.ndarray)):
            checkpoint_generator = iter(checkpoint_generator)
        elif isinstance(checkpoint_generator, (int, float)):
            step = checkpoint_generator
            checkpoint_generator = (step*i for i in itertools.count(0))
        else:
            assert isinstance(checkpoint_generator, types.GeneratorType)

        if skip_first:
            next(checkpoint_generator)

        self.checkpoint_generator = checkpoint_generator
        self._next_checkpoint = float('inf') if checkpoint_generator is None else next(checkpoint_generator)
        self._counter = 0
        self._start_time = time.time()

    def __call__(self, t=None):
        if t is None:
            t = self._counter if self.default_units == 'iter' else time.time() - self._start_time
        self._counter += 1
        if t >= self._next_checkpoint:
            while t >= self._next_checkpoint:
                self._next_checkpoint = next(self.checkpoint_generator)
            return True
        else:
            return False

    def get_count(self):
        return self._counter


_COUNTERS_DICT = {}


def do_every(interval, counter_id=None, units = None):
    """
    Return true periodically.  Eg.

        # Plot every 100'th image.
        for im in images:
            if do_every(100):
                plot(current_image)

    :param interval: A number saying how often to return true.
    :param counter_id: ID used to uniquely identify this call to do_every.  (useful if you have more than
        one do_every call)
    :param units: 'iter' or 'sec'
    :return: True if checkpoint has just been passed, otherwise false.
    """
    if counter_id not in _COUNTERS_DICT:
        _COUNTERS_DICT[counter_id] = Checkpoints(checkpoint_generator=interval, default_units=units)
    return _COUNTERS_DICT[counter_id]()
