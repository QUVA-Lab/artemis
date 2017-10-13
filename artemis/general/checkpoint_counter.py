import itertools

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

    def __init__(self, checkpoint_generator):
        """
        :param checkpoint_generator: Can be:
            A generator object returning checkpoints
            A list/tuple/array of checkpoints
            ('even', interval)
            ('exp', first, growth)
            None
        """

        if isinstance(checkpoint_generator, tuple):
            distribution = checkpoint_generator[0]
            if distribution == 'even':
                interval, = checkpoint_generator[1:]
                checkpoint_generator = (interval*i for i in itertools.count(1))
            elif distribution == 'exp':
                first, growth = checkpoint_generator[1:]
                checkpoint_generator = (first*i*(1+growth)**(i-1) for i in itertools.count(1))
            else:
                raise Exception("Can't make a checkpoint generator {}".format(checkpoint_generator))
        elif isinstance(checkpoint_generator, (list, tuple, np.ndarray)):
            checkpoint_generator = iter(checkpoint_generator)
        self.checkpoint_generator = checkpoint_generator
        self._next_checkpoint = float('inf') if checkpoint_generator is None else next(checkpoint_generator)

    def __call__(self, t):
        if t >= self._next_checkpoint:
            while self._next_checkpoint<t:
                self._next_checkpoint = next(self.checkpoint_generator)
            return True
        else:
            return False
