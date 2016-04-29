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
