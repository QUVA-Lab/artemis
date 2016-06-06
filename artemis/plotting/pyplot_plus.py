from artemis.fileman.local_dir import get_local_path
import matplotlib.pyplot as plt
import logging
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')

"""
An extension to matplotlib's pyplot library.

We use this to

a) Get rid of this "interactive/non-interactive" mode paradigm.  Instead we just have different functions to plot /
   plot and hang.
b) Add some useful functions.
"""


def axhlines(ys, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axhline(y, **specs) for y in ys]


def axvlines(xs, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axvline(x, **specs) for x in xs]


_SHOW_FCN = plt.show


def draw():
    plt.draw()
    plt.pause(0.00001)


_SHOW_AND_CONTINUE_FCN = draw


def set_show_and_hang_callback(cb):
    """
    :param cb: A function that takes an optional figure as an argument.  Or None to reset to the original show.
    """
    global _SHOW_FCN
    if cb is None:
        _SHOW_FCN = plt.show
    else:
        _SHOW_FCN = cb


def set_show_and_continue_callback(cb):
    global _SHOW_AND_CONTINUE_FCN
    if cb is None:
        _SHOW_AND_CONTINUE_FCN = draw
    else:
        _SHOW_AND_CONTINUE_FCN = show_and_continue


def show_and_hang():
    _SHOW_FCN()


def show_and_continue():
    _SHOW_AND_CONTINUE_FCN()
    # pause(0.00001)  # Required to display, at least on some backends.


class SaveToFile():

    def __init__(self, path):
        self._path = get_local_path(path)
        self._first = False

    def __call__(self, fig = None):
        if self._first:
            ARTEMIS_LOGGER.info('Creating figure at "%s"' % (self._path, ))
            self._first = False
        if fig is None:
            fig = plt.gcf()
        fig.savefig(self._path)
