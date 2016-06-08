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



