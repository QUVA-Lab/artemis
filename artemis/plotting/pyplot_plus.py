import matplotlib.pyplot as plt
import logging
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')

"""
An few extension functions to pyplot
"""


def axhlines(ys, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axhline(y, **specs) for y in ys]


def axvlines(xs, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axvline(x, **specs) for x in xs]


def set_default_figure_size(width, height):
    """
    :param width: Width (in inches, for some reason)
    :param height: Height (also in inches.  One inch is about 2.54cm)
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height
