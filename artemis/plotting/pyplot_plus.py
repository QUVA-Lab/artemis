import matplotlib.pyplot as plt
import logging
import numpy as np
import matplotlib.colors as colors
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')

"""
An few extension functions to pyplot
"""


def axhlines(ys, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = plt.gca().get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = plt.plot(x_points, y_points, scalex = False, **plot_kwargs)
    return plot


def axvlines(xs, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = plt.gca().get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = plt.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot

    # return [plt.axhline(y, **specs) for y in ys]
    # return [plt.axvline(x, **specs) for x in xs]


def set_default_figure_size(width, height):
    """
    :param width: Width (in inches, for some reason)
    :param height: Height (also in inches.  One inch is about 2.54cm)
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height


_lines_colour_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]


def get_lines_color_cycle():
    return _lines_colour_cycle


def get_line_color(ix, modifier=None):
    colour = _lines_colour_cycle[ix]
    if modifier=='dark':
        return tuple(c/2 for c in colors.hex2color(colour))
    elif modifier=='light':
        return tuple(1-(1-c)/2 for c in colors.hex2color(colour))
    elif modifier is not None:
        raise NotImplementedError(modifier)
    return
