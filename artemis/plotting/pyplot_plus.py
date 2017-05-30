import matplotlib.pyplot as plt
import logging
import numpy as np
import matplotlib.colors as colors
from si_prefix import si_format

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
    return colors.hex2color(colour)


def non_uniform_imshow(im, x_locs=None, y_locs=None, spacing='lin', format_str='{:.2g}', **other_imagesc_args):

    if x_locs is not None:
        assert len(x_locs)==im.shape[1]
        assert np.all(np.diff(x_locs)>0)
    if y_locs is not None:
        assert len(y_locs)==im.shape[0]
        assert np.all(np.diff(y_locs)>0)

    handle = plt.imshow(im, **other_imagesc_args)
    plt.gca().invert_yaxis()

    relabel_axes(plt.gca(), xvalues=x_locs, yvalues=y_locs, format_str=format_str)
    return handle


def relabel_axis(axis, value_array, n_points = 5, format_str='{:.2g}'):
    ticks = np.round(np.linspace(0, len(value_array)-1, num=n_points)).astype('int')
    axis.set_ticks(ticks)
    if format_str=='SI':
        axis.set_ticklabels([si_format(t, format_str='{value}{prefix}') for t in value_array[ticks]])
    else:
        axis.set_ticklabels([format_str.format(t) for t in value_array[ticks]])


def relabel_axes(axes, n_points='auto', xvalues=None, yvalues=None, xlabel=None, ylabel=None, format_str='{:.2g}'):

    if n_points=='auto':
        n_points = 5 if len(xvalues)%5==0 else \
            4 if len(xvalues)%4==0 else \
            3 if len(xvalues)%3==0 else \
            5
    if xvalues is not None:
        relabel_axis(axes.xaxis, xvalues, n_points=n_points, format_str=format_str)
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if yvalues is not None:
        relabel_axis(axes.yaxis, yvalues, n_points=n_points, format_str=format_str)
    if ylabel is not None:
        axes.set_ylabel(ylabel)


def remove_x_axis():
    plt.tick_params(axis='x', labelbottom='off')


def remove_y_axis():
    plt.tick_params(axis='y', labelbottom='off')
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
