from cycler import cycler
import matplotlib.pyplot as plt
import logging
import numpy as np
import matplotlib.colors as colors
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap
from si_prefix import si_format
import matplotlib
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')

"""
An few extension functions to pyplot
"""


def axhlines(ys, ax=None, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = ax.get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scalex = False, **plot_kwargs)
    return plot


def axvlines(xs, ax=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot


def parse_plot_args(x_or_y, y):
    if y is not None:
        x, y = x_or_y, y
        y = np.array(y, copy=False)
    else:
        y = np.array(x_or_y, copy=False)
        x = np.arange(y.shape[0])
    return x, y


def plot_stacked_signals(x_or_y, y=None, draw_zero_lines = True, ax=None, sep=None, labels=None, **kwargs):
    """
    Stack signals into a single plot and plot them together (e.g. for e.e.g. data)
    :param x_or_y: y or (x, y) signals in the same format as plt.plot
    :param y: Y signal (if you include it)
    :param draw_zero_lines: Draw horizontal lines marking y=0
    :param ax: Axis to plot in
    :param sep: Vertical separation between signal plots (None means automatic)
    :param kwargs: Other keyword args from plt.plot
    :return: A list of linehandles returned from plt.plot
    """
    x, y = parse_plot_args(x_or_y, y)

    if y.ndim==1:
        y = y[:, None]

    if sep is None:
        sep = np.abs(np.min(y)) + np.abs(np.max(y))

    if ax is None:
        ax = plt.gca()

    offsets = - np.arange(y.shape[1])*sep
    y = y + offsets

    h = ax.plot(x, y, **kwargs)

    if labels is False:
        ax.tick_params(axis='y', labelleft='off')
    elif isinstance(labels, (list, tuple)):
        assert len(labels)==y.shape[1], 'Number of labels must match number of signals.'
        ax.set_yticks(offsets)
        ax.set_yticklabels(labels)
    if draw_zero_lines:
        hz = axhlines(offsets, color='k', zorder=1.5)
    else:
        hz = None

    return h, hz


def stemlight(x_or_y, y=None, ax=None, **plot_kwargs):
    """
    A ligher version of a stem plot.  Avoids creating a new plot object for each point.
    :param x_or_y: The x value (if any) otherwise the y value
    :param y: The y value, if x provided
    :param ax: Optionally, the axis to plot on
    :param plot_kwargs: Passed to plt.plot
    :return: The handles from plt.plot
    """
    if ax is None:
        ax = plt.gca()
    x, y = parse_plot_args(x_or_y, y)
    # Replicate all x-points thrice:
    ixs = np.arange(len(x)*3)//3
    x_pts = x[ixs]
    y_pts = y[ixs]

    y_pts[::3] = 0
    y_pts[2::3] = 0
    return plt.plot(x_pts, y_pts, **plot_kwargs)


def event_raster_plot(events, sep=1, ax=None, marker = '+', **scatter_kwargs):
    """
    Make a "raster plot" of events (something like a spike-raster plot from neuroscience).
    :param Sequence[Sequence[float]] events: A series of event times
    :param sep: The vertical spacing between rows in the raster plot
    :param ax: Optinally, the axis to plot in
    :param marker: The marker to use to mark events
    :param scatter_kwargs: Passed to plt.scatter
    :return: A list of handles returned by plt.scatter
    """
    if ax is None:
        ax = plt.gca()
    h = [ax.scatter(x=event_list, y=np.zeros_like(event_list)-i*sep, marker=marker, **scatter_kwargs) for i, event_list in enumerate(events)]
    return h


def set_default_figure_size(width, height):
    """
    :param width: Width (in inches, for some reason)
    :param height: Height (also in inches.  One inch is about 2.54cm)
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height


_lines_colour_cycle = (p['color'] for p in plt.rcParams['axes.prop_cycle'])


def get_lines_color_cycle():
    return (p['color'] for p in plt.rcParams['axes.prop_cycle'])


def get_color_cycle_map(name, length):
    cmap = getattr(plt.cm, name)
    cycle = cycler('color', cmap(np.linspace(0, 1, length)))
    return [c['color'] for c in cycle]


def set_lines_color_cycle_map(name, length):
    cmap = getattr(plt.cm, name)
    c = cycler('color', cmap(np.linspace(0, 1, length)))
    matplotlib.rcParams['axes.prop_cycle'] = c


def get_line_color(ix, modifier=None):
    colour = next(c for i, c in enumerate(get_lines_color_cycle()) if i==ix)
    if modifier=='dark':
        return tuple(c/2 for c in colors.hex2color(colour))
    elif modifier=='light':
        return tuple(1-(1-c)/2 for c in colors.hex2color(colour))
    elif modifier is not None:
        raise NotImplementedError(modifier)
    return colors.hex2color(colour)


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


def _get_centered_colour_scale(cmin, cmax):

    total_max = np.maximum(np.abs(cmin), np.abs(cmax))

    crange = np.linspace(cmin/total_max, cmax/total_max, 256)

    r = np.minimum(0, crange)
    g = np.zeros_like(crange)
    b = np.maximum(0, crange)
    # (now we have
    scale = np.concatenate([r[:, None], g[:, None], b[:, None]], axis=1)/2.+.5
    return scale


register_cmap('redgreyblue', LinearSegmentedColormap('redgreyblue', {
        'red': [(0, 1, 1), (0.5, .5, .5), (1, 0, 0)],
        'green': [(0, 0, 0), (0.5, .5, .5), (1, 0, 0)],
        'blue': [(0, 0, 0), (0.5, .5, .5), (1, 1, 1)],
        }))

register_cmap('redblackblue', LinearSegmentedColormap('redgreyblue', {
        'red': [(0, 1, 1), (0.5, 0, 0), (1, 0, 0)],
        'green': [(0, 0, 0), (1, 0, 0)],
        'blue': [(0, 0, 0), (0.5, 0, 0), (1, 1, 1)],
        }))


def center_colour_scale(h):
    current_min, current_max = h.get_clim()
    absmax = np.maximum(np.abs(current_min), np.abs(current_max))
    h.set_clim((-absmax, absmax))


def set_centered_colour_map(h, map='redblackblue'):
    """
    Set a colourmap that is red in negative regions, grey at zero, and blue in positive regions.
    :param h: An AxesImage handle (returned by plt.imshow)
    """
    h.set_cmap(map)
    center_colour_scale(h)


def outside_right_legend(ax=None, width_squeeze = 0.8):
    if ax is None:
        ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * width_squeeze, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
