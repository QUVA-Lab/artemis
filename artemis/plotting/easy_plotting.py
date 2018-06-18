from collections import OrderedDict
from artemis.general.nested_structures import flatten_struct
from artemis.plotting.data_conversion import vector_length_to_tile_dims
import numpy as np
from matplotlib import pyplot as plt
from artemis.plotting.matplotlib_backend import get_plot_from_data

__author__ = 'peter'


def ezplot(anything, plots = None, hang = True, **plot_preference_kwargs):
    """
    Make a plot of anything.  Anything at all.
    :param anything: Anything.
    """
    data_dict = flatten_struct(anything)
    figure, plots = plot_data_dict(data_dict, plots, mode = 'static', hang = hang, **plot_preference_kwargs)
    return figure, plots


def plot_data_dict(data_dict, plots = None, mode = 'static', hang = True, figure = None, size = None, **plot_preference_kwargs):
    """
    Make a plot of data in the format defined in data_dict
    :param data_dict: dict<str: plottable_data>
    :param plots: Optionally, a dict of <key: IPlot> identifying the plot objects to use (keys should
        be the same as those in data_dict).
    :return: The plots (same ones you provided if you provided them)
    """

    assert mode in ('live', 'static')
    if isinstance(data_dict, list):
        assert all(len(d) == 2 for d in data_dict), "You can provide data as a list of 2 tuples of (plot_name, plot_data)"
        data_dict = OrderedDict(data_dict)

    if plots is None:
        plots = {k: get_plot_from_data(v, mode = mode, **plot_preference_kwargs) for k, v in data_dict.items()}

    if figure is None:
        if size is not None:
            from pylab import rcParams
            rcParams['figure.figsize'] = size
        figure = plt.figure()
    n_rows, n_cols = vector_length_to_tile_dims(len(data_dict))
    for i, (k, v) in enumerate(data_dict.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plots[k].update(v)
        plots[k].plot()
        plt.title(k, fontdict = {'fontsize': 8})
    oldhang = plt.isinteractive()
    plt.interactive(not hang)
    plt.show()
    plt.interactive(oldhang)
    return figure, plots


def funplot(func, xlims = None, n_points = 100, keep_ylims = False, ax=None, **plot_args):
    """
    Plot a function
    :param func:
    :param xlims:
    :param n_points:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    if xlims is None:
        xlims = ax.get_xbound()
    xs, xe = xlims
    x = np.logspace(np.log10(xs), np.log10(xe), n_points) if ax.get_xscale() else np.linspace(xs, xe, n_points)
    if keep_ylims:
        ylims = ax.get_ybound()
    h=ax.plot(x, func(x), **plot_args)
    if keep_ylims:
        ax.set_ybound(*ylims)
    ax.set_xbound(*xlims)
    return h


def right_side_legend(handles=None, ax = None):

    if handles is None:
        if ax is None:
            ax = plt.gca()

        handles = (ax.lines + ax.patches + ax.collections + ax.containers)

    labels = [h.get_label() for h in handles]

    x_left, x_right = ax.get_xbound()

    for h, l in zip(handles, labels):
        plt.text(x=x_right, y=h.get_ydata()[-1], s=l, color=h.get_color(), horizontalalignment='left')

