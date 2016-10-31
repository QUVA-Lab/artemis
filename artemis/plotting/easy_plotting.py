from collections import OrderedDict
from artemis.general.nested_structures import flatten_struct
from artemis.plotting.data_conversion import vector_length_to_tile_dims
# import plotting.matplotlib_backend as eplt
import artemis.plotting.plotting_backend as eplt
import numpy as np

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
        plots = {k: eplt.get_plot_from_data(v, mode = mode, **plot_preference_kwargs) for k, v in data_dict.iteritems()}

    if figure is None:
        if size is not None:
            from pylab import rcParams
            rcParams['figure.figsize'] = size
        figure = eplt.figure()
    n_rows, n_cols = vector_length_to_tile_dims(len(data_dict))
    for i, (k, v) in enumerate(data_dict.iteritems()):
        eplt.subplot(n_rows, n_cols, i+1)
        plots[k].update(v)
        plots[k].plot()
        eplt.title(k, fontdict = {'fontsize': 8})
    oldhang = eplt.isinteractive()
    eplt.interactive(not hang)
    eplt.show()
    eplt.interactive(oldhang)
    return figure, plots


def funplot(func, xlims = None, n_points = 100, keep_ylims = False, **plot_args):
    """
    Plot a function
    :param func:
    :param xlims:
    :param n_points:
    :return:
    """
    if xlims is None:
        xlims = eplt.gca().get_xbound()
    xs, xe = xlims
    x = np.linspace(xs, xe, n_points)
    if keep_ylims:
        ylims = eplt.gca().get_ybound()
    eplt.plot(x, func(x), **plot_args)
    if keep_ylims:
        eplt.gca().set_ybound(*ylims)
    eplt.gca().set_xbound(*xlims)
