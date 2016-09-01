from collections import OrderedDict, namedtuple
from artemis.plotting.data_conversion import vector_length_to_tile_dims
from artemis.plotting.manage_plotting import redraw_figure
from artemis.plotting.matplotlib_backend import get_plot_from_data
from artemis.plotting.plotting_backend import LinePlot, ImagePlot
from matplotlib import gridspec
from matplotlib import pyplot as plt
__author__ = 'peter'


_PlotWindow = namedtuple('PlotWindow', ['figure', 'subplots'])

_Subplot = namedtuple('Subplot', ['axis', 'plot_object'])


# _PLOT_DATA_OBJECTS = OrderedDict()
# _SUBPLOTS = {}  # A dict<figure_identifier: OrdereDicts<subplot_name:axis_object>>

_DBPLOT_FIGURES = {}  # An dict<figure_name: OrderedDict<subplot_name:_Subplot>>
# _DBPLOT_CURRENT_FIGURE = None


def dbplot(data, name = None, plot_constructor = None, plot_mode = 'live', draw_now = True, hang = False, title=None, fig = None):

    if isinstance(fig, plt.Figure):
        # figure = fig
        assert None not in _DBPLOT_FIGURES, "If you pass a figure, you can only do it on the first call to dbplot (for now)"
        _DBPLOT_FIGURES[None] = fig
        fig = None
    elif fig not in _DBPLOT_FIGURES:
        _DBPLOT_FIGURES[fig] = _PlotWindow(figure = plt.figure(), subplots=OrderedDict())
        if name is not None:
            _DBPLOT_FIGURES[fig].figure.canvas.set_window_title(fig)
    # elif fig in _DBPLOT_FIGURES:
    #     pass
        # figure = _DBPLOT_FIGURES[fig]
    # else:
    #     figure = plt.figure()
    #     _DBPLOT_FIGURES[fig] = figure

    # plt.figure(figure)
    #     assert _DBPLOT_CURRENT_FIGURE is None, "You can only pass fig as an argument on your first call to dbplot"
    #     _DBPLOT_CURRENT_FIGURE = fig
    # elif _DBPLOT_CURRENT_FIGURE is None:
    #     _DBPLOT_CURRENT_FIGURE = plt.figure()

    # _plot_window =

    suplot_dict = _DBPLOT_FIGURES[fig].subplots

    if name not in suplot_dict:
        if isinstance(plot_constructor, str):
            plot = {
                'line': LinePlot,
                'img': ImagePlot,
                'colour': lambda: ImagePlot(is_colour_data=True),
                'pic': lambda: ImagePlot(show_clims=False)
                }[plot_constructor]()
        elif plot_constructor is None:
            plot = get_plot_from_data(data, mode=plot_mode)
        else:
            assert hasattr(plot_constructor, "__call__")
            plot = plot_constructor()

        _extend_subplots(fig=fig, subplot_name=name, plot_object=plot)  # This guarantees that the new plot will exist

        # _PLOT_DATA_OBJECTS[name] = plot
        # _extend_subplots(_PLOT_DATA_OBJECTS.keys(), fig)

    # Update the relevant data and plot it.  TODO: Add option for plotting update interval
    # plot = _PLOT_DATA_OBJECTS[name]
    plot = _DBPLOT_FIGURES[fig].subplots[name].plot_object
    plot.update(data)
    # plt.subplot(_SUBPLOTS[name])
    if title is not None:
        _DBPLOT_FIGURES[fig].subplots[name].axis.set_title(title)
        # plt.subplot(_SUBPLOTS[name]).set_title(title)
    plot.plot()
    plt.figure(_DBPLOT_FIGURES[fig].figure.number)
    if draw_now:
        # plt.draw()  # Note: Could be optimized with blit.
        if hang:
            plt.show()
        else:
            redraw_figure()  # Ensures that plot actually shows (whereas plt.draw() may not)


def clear_dbplot(fig = None):
    plt.figure(_DBPLOT_FIGURES[fig].figure.number)
    plt.clf()
    _DBPLOT_FIGURES[fig].subplots.clear()
    # _SUBPLOTS.clear()
    # _PLOT_DATA_OBJECTS.clear()


# def _extend_subplots(key_names, fig):
def _extend_subplots(fig, subplot_name, plot_object):
    """
    :param key_names: New list of names of subplots for figure
    :param fig: Name for figure to extend subplots on
    :return:
    """
    assert fig in _DBPLOT_FIGURES
    old_key_names = _DBPLOT_FIGURES[fig].subplots.keys()
    plt.figure(_DBPLOT_FIGURES[fig].figure.number)
    n_rows, n_cols = vector_length_to_tile_dims(len(old_key_names)+1)
    gs = gridspec.GridSpec(n_rows, n_cols)
    for g, k in zip(gs, old_key_names):  # (gs can be longer but zip will just go to old_key_names)
        ax = _DBPLOT_FIGURES[fig].subplots[k].axis
        ax.set_position(g.get_position(_DBPLOT_FIGURES[fig].figure))

    # Add the new plot
    ax=_DBPLOT_FIGURES[fig].figure.add_subplot(gs[len(old_key_names)])
    ax.set_title(subplot_name)
    _DBPLOT_FIGURES[fig].subplots[subplot_name] = _Subplot(axis=ax, plot_object=plot_object)
