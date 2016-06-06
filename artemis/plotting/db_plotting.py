from collections import OrderedDict
from artemis.plotting.data_conversion import vector_length_to_tile_dims
from artemis.plotting.matplotlib_backend import get_plot_from_data
from artemis.plotting.plotting_backend import LinePlot, ImagePlot
from artemis.plotting.pyplot_plus import show_and_continue, show_and_hang
from matplotlib import gridspec
from matplotlib import pyplot as plt
__author__ = 'peter'


_PLOT_DATA_OBJECTS = OrderedDict()
_SUBPLOTS = OrderedDict()


def dbplot(data, name = None, plot_constructor = None, plot_mode = 'live', draw_now = True, hang = False, title=None):

    if name not in _PLOT_DATA_OBJECTS:
        if isinstance(plot_constructor, str):
            plot = {
                'line': LinePlot,
                'img': ImagePlot,
                'colour': lambda: ImagePlot(is_colour_data=True)
                }[plot_constructor]()
        elif plot_constructor is None:
            plot = get_plot_from_data(data, mode=plot_mode)
        else:
            assert hasattr(plot_constructor, "__call__")
            plot = plot_constructor()
        _PLOT_DATA_OBJECTS[name] = plot
        _extend_subplots(_PLOT_DATA_OBJECTS.keys())

    # Update the relevant data and plot it.  TODO: Add option for plotting update interval
    plot = _PLOT_DATA_OBJECTS[name]
    plot.update(data)
    plt.subplot(_SUBPLOTS[name])
    if title is not None:
        plt.subplot(_SUBPLOTS[name]).set_title(title)
    plot.plot()
    if draw_now:
        # plt.draw()  # Note: Could be optimized with blit.
        if hang:
            show_and_hang()
        else:
            show_and_continue()

def _extend_subplots(key_names):
    n_rows, n_cols = vector_length_to_tile_dims(len(key_names))
    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.gcf()
    for g, k in zip(gs, key_names):
        if k in _SUBPLOTS:
            ax = _SUBPLOTS[k]
            ax.set_position(g.get_position(plt.gcf()))
        else:
            ax=fig.add_subplot(g)
            ax.set_title(k)
            _SUBPLOTS[k] = ax
