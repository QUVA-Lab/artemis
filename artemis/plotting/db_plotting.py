from collections import OrderedDict, namedtuple
from artemis.plotting.drawing_plots import redraw_figure
from artemis.plotting.expanding_subplots import select_subplot
from artemis.plotting.matplotlib_backend import get_plot_from_data, TextPlot, MovingPointPlot, Moving2DPointPlot, \
    MovingImagePlot, HistogramPlot, CumulativeLineHistogram
from artemis.plotting.plotting_backend import LinePlot, ImagePlot
from contextlib import contextmanager
from matplotlib import pyplot as plt
import numpy as np
__author__ = 'peter'

"""
Are you tired of setting up subplots, looking through overly complicated matplotlib documentation, and having to restructure
your code just so you can SEE YOUR DAMN VARIABLES?

Well now your troubles are over.

Presenting: dbplot!

dbplot just takes your data, and plots it.  Simple as 1, 2, plot!

No more thinking about what kind plot to use, or how to make updating plots of changing variables.  Just dbplot it!

    dbplot(data, 'my-data')

dbplot will look at your data, and figure out which type of plot is appropriate.  If you don't like it, you can fully
customize it, using the plot_type argument.

dbplot makes online plotting easy.  You want to plot updates to your variable?  Just dbplot it!

    dbplot(var, 'my-var')
    dbplot(updated_var, 'my-var')

For just float('inf') easy payments of $0, you can make dbplot yours for use around the office, home, or garden.

Check out demo_dbplot.py for some exciting demos of what dbplot can do.

Remember:  If you can't see your data, you are a fraud and your research career will fail.  So try dbplot today!
"""


def dbplot(data, name = None, plot_type = None, axis=None, plot_mode = 'live', draw_now = True, hang = False, title=None,
           fig = None, xlabel = None, ylabel = None, draw_every = None, layout=None, legend=None, plot_constructor=None):
    """
    Plot arbitrary data.  This program tries to figure out what type of plot to use.

    :param data: Any data.  Hopefully, we at dbplot will be able to figure out a plot for it.
    :param name: A name uniquely identifying this plot.
    :param plot_type: A specialized constructor to be used the first time when plotting.  You can also pass
        certain string to give hints as to what kind of plot you want (can resolve cases where the given data could be
        plotted in multiple ways):
        'line': Plots a line plot
        'img': An image plot
        'colour': A colour image plot
        'pic': A picture (no scale bars, axis labels, etc).
    :param axis: A string identifying which axis to plot on.  By default, it is the same as "name".  Only use this
        argument if you indend to make multiple dbplots share the same axis.
    :param plot_mode: Influences how the data should be used to choose the plot type:
        'live': Best for 'live' plots that you intend to update as new data arrives
        'static': Best for 'static' plots, that you do not intend to update
        'image': Try to represent the plot as an image
    :param draw_now: Draw the plot now (you may choose false if you're going to add another plot immediately after and
        don't want have to draw this one again.
    :param hang: Hang on the plot (wait for it to be closed before continuing)
    :param title: Title of the plot (will default to name if not included)
    :param fig: Name of the figure - use this when you want to create multiple figures.
    """

    if isinstance(fig, plt.Figure):
        assert None not in _DBPLOT_FIGURES, "If you pass a figure, you can only do it on the first call to dbplot (for now)"
        _DBPLOT_FIGURES[None] = fig
        fig = None
    elif fig not in _DBPLOT_FIGURES:
        _DBPLOT_FIGURES[fig] = _PlotWindow(figure = _make_dbplot_figure(), subplots=OrderedDict(), axes = {})
        if fig is not None:
            _DBPLOT_FIGURES[fig].figure.canvas.set_window_title(fig)

    suplot_dict = _DBPLOT_FIGURES[fig].subplots

    if axis is None:
        axis=name

    if name not in suplot_dict:
        if plot_constructor is not None:
            print "Warning: The 'plot_constructor' argument to dbplot is deprecated.  Use plot_type instead"
            assert plot_type is None
            plot_type = plot_constructor

        if isinstance(plot_type, str):
            plot = {
                'line': LinePlot,
                'thick-line': lambda: LinePlot(plot_kwargs={'linewidth': 3}),
                'pos_line': lambda: LinePlot(y_bounds=(0, None), y_bound_extend=(0, 0.05)),
                # 'pos_line': lambda: LinePlot(y_bounds=(0, None)),
                'img': ImagePlot,
                'colour': lambda: ImagePlot(is_colour_data=True),
                'equal_aspect': lambda: ImagePlot(aspect='equal'),
                'image_history': lambda: MovingImagePlot(),
                'pic': lambda: ImagePlot(show_clims=False, aspect='equal'),
                'notice': lambda: TextPlot(max_history=1, horizontal_alignment='center', vertical_alignment='center', size='x-large'),
                'cost': lambda: MovingPointPlot(y_bounds=(0, None), y_bound_extend=(0, 0.05)),
                'percent': lambda: MovingPointPlot(y_bounds=(0, 100)),
                'trajectory': lambda: Moving2DPointPlot(axes_update_mode='expand'),
                'trajectory+': lambda: Moving2DPointPlot(axes_update_mode='expand', x_bounds=(0, None), y_bounds=(0, None)),
                'histogram': lambda: HistogramPlot(edges = np.linspace(-5, 5, 20)),
                'cumhist': lambda: CumulativeLineHistogram(edges = np.linspace(-5, 5, 20)),
                }[plot_type]()
        elif plot_type is None:
            plot = get_plot_from_data(data, mode=plot_mode)
        else:
            assert hasattr(plot_type, "__call__")
            plot = plot_type()

        if axis in _DBPLOT_FIGURES[fig].axes:
            _DBPLOT_FIGURES[fig].subplots[name] = _Subplot(axis=_DBPLOT_FIGURES[fig].axes[axis], plot_object=plot)
            plt.sca(_DBPLOT_FIGURES[fig].axes[axis])
        else:  # Make a new axis
            # _extend_subplots(fig=fig, subplot_name=name, axis_name=axis, plot_object=plot)  # This guarantees that the new plot will exist
            ax = select_subplot(axis, fig=_DBPLOT_FIGURES[fig].figure, layout=_default_layout if layout is None else layout)
            
            ax.set_title(axis)

            _DBPLOT_FIGURES[fig].subplots[name] = _Subplot(axis=ax, plot_object=plot)
            _DBPLOT_FIGURES[fig].axes[axis] = ax
            
            if xlabel is not None:
                _DBPLOT_FIGURES[fig].subplots[name].axis.set_xlabel(xlabel)
            if ylabel is not None:
                _DBPLOT_FIGURES[fig].subplots[name].axis.set_ylabel(ylabel)
            if draw_every is not None:
                _draw_counters[fig, name] = -1

    # Update the relevant data and plot it.  TODO: Add option for plotting update interval
    plot = _DBPLOT_FIGURES[fig].subplots[name].plot_object
    plot.update(data)
    plot.plot()
    if title is not None:
        _DBPLOT_FIGURES[fig].subplots[name].axis.set_title(title)
    if legend is not None:
        _DBPLOT_FIGURES[fig].subplots[name].axis.legend(legend, loc='best', framealpha=0.5)

    if draw_now and not _hold_plots:
        if draw_every is not None:
            _draw_counters[fig, name]+=1
            if _draw_counters[fig, name] % draw_every != 0:
                return _DBPLOT_FIGURES[fig].subplots[name].axis
        if hang:
            plt.figure(_DBPLOT_FIGURES[fig].figure.number)
            plt.show()
        else:
            redraw_figure(_DBPLOT_FIGURES[fig].figure)
    return _DBPLOT_FIGURES[fig].subplots[name].axis


_PlotWindow = namedtuple('PlotWindow', ['figure', 'subplots', 'axes'])

_Subplot = namedtuple('Subplot', ['axis', 'plot_object'])

_DBPLOT_FIGURES = {}  # An dict<figure_name: _PlotWindow(figure, OrderedDict<subplot_name:_Subplot>)>

_DEFAULT_SIZE = None

_draw_counters = {}

_hold_plots = False

_hold_plot_counter = 0

_default_layout = 'grid'

def reset_dbplot():
    for fig_name, plot_window in _DBPLOT_FIGURES.items():
        plt.close(plot_window.figure)
        del _DBPLOT_FIGURES[fig_name]


def set_dbplot_figure_size(width, height):
    global _DEFAULT_SIZE
    _DEFAULT_SIZE = (width, height)


def set_dbplot_default_layout(layout):
    global _default_layout
    _default_layout = layout


def get_dbplot_figure(name=None):
    return _DBPLOT_FIGURES[name].figure


def get_dbplot_subplot(name, fig_name=None):
    return _DBPLOT_FIGURES[fig_name].subplots[name].axis


def _make_dbplot_figure():

    if _DEFAULT_SIZE is None:
        fig= plt.figure()
    else:
        fig= plt.figure(figsize=_DEFAULT_SIZE)
    return fig



def freeze_dbplot(name, fig = None):
    del _DBPLOT_FIGURES[fig].subplots[name]


def freeze_all_dbplots(fig = None):
    for name in _DBPLOT_FIGURES[fig].subplots.keys():
        freeze_dbplot(name, fig=fig)

@contextmanager
def hold_dbplots(fig = None, plot_every = None):
    """
    Use this in a "with" statement to prevent plotting until the end.
    :param fig:
    :return:
    """
    global _hold_plots
    _hold_plots = True
    yield
    _hold_plots = False

    if plot_every is not None:
        global _hold_plot_counter
        plot_now = _hold_plot_counter % plot_every == 0
        _hold_plot_counter+=1
    else:
        plot_now = True

    if plot_now:
        redraw_figure(_DBPLOT_FIGURES[fig].figure)


def clear_dbplot(fig = None):
    if fig in _DBPLOT_FIGURES:
        plt.figure(_DBPLOT_FIGURES[fig].figure.number)
        plt.clf()
        _DBPLOT_FIGURES[fig].subplots.clear()
        _DBPLOT_FIGURES[fig].axes.clear()


def get_dbplot_axis(axis_name, fig=None):
    """
    Get the named axis of a dbplot.
    """
    return _DBPLOT_FIGURES[fig].axes[axis_name]


def dbplot_hang():
    plt.show()
