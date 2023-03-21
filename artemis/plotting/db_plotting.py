from collections import OrderedDict, namedtuple
from functools import partial

from six import string_types

from artemis.config import get_artemis_config_value
from artemis.general.checkpoint_counter import Checkpoints
from artemis.plotting.matplotlib_backend import BarPlot, BoundingBoxPlot, ResamplingLineHistory
from matplotlib.axes import Axes
from matplotlib.gridspec import SubplotSpec
from contextlib import contextmanager
import numpy as np
from matplotlib import pyplot as plt
from artemis.plotting.drawing_plots import redraw_figure
from artemis.plotting.expanding_subplots import select_subplot
from artemis.plotting.matplotlib_backend import get_plot_from_data, TextPlot, MovingPointPlot, Moving2DPointPlot, \
    MovingImagePlot, HistogramPlot, CumulativeLineHistogram
from artemis.plotting.matplotlib_backend import LinePlot, ImagePlot, is_server_plotting_on

if is_server_plotting_on():
    from artemis.remote.plotting.plotting_client import deconstruct_plotting_server

__author__ = 'peter'

"""
dbplot just takes your data, and plots it.  No fuss, no muss.  No more thinking about what kind plot to use, or how to
make updating plots of changing variables.  Just dbplot it.

    dbplot(data, 'my-data')

dbplot will look at your data, and figure out which type of plot is appropriate.  If you don't like it, you can
customize it, using the plot_type argument.

dbplot makes online plotting easy.  You want to plot updates to your variable?  Just dbplot it.

    dbplot(var, 'my-var')
    dbplot(updated_var, 'my-var')

See demo_dbplot.py for some demos of what dbplot can do.
"""


def dbplot(data, name = None, plot_type = None, axis=None, plot_mode = 'live', draw_now = True, hang = False, title=None,
           fig = None, xlabel = None, ylabel = None, draw_every = None, layout=None, legend=None, grid=False,
           wait_for_display_sec=0, cornertext = None, reset_color_cycle = False):
    """
    Plot arbitrary data and continue execution.  This program tries to figure out what type of plot to use.

    :param data: Any data.  Hopefully, we at dbplot will be able to figure out a plot for it.
    :param name: A name uniquely identifying this plot.
    :param Union[Callable[[],LinePlot],str,Tuple[Callable, Dict]] plot_type : A specialized constructor to be used the
        first time when plotting.  Several predefined constructors are defined in the DBPlotTypes class - you can pass
        those.  For back-compatibility you can also pass a string matching the name of one of the fields in the DBPlotTypes
        class.
        DBPlotTypes.LINE: Plots a line plot
        DBPlotTypes.IMG: An image plot
        DBPlotTypes.COLOUR: A colour image plot
        DBPlotTypes.PIC: A picture (no scale bars, axis labels, etc)
        You can also, pass a tuple of (constructor, keyword_args) where keyword args is a dict of arcuments to the plot
        constructor.
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
    :param grid: Turn the grid on
    :param wait_for_display_sec: In server mode, you can choose to wait maximally wait_for_display_sec seconds before this
        call returns. In case plotting is finished earlier, the call returns earlier. Setting wait_for_display_sec to a negative number will cause the call to block until the plot has been displayed.
    """
    if is_server_plotting_on():
        # Redirect the function call to the plotting server.  The flag gets turned on in a configuration file.  It is
        # turned off when this file is run ON the plotting server, from the first line in plotting_server.py
        arg_locals = locals().copy()
        from artemis.remote.plotting.plotting_client import dbplot_remotely
        dbplot_remotely(arg_locals=arg_locals)
        return

    if data.__class__.__module__ == 'torch' and data.__class__.__name__ == 'Tensor':
        data = data.detach().cpu().numpy()

    plot_object = _get_dbplot_plot_object(fig)  # type: _PlotWindow

    suplot_dict = plot_object.subplots

    if axis is None:
        axis=name

    if name not in suplot_dict:  # Initialize new axis

        if isinstance(plot_type, str):
            plot = DBPlotTypes.from_string(plot_type)()
        elif isinstance(plot_type, tuple):
            assert len(plot_type)==2 and isinstance(plot_type[0], str) and isinstance(plot_type[1], dict), 'If you specify a tuple for plot_type, we expect (name, arg_dict).  Got: {}'.format(plot_type)
            plot_type_name, plot_type_args = plot_type
            if isinstance(plot_type_name, str):
                plot = DBPlotTypes.from_string(plot_type_name)(**plot_type_args)
            elif callable(plot_type_name):
                plot = plot_type_name(**plot_type_args)
            else:
                raise Exception('The first argument of the plot type tuple must be a plot type name or a callable plot type constructor.')
        elif plot_type is None:
            plot = get_plot_from_data(data, mode=plot_mode)
        else:
            assert hasattr(plot_type, "__call__")
            plot = plot_type()

        if isinstance(axis, SubplotSpec):
            axis = plt.subplot(axis)
        if isinstance(axis, Axes):
            ax = axis
            ax_name = str(axis)
        elif isinstance(axis, string_types) or axis is None:
            ax = select_subplot(axis, fig=plot_object.figure, layout=_default_layout if layout is None else layout)
            ax_name = axis
            # ax.set_title(axis)
        else:
            raise Exception("Axis specifier must be a string, an Axis object, or a SubplotSpec object.  Not {}".format(axis))

        if ax_name not in plot_object.axes:
            ax.set_title(name)
            plot_object.subplots[name] = _Subplot(axis=ax, plot_object=plot)
            plot_object.axes[ax_name] = ax

        plot_object.subplots[name] = _Subplot(axis=plot_object.axes[ax_name], plot_object=plot)
        plt.sca(plot_object.axes[ax_name])
        if xlabel is not None:
            plot_object.subplots[name].axis.set_xlabel(xlabel)
        if ylabel is not None:
            plot_object.subplots[name].axis.set_ylabel(ylabel)
        if draw_every is not None:
            _draw_counters[fig, name] = Checkpoints(draw_every)

        if grid:
            plt.grid()

    plot = plot_object.subplots[name].plot_object
    if reset_color_cycle:
        use_dbplot_axis(axis, fig=fig, clear=False).set_color_cycle(None)

    plot.update(data)

    # Update Labels...
    if cornertext is not None:
        if not hasattr(plot_object.figure, '__cornertext'):
            plot_object.figure.__cornertext = next(iter(plot_object.subplots.values())).axis.annotate(cornertext, xy=(0, 0), xytext=(0.01, 0.98), textcoords='figure fraction')
        else:
            plot_object.figure.__cornertext.set_text(cornertext)
    if title is not None:
        plot_object.subplots[name].axis.set_title(title)
    if legend is not None:
        plot_object.subplots[name].axis.legend(legend, loc='best', framealpha=0.5)

    if draw_now and not _hold_plots and (draw_every is None or ((fig, name) not in _draw_counters) or _draw_counters[fig, name]()):
        plot.plot()
        display_figure(plot_object.figure, hang=hang)

    return plot_object.subplots[name].axis


_PlotWindow = namedtuple('PlotWindow', ['figure', 'subplots', 'axes'])

_Subplot = namedtuple('Subplot', ['axis', 'plot_object'])

_DBPLOT_FIGURES = {}  # An dict<figure_name: _PlotWindow(figure, OrderedDict<subplot_name:_Subplot>)>

_DEFAULT_SIZE = get_artemis_config_value(section='plotting', option='default_fig_size', default_generator=lambda: (10, 8), write_default=True, read_method='eval')

_draw_counters = {}

_hold_plots = False

_hold_plot_counter = None

_default_layout = 'grid'


class DBPlotTypes:
    LINE= LinePlot
    THICK_LINE= partial(LinePlot, plot_kwargs={'linewidth': 3})
    POS_LINE= partial(LinePlot, y_bounds=(0, None), y_bound_extend=(0, 0.05))
    SCATTER= partial(LinePlot, plot_kwargs=dict(marker='.', markersize=7), linestyle='')
    BBOX= partial(BoundingBoxPlot, linewidth=2, axes_update_mode='expand')
    BBOX_R= partial(BoundingBoxPlot, linewidth=2, color='r', axes_update_mode='expand')
    BBOX_B= partial(BoundingBoxPlot, linewidth=2, color='b', axes_update_mode='expand')
    BBOX_G= partial(BoundingBoxPlot, linewidth=2, color='g', axes_update_mode='expand')
    BAR= BarPlot
    IMG= ImagePlot
    CIMG= partial(ImagePlot, channel_first=True)
    LINE_HISTORY= MovingPointPlot
    IMG_STABLE= partial(ImagePlot, only_grow_clims=True)
    COLOUR= partial(ImagePlot, is_colour_data=True)
    EQUAL_ASPECT= partial(ImagePlot, aspect='equal')
    IMAGE_HISTORY= MovingImagePlot
    FIXED_LINE_HISTORY= partial(MovingPointPlot, buffer_len=100)
    LINE_HISTORY_RESAMPLED= partial(ResamplingLineHistory, buffer_len=400)
    PIC= partial(ImagePlot, show_clims=False, aspect='equal')
    NOTICE= partial(TextPlot, max_history=1, horizontal_alignment='center', vertical_alignment='center', size='x-large')
    COST= partial(MovingPointPlot, y_bounds=(0, None), y_bound_extend=(0, 0.05))
    PERCENT= partial(MovingPointPlot, y_bounds=(0, 100))
    TRAJECTORY= partial(Moving2DPointPlot, axes_update_mode='expand')
    TRAJECTORY_PLUS= partial(Moving2DPointPlot, axes_update_mode='expand', x_bounds=(0, None), y_bounds=(0, None))
    HISTOGRAM= partial(HistogramPlot, edges = np.linspace(-5, 5, 20))
    CUMHIST= partial(CumulativeLineHistogram, edges = np.linspace(-5, 5, 20))

    @classmethod
    def from_string(cls, str):  # For back-compatibility
        return getattr(cls, str.upper().replace('-', '_').replace('+', '_PLUS'))


def reset_dbplot():
    if is_server_plotting_on():
        deconstruct_plotting_server()
    else:
        for fig_name, plot_window in list(_DBPLOT_FIGURES.items()):
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


def _get_dbplot_plot_object(fig):
    if isinstance(fig, plt.Figure):
        assert None not in _DBPLOT_FIGURES, "If you pass a figure, you can only do it on the first call to dbplot (for now)"
        _DBPLOT_FIGURES[None] = _PlotWindow(figure=fig, subplots=OrderedDict(), axes={})
        fig = None
    elif fig not in _DBPLOT_FIGURES or not plt.fignum_exists(_DBPLOT_FIGURES[fig].figure.number):  # Second condition handles closed figures.
        _DBPLOT_FIGURES[fig] = _PlotWindow(figure = _make_dbplot_figure(), subplots=OrderedDict(), axes = {})
        if fig is not None:
            _DBPLOT_FIGURES[fig].figure.canvas.set_window_title(fig)
    return _DBPLOT_FIGURES[fig]


def get_dbplot_subplot(name, fig_name=None):
    return _DBPLOT_FIGURES[fig_name].subplots[name].axis


def _make_dbplot_figure():

    if _DEFAULT_SIZE is None:
        fig= plt.figure()
    else:
        fig= plt.figure(figsize=_DEFAULT_SIZE)  # This is broken in matplotlib2 for some reason

    return fig


def freeze_dbplot(name, fig = None):
    del _DBPLOT_FIGURES[fig].subplots[name]


def freeze_all_dbplots(fig = None):
    for name in _DBPLOT_FIGURES[fig].subplots.keys():
        freeze_dbplot(name, fig=fig)


def dbplot_redraw_all(fig = None, hang = False):

    for subplot in _DBPLOT_FIGURES[fig].subplots.values():
        plt.subplot(subplot.axis)
        subplot.plot_object.plot()

    display_figure(_DBPLOT_FIGURES[fig].figure, hang)


def display_figure(fig, hang):
    if hang is True:
        plt.figure(fig.number)
        plt.show()
    elif hang in (None, False):
        redraw_figure(fig)
    elif isinstance(hang, (int, float)):
        redraw_figure(fig)
        plt.pause(hang)
    else:
        raise TypeError("Can't interpret hang argument {}".format(hang))


@contextmanager
def hold_dbplots(fig = None, hang=False, draw_every = None):
    """
    Use this in a "with" statement to prevent plotting until the end.
    :param fig:
    :return:
    """
    if is_server_plotting_on():
        # For now, this does nothing.  Eventually, it should be made to send a "draw" command through the pipe
        yield
        return

    global _hold_plots
    _old_hold_state = _hold_plots
    _hold_plots = True
    yield
    _hold_plots = _old_hold_state

    if _old_hold_state:
        plot_now = False
    elif draw_every is not None:
        global _hold_plot_counter
        if _hold_plot_counter is None:
            _hold_plot_counter = Checkpoints(draw_every)
        plot_now = _hold_plot_counter()
    else:
        plot_now = True

    if plot_now and fig in _DBPLOT_FIGURES:
        dbplot_redraw_all(fig, hang = hang)


def clear_dbplot(fig = None):
    if fig in _DBPLOT_FIGURES:
        plt.figure(_DBPLOT_FIGURES[fig].figure.number)
        plt.clf()
        _DBPLOT_FIGURES[fig].subplots.clear()
        _DBPLOT_FIGURES[fig].axes.clear()


def use_dbplot_axis(name, fig=None, layout=None, clear = False, ):
    ax = select_subplot(name, fig=_get_dbplot_plot_object(fig).figure, layout=_default_layout if layout is None else layout)
    if clear:
        ax.clear()
    return ax


def dbplot_hang(timeout=None):
    if timeout is None:
        plt.show()
    else:
        redraw_figure()
        plt.pause(timeout)


def dbplot_collection(collection, name, hang=False, axis = None, draw_every=None, **kwargs):
    """
    Plot a collection of items in one go.
    :param collection:
    :param name:
    :param kwargs:
    :return:
    """
    with hold_dbplots(draw_every=draw_every, hang=hang):
        if isinstance(collection, (list, tuple)):
            for i, el in enumerate(collection):
                dbplot(el, '{}[{}]'.format(name, i), axis='{}[{}]'.format(axis, i) if axis is not None else None, **kwargs)
