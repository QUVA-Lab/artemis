from collections import OrderedDict
from contextlib import contextmanager

import itertools
import matplotlib.pyplot as plt
from artemis.general.should_be_builtins import bad_value, ceildiv, izip_equal
from artemis.plotting.data_conversion import vector_length_to_tile_dims

__author__ = 'peter'


_newplot_settings = dict(
    layout='grid',
    grid = False,
    show_x = True,
    show_y = True,
    xscale = None,
    yscale = None,
    sharex = None,
    sharey = None,
    xlabel = None,
    ylabel = None,
    xlim = None,
    ylim = None,
    )


def _subplot_loc_to_subplot_index(loc, size):
    row, col = loc
    n_rows, n_cols = size
    return 1+n_cols*row + col


def get_new_positions(fixed_positions, layout, n_plots):
    """
    :param fixed_positions: A dict of {index: position}
    :param layout:
    :return: A list of (row, column) positions.
    """
    n_rows_min = (1 if len(fixed_positions)==0 else max(r+1 for r, _ in fixed_positions.values()))
    n_cols_min = (1 if len(fixed_positions)==0 else max(c+1 for _, c in fixed_positions.values()))

    # if layout in ('h', 'horizontal'):
    #     n_rows = n_rows_min
    #     n_cols = max(ceildiv(n_plots, n_rows), n_rows_min)
    # elif layout in ('v', 'vertical'):
    #     n_cols = n_cols_min
    #     n_rows = max(ceildiv(n_plots, n_cols), n_cols_min)
    # elif layout in ('g', 'grid'):
    #
    #
    #     n_rows_, n_cols_ = vector_length_to_tile_dims(n_plots)
    #     n_rows, n_cols = max(n_rows_min, n_rows_), max(n_cols_min, n_cols_)
    # else:
    #     raise NotImplementedError(layout)
    #
    n_rows, n_cols = n_rows_min, n_cols_min
    while n_plots>n_rows*n_cols:
        if layout in ('g', 'grid'):
            if n_cols>n_rows:
                n_rows+=1
            else:
                n_cols+=1
        elif layout in ('h', 'horizontal'):
            n_cols+=1
        elif layout in ('v', 'vertical'):
            n_rows +=1
        else:
            raise NotImplementedError(layout)

    positions = []
    ix = 0
    taken_positions = set(fixed_positions.values())
    for i in range(n_plots):
        if i in fixed_positions:
            positions.append(fixed_positions[i])
        else:
            while True:
                row, col = ix/n_cols, ix%n_cols
                if (row, col) not in taken_positions:
                    positions.append((row, col))
                    taken_positions.add((row, col))
                    i+=1
                    break
                ix+=1


    return positions, (n_rows, n_cols)


_FIXED_POSITIONS = {}


def _create_subplot(fig = None, layout = None, position = None, **subplot_args):

    if layout is None:
        layout = _newplot_settings['layout']
    if fig is None:
        fig = plt.gcf()
    n = len(fig.axes)

    if position is not None:
        assert len(position)==2, 'Position must be a row, col'
        _FIXED_POSITIONS[n] = position
        # print _FIXED_POSITIONS
    # print _FIXED_POSITIONS

    positions, (n_rows, n_cols) = get_new_positions(fixed_positions=_FIXED_POSITIONS, layout=layout, n_plots=n+1)

    for i, (row, col) in enumerate(positions[:-1]):
        fig.axes[i].change_geometry(n_rows, n_cols, row*n_cols+col+1)

    new_row, new_col = positions[-1]
    ax = fig.add_subplot(n_rows, n_cols, new_row*n_cols + new_col+1, **subplot_args)

    # if position is None:
    #     if len(fig.axes)==0:
    #         position = (1, 1)
    #     else:
    #         (last_row, last_col) = fig.axes[-1]._position
    #         position = \
    #             last_row+1
    #
    # n_rows, n_cols = (1, n+1) if layout in ('h', 'horizontal') else (n+1, 1) if layout in ('v', 'vertical') else \
    #     vector_length_to_tile_dims(n+1) if layout in ('g', 'grid') else bad_value(layout)
    #
    # for i in range(n):
    #     fig.axes[i].change_geometry(n_rows, n_cols, i+1)
    #
    #



    for arg in ('sharex', 'sharey'):
        if isinstance(_newplot_settings[arg], plt.Axes):
            subplot_args[arg]=_newplot_settings[arg]
    #
    # ax = fig.add_subplot(n_rows, n_cols, n+1, **subplot_args)

    if _newplot_settings['xlabel'] is not None:
        ax.set_xlabel(_newplot_settings['xlabel'])
    if _newplot_settings['ylabel'] is not None:
        ax.set_ylabel(_newplot_settings['ylabel'])

    if _newplot_settings['xlim'] is not None:
        ax.set_xlim(_newplot_settings['xlim'])
    if _newplot_settings['ylim'] is not None:
        ax.set_ylim(_newplot_settings['ylim'])

    if _newplot_settings['grid']:
        plt.grid()

    for arg in ('sharex', 'sharey'):
        if _newplot_settings[arg] is True:
            _newplot_settings[arg]=ax

    if not _newplot_settings['show_x']:
        ax.tick_params(axis='x', labelbottom='off')
        # ax.get_xaxis().set_visible(False)
    if not _newplot_settings['show_y']:
        ax.tick_params(axis='y', labelleft='off')
        # ax.get_yaxis().set_visible(False)
    return ax



_subplots = OrderedDict()

_plot_name_generator = ('_autogenerated_subplot_{}'.format(i) for i in itertools.count(0))


def select_subplot(name=None, fig=None, layout=None, position=None, **subplot_args):
    """
    Set the current axes.  If "name" has been defined, just return that axes, otherwise make a new one.

    :param name: The name of the subplot
    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: An axes object
    """

    if fig is None:
        fig = plt.gcf()

    if name is None:
        name = next(_plot_name_generator) if position is None else position

    if name in _subplots and fig is _subplots[name].get_figure():
        # (subplot has been created) and (figure containing it has not been closed)
        plt.subplot(_subplots[name])
    else:
        _subplots[name] = _create_subplot(fig=fig, layout=layout, position=position, **subplot_args)
    return _subplots[name]


def add_subplot(layout = None, fig = None, **subplot_args):
    """
    Add a subplot, and adjust the positions of the other subplots appropriately.
    Lifted from this answer: http://stackoverflow.com/a/29962074/851699

    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: A new axes object
    """
    return select_subplot(name=None, fig=fig, layout=layout, **subplot_args)


def subplot_at(row, col):
    """
    Create or select a the subplot at position (row, col)
    :param row: The row
    :param col: The column
    :return: An axes object
    """
    return select_subplot(position=(row, col))


@contextmanager
def _define_plot_settings(**settings):
    global _newplot_settings
    old_settings = _newplot_settings
    _newplot_settings = _newplot_settings.copy()
    _newplot_settings.update(settings)
    yield
    _newplot_settings = old_settings


class CaptureNewSubplots(object):

    def __enter__(self):
        self._old_subplots = _subplots.copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.new_subplots = OrderedDict((s, p) for s, p in _subplots.items() if s not in self._old_subplots)

    def get_new_subplots(self):
        return self.new_subplots


@contextmanager
def hstack_plots(spacing=0):

    with CaptureNewSubplots() as cap:
        with _define_plot_settings(layout='h', show_y = False):
            plt.subplots_adjust(wspace=spacing)
            yield
    new_subplots = cap.get_new_subplots().values()
    new_subplots[0].get_yaxis().set_visible(True)
    for ax in new_subplots[:-1]:
        ax.set_xticks(ax.get_xticks()[:-1])


def set_same_xlims(axes):
    xmins, xmaxs = zip(*[ax.get_xlim() for ax in axes])
    x_range = min(xmins), max(xmaxs)
    for ax in axes:
        ax.set_ylim(x_range)


def set_same_ylims(axes):
    ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes])
    y_range = min(ymins), max(ymaxs)
    for ax in axes:
        ax.set_ylim(y_range)


def set_figure_border_size(size=0.05):
    plt.subplots_adjust(left=size, right=1.-size, top=1-size, bottom=size)

@contextmanager
def hstack_plots(spacing=0, sharex=False, sharey = True, grid=False, show_x=True, show_y='once', clip_x=False, clip_y=False, remove_ticks = True, xlabel=None, ylabel=None, xlim=None, ylim=None, **adjust_kwargs):

    with CaptureNewSubplots() as cap:
        with _define_plot_settings(layout='h', show_y = False if show_y=='once' else show_y, show_x = show_x, grid=grid, sharex=sharex, sharey=sharey, xlabel=xlabel, xlim=xlim, ylim=ylim):
            plt.subplots_adjust(wspace=spacing, **adjust_kwargs)
            yield
    new_subplots = cap.get_new_subplots().values()

    if clip_x:
        set_same_xlims(new_subplots)
    if clip_y:
        set_same_ylims(new_subplots)

    assert len(new_subplots)>0, "No new plots have been created in this block... Why did you create the block at all?"
    if show_y in (True, 'once'):
        new_subplots[0].tick_params(axis='y', labelleft='on')
    new_subplots[0].set_ylabel(ylabel)

    if remove_ticks:
        for ax in new_subplots[:-1]:
            ax.set_xticks(ax.get_xticks()[:-1])


@contextmanager
def vstack_plots(spacing=0, sharex=True, sharey = False, show_x = 'once', show_y=True, clip_x=False, clip_y=False, grid=False, remove_ticks = True, xlabel=None, ylabel=None, xlim=None, ylim=None, **adjust_kwargs):

    with CaptureNewSubplots() as cap:
        with _define_plot_settings(layout='v', show_x = False if show_x=='once' else show_x, show_y=show_y, grid=grid, sharex=sharex, sharey=sharey, ylabel=ylabel, xlim=xlim, ylim=ylim):
            plt.subplots_adjust(hspace=spacing, **adjust_kwargs)
            yield
    new_subplots = cap.get_new_subplots().values()

    if clip_x:
        set_same_xlims(new_subplots)
    if clip_y:
        set_same_ylims(new_subplots)

    assert len(new_subplots)>0, "No new plots have been created in this block... Why did you create the block at all?"
    if show_x in (True, 'once'):
        new_subplots[-1].tick_params(axis='x', labelbottom='on')
    new_subplots[-1].set_xlabel(xlabel)

    if remove_ticks:
        new_subplots[-1].get_xaxis().set_visible(True)
        for ax in new_subplots[:-1]:
            ax.set_yticks(ax.get_yticks()[:-1])
