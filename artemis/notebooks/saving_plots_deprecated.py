from datetime import datetime
from functools import partial
import pickle
import subprocess
from artemis.fileman.local_dir import get_local_path, make_file_dir
from artemis.plotting.manage_plotting import ShowContext
import os
from matplotlib import pyplot as plt


"""
This module is deprecated.  It will soon be deleted.

"""

__author__ = 'peter'


_ORIGINAL_SHOW_FCN = plt.show
_ORIGINAL_DRAW_FCN = plt.draw


_SAVED_FIGURES = []


def get_local_figures_dir(subdir = None):
    figures_dir = get_local_path('figures')
    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)
    return figures_dir


def save_and_show(fig = None, name = '%T-%N', ext = 'pdf', base_dir = 'figures',
        subdir = 'dump', block = None, print_loc = True, show = True):
    """
    Save and show a figure.
    :param fig: The figure in question (or None to just get the current figure)
    :param name: The base-name to save it under.  If you put %T in, it will be replaced with the date/time.  E.g.
        '%T-test_fig.pdf' becomes '2015.03.20T13.35.00.068260-test_fig.pdf'
    :param ext: The file format, if not specified in the name\
    :param base_dir: The root directory for figures
    :param subdir: The subdirectory in which to save this figure.  You can also put %T in this to replace with current time.
    :param block: Should the desiplay block execution?  (If None, just go with current interactivemode setting)
    :param print_loc: Print the save location?
    :param show: Actually show the figure?
    :return: The relative file-path of the saved figure.
    """

    if fig is None:
        fig = plt.gcf()

    fig_name = fig.get_label() if fig.get_label() is not '' else 'unnamed'

    now = datetime.now().isoformat().replace(':', '.').replace('-', '.')
    subdir = subdir.replace('%T', now)
    name = name.replace('%T', now) + '.'+ext
    name = name.replace('%N', fig_name)

    fig.canvas.set_window_title(name)

    is_interactive = plt.isinteractive()
    if block is None:
        block = not is_interactive

    figure_loc = os.path.join(base_dir, subdir, name)
    local_figure_loc = get_local_path(figure_loc)

    make_file_dir(local_figure_loc)

    fig.savefig(local_figure_loc)
    if print_loc:
        print 'Saved figure to "%s"' % (local_figure_loc, )
    _SAVED_FIGURES.append(figure_loc)  # Which is technically a memory leak, but you'd have to make a lot of figures.

    if show:
        plt.interactive(not block)
        _ORIGINAL_SHOW_FCN()  # There's an argument block, but it's not supported for all backends
        plt.interactive(is_interactive)
    else:
        plt.close()

    return figure_loc


def get_saved_figure_locs():
    return _SAVED_FIGURES


def clear_saved_figure_locs():
    global _SAVED_FIGURES
    _SAVED_FIGURES = []


def set_show_callback(cb):
    """
    :param cb: Can be:
        - A function which is called instead of plt.show()
        - None, which just returns things to the default.
    """
    _old_show_callback = plt.show
    plt.show = cb if cb is not None else _ORIGINAL_SHOW_FCN
    return _old_show_callback


def set_draw_callback(cb):
    """
    :param cb: A function that is called in place of plt.draw()
    """
    plt.draw = cb


def _get_alternate_show_callback(**default_kwargs):

    def _alternate_show_callback(fig, **kwargs):
        old_kwargs = default_kwargs.copy()
        old_kwargs.update(kwargs)
        save_and_show(fig, **old_kwargs)


def always_save_figures(state = True, **save_and_show_args):
    """
    :param state: True to save figures, False to not save them.
    :param **save_and_show_args: See "save_and_show"
    """

    if state:
        # set_show_and_hang_callback(lambda fig = None, **kwargs: save_and_show(fig, **save_and_show_args))
        set_show_callback(_get_alternate_show_callback(**save_and_show_args))
    else:
        set_show_callback(None)





def show_saved_figure(relative_loc):
    _, ext = os.path.splitext(relative_loc)
    abs_loc = get_local_path(relative_loc)
    assert os.path.exists(abs_loc), '"%s" did not exist.  That is odd.' % (abs_loc, )
    subprocess.call('open "%s"' % abs_loc, shell = True)


_SERIALIZED_FIGURES = []


class FigureCollector(object):

    def __init__(self):
        self._ser_figures = []

    def __call__(self):
        fig = plt.gcf()
        self._ser_figures.append(pickle.dumps(fig))
        plt.close(fig)

    def iter_figures(self):
        for serfig in self._ser_figures:
            yield pickle.loads(serfig)


def serialize_figure(fig):

    fig_dict = {
        'label': fig.get_label(),
        'axes': pickle.dumps(fig.get_axes()),
        'VERSION': 0
    }
    return pickle.dumps(fig_dict)


def deserialize_figure(serialized):
    fig_dict = pickle.loads(serialized)
    fig = plt.figure(fig_dict['label'])
    ax = pickle.loads(fig_dict['axes'])
    for a in ax:
        # fig.axes.append(a)
        a.set_figure(fig)
        fig.add_axes(a)
    return fig
