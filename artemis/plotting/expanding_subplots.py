import matplotlib.pyplot as plt
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.data_conversion import vector_length_to_tile_dims

__author__ = 'peter'


def add_subplot(fig = None, layout = 'grid'):
    """
    Add a subplot, and adjust the positions of the other subplots appropriately.
    Lifted from this answer: http://stackoverflow.com/a/29962074/851699

    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: A new axes object
    """
    if fig is None:
        fig = plt.gcf()
    n = len(fig.axes)
    n_rows, n_cols = (1, n+1) if layout in ('h', 'horizontal') else (n+1, 1) if layout in ('v', 'vertical') else \
        vector_length_to_tile_dims(n+1) if layout in ('g', 'grid') else bad_value(layout)
    for i in range(n):
        fig.axes[i].change_geometry(n_rows, n_cols, i+1)
    ax = fig.add_subplot(n_rows, n_cols, n+1)
    return ax


_subplots = {}


def select_subplot(name, fig=None, layout='grid'):
    """
    Set the current axes.  If "name" has been defined, just return that axes, otherwise make a new one.

    :param name: The name of the subplot
    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: An axes object
    """

    if fig is None:
        fig = plt.gcf()

    if name in _subplots and fig is _subplots[name].get_figure():
        # (subplot has been created) and (figure containing it has not been closed)
        plt.subplot(_subplots[name])
    else:
        _subplots[name] = add_subplot(fig=fig, layout=layout)
    return _subplots[name]
