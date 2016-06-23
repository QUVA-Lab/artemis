from functools import partial
from artemis.fileman.local_dir import make_file_dir, format_filename
from artemis.plotting.manage_plotting import ShowContext
import os
__author__ = 'peter'

import logging
ARTEMIS_LOGGER = logging.getLogger('artemis')
logging.basicConfig()


def save_figure(fig, path, default_ext = '.pdf', print_loc = True):
    """
    :param fig: The figure to show
    :param path: The path to the figure.  If it does not start with "/", it is assumed to be relative to the Data directory.
    :param default_ext: The default extension to use, if none is specified.
    :param print_loc: Print the location when you save it.
    :return:
    """

    _, ext = os.path.splitext(path)
    if ext == '':
        path += default_ext

    if '%L' in path:
        path = path.replace('%L', fig.get_label() if fig.get_label() is not '' else 'unnamed')
    path = format_filename(path)

    make_file_dir(path)
    fig.savefig(path)

    if print_loc:
        ARTEMIS_LOGGER.warn('Saved Figure: %s' % path)
    else:
        ARTEMIS_LOGGER.info('Saved Figure: %s' % path)
    return path


class SaveFiguresOnShow(ShowContext):

    def __init__(self, path, also_show=True, default_ext = '.pdf', print_loc = True):
        """
        :param path: The path to the figure.  If it does not start with "/", it is assumed to be relative to the Data directory.
        :param also_show: Also show the figures.
        :param default_ext: The default extension to use, if none is specified.
        :param print_loc: Print the location when you save it.
        """
        self._path = path
        self._default_ext = default_ext
        self._print_loc = print_loc
        self._locations = []
        ShowContext.__init__(self, self.save_figure, clear_others=not also_show)

    def save_figure(self, fig):
        loc = save_figure(fig, self._path, default_ext=self._default_ext, print_loc = self._print_loc)
        self._locations.append(loc)

    def get_figure_locs(self):
        return list(self._locations)
