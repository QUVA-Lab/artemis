from functools import partial
from artemis.fileman.local_dir import make_file_dir
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

    path, ext = os.path.splitext(path)
    if ext == '':
        path += default_ext

    if '%L' in path:
        path = path.replace('%L', fig.get_label() if fig.get_label() is not '' else 'unnamed')

    make_file_dir(path)
    fig.savefig(path)

    if print_loc:
        ARTEMIS_LOGGER.warn('Saved Figure: %s' % path)
    else:
        ARTEMIS_LOGGER.info('Saved Figure: %s' % path)


class SaveFiguresOnShow(ShowContext):

    def __init__(self, also_show,  **kwargs):
        """
        :param path: The path to the figure.  If it does not start with "/", it is assumed to be relative to the Data directory.
        :param default_ext: The default extension to use, if none is specified.
        :param print_loc: Print the location when you save it.
        """
        ShowContext.__init__(self, partial(save_figure, **kwargs), clear_others=not also_show)
