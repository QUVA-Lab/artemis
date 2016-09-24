from artemis.fileman.local_dir import make_file_dir, format_filename, get_local_path
from artemis.plotting.manage_plotting import ShowContext
import os
__author__ = 'peter'

import logging
ARTEMIS_LOGGER = logging.getLogger('artemis')
logging.basicConfig()
from matplotlib import pyplot as plt


def save_figure(fig, path, default_ext = '.pdf'):
    """
    :param fig: The figure to show
    :param path: The absolute path to the figure.
    :param default_ext: The default extension to use, if none is specified.
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

    ARTEMIS_LOGGER.info('Saved Figure: %s' % path)
    return path


def show_saved_figure(relative_loc):
    """
    Display a saved figure.

    Behaviour: this simply opens a window with the figure, and then continues executing the code.

    :param relative_loc: Relative path (within the data directory) to the figure.  Treated as an absolute path
        if it begins with "/"
    :return:
    """
    _, ext = os.path.splitext(relative_loc)
    abs_loc = get_local_path(relative_loc)
    assert os.path.exists(abs_loc), '"%s" did not exist.  That is odd.' % (abs_loc, )
    if ext in ('.jpg', '.png', '.tif'):
        try:
            from PIL import Image
            Image.open(abs_loc).show()
        except ImportError:
            ARTEMIS_LOGGER.error("Cannot display image '%s', because PIL is not installed.  Go pip install pillow to use this.  Currently it is a soft requirement.")
    else:
        import webbrowser
        webbrowser.open('file://'+abs_loc)


class SaveFiguresOnShow(ShowContext):

    def __init__(self, path, also_show=True, default_ext = '.pdf'):
        """
        :param path: The path to the figure.  If it does not start with "/", it is assumed to be relative to the Data directory.
        :param also_show: Also show the figures.
        :param default_ext: The default extension to use, if none is specified.
        """
        self._path = path
        self._default_ext = default_ext
        self._locations = []
        ShowContext.__init__(self, self.save_figure, clear_others=not also_show)

    def save_figure(self, fig=None):
        if fig is None:
            fig = plt.gcf()
        loc = save_figure(fig, self._path, default_ext=self._default_ext)
        self._locations.append(loc)

    def get_figure_locs(self):
        return list(self._locations)
