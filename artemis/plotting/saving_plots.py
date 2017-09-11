from contextlib import contextmanager
import pickle
from artemis.fileman.local_dir import make_file_dir, format_filename, get_artemis_data_path
from artemis.plotting.drawing_plots import redraw_figure
from artemis.plotting.manage_plotting import ShowContext
import os
__author__ = 'peter'
import logging
ARTEMIS_LOGGER = logging.getLogger('artemis')
logging.basicConfig()
from matplotlib import pyplot as plt

_supported_filetypes = ('.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz', '.tif', '.tiff', '.pkl')


def save_figure(fig, path, ext=None, default_ext = '.pdf'):
    """
    :param fig: The figure to show
    :param path: The absolute path to the figure.
    :param default_ext: The default extension to use, if none is specified.
    :return:
    """

    if ext is None:
        _, ext = os.path.splitext(path)
        if ext == '':
            path += default_ext
        else:
            assert ext in _supported_filetypes, "We inferred the extension '{}' from your filename, but it was not in the list of supported extensions: {}" \
                .format(ext, _supported_filetypes)
    else:
        path += ext if ext.startswith('.') else '.'+ext

    if '%L' in path:
        path = path.replace('%L', fig.get_label() if fig.get_label() is not '' else 'unnamed')
    path = format_filename(path)

    make_file_dir(path)
    if ext=='.pkl':
        with open(path, 'wb') as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        fig.savefig(path)

    ARTEMIS_LOGGER.info('Saved Figure: %s' % path)
    return path


@contextmanager
def interactive_matplotlib_context(on=True):
    old_mode = plt.isinteractive()
    plt.interactive(on)
    yield
    plt.interactive(old_mode)


def show_saved_figure(relative_loc, title=None):
    """
    Display a saved figure.

    Behaviour: this simply opens a window with the figure, and then continues executing the code.

    :param relative_loc: Relative path (within the data directory) to the figure.  Treated as an absolute path
        if it begins with "/"
    :return:
    """
    fig_path, ext = os.path.splitext(relative_loc)
    if title is None:
        _, title = os.path.split(fig_path)
    abs_loc = get_artemis_data_path(relative_loc)
    assert os.path.exists(abs_loc), '"%s" did not exist.  That is odd.' % (abs_loc, )
    if ext in ('.jpg', '.png', '.tif'):
        try:
            from PIL import Image
            Image.open(abs_loc).show()
        except ImportError:
            ARTEMIS_LOGGER.error("Cannot display image '%s', because PIL is not installed.  Go pip install pillow to use this.  Currently it is a soft requirement.")
    elif ext == '.pkl':
        with interactive_matplotlib_context():
            with open(abs_loc) as f:
                fig = pickle.load(f)
                fig.canvas.set_window_title(title)
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
