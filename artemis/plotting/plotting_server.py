from weakref import WeakKeyDictionary
from artemis.fileman.local_dir import get_local_path
import SimpleHTTPServer
import SocketServer
import os
from shutil import copyfile

from matplotlib import pyplot as plt


import numpy as np

__author__ = 'peter'


_CURRENT_FIGURE = None

_FIG_DIRS = WeakKeyDictionary()


def update_figure(fig=None):
    if fig is None:
        plt.savefig('artemis_figure.png')
    else:
        fig.savefig('artemis_figure.png')


def get_figure_file(fig):
    if fig in _FIG_DIRS:
        return _FIG_DIRS[fig]
    else:
        new_plot_dir = get_local_path('.live_plots/%R%R%R%R%R%R.png', make_local_dir=True)
        _FIG_DIRS[fig] = new_plot_dir
        return new_plot_dir


def save_fig(fig):
    file = get_figure_file(fig)
    fig.savefig(file)


def make_plot_file_if_necessary():
    dest = get_local_path('.live_plots/index.html', make_local_dir=True)
    if not os.path.exists(dest):
        plot_html = os.path.join(os.path.dirname(__file__), 'artemis_plots.html')
        copyfile(plot_html, dest)
    return dest


def start_plotting_server(port = 8000):
    #
    # plot_html = os.path.join(os.path.dirname(__file__), 'artemis_plots.html')
    # dest = get_local_path('.live_plots/index.html')

    import thread

    dest = make_plot_file_if_necessary()
    plot_directory, _ = os.path.split(dest)
    os.chdir(plot_directory)
    Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("", port), Handler)
    print "serving at port", port
    thread.start_new(httpd.serve_forever, ())


if __name__ == '__main__':
    start_plotting_server()
