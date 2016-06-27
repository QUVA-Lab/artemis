from tempfile import gettempdir
import time
import thread
import SimpleHTTPServer
import SocketServer
from artemis.plotting.manage_plotting import set_show_callback, set_draw_callback
import os
from matplotlib import pyplot as plt
import logging
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')

__author__ = 'peter'


def _make_plot_html(plot_directory, update_period=1.):
    """
    Generate the html file which will show the plots.
    :param update_period: The period of the update, in seconds.
    :return:
    """

    with open(os.path.join(os.path.dirname(__file__), 'artemis_plots.html')) as f:
        html_template = f.read()

    raw_html = html_template.replace('{update_period}', '%s' % update_period)

    # dest = get_local_path(os.path.join(plot_directory, 'index.html'), make_local_dir=True)
    dest = os.path.join(plot_directory, 'index.html')
    with open(dest, 'w') as f:
        f.write(raw_html)
    return dest


class QuietHTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    def log_request(self, code='-', size='-'):
        pass  # Overrides: # self.log_message('"%s" %s %s', self.requestline, str(code), str(size))


def _launch_on_first_available_port(first_port):
    Handler = QuietHTTPRequestHandler
    port = first_port
    while True:
        try:
            httpd = SocketServer.TCPServer(('', port), Handler)
            # print 'Serving on port', port
            ARTEMIS_LOGGER.warn("Serving Plots at http://localhost:%s" % (port, ))
            httpd.serve_forever()
        except SocketServer.socket.error as exc:
            if exc.args[0] != 48:
                raise
            ARTEMIS_LOGGER.info('Port', port, 'already in use')
            port += 1
        else:
            break


def _start_plotting_server(plot_directory, port = 8000, update_period=1.):
    dest = _make_plot_html(plot_directory, update_period=update_period)
    plot_directory, _ = os.path.split(dest)
    os.chdir(plot_directory)
    thread.start_new(_launch_on_first_available_port, (port, ))


class TimedFigureSaver(object):

    def __init__(self, fig_loc, update_period = 1):
        self.fig_loc = fig_loc
        self._last_update = -float('inf')
        self.update_period = update_period

    def __call__(self, fig=None, block=False):
        current_time = time.time()
        if current_time - self._last_update > self.update_period:
            if fig is None:
                plt.savefig(self.fig_loc)
            else:
                fig.savefig(self.fig_loc)
            self._last_update = current_time


def setup_web_plotting(update_period = 1.):
    plot_directory = gettempdir()  # Temporary directory
    _start_plotting_server(plot_directory=plot_directory, update_period=update_period)
    set_draw_callback(TimedFigureSaver(os.path.join(plot_directory, 'artemis_figure.png'), update_period=update_period))
    set_show_callback(TimedFigureSaver(os.path.join(plot_directory, 'artemis_figure.png'), update_period=update_period))
