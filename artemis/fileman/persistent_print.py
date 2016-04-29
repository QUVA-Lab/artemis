from datetime import datetime
import sys
from StringIO import StringIO
from IPython.core.display import display, HTML
from fileman.local_dir import get_local_path, make_file_dir, format_filename
from fileman.notebook_utils import get_relative_link_from_local_path, get_relative_link_from_relative_path
import os


__author__ = 'peter'

"""
Save Print statements:

Useful in ipython notebooks where you lose output when printing to the browser.

On advice from:
http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class PrintAndStoreLogger(object):
    """
    An logger that both prints to stdout and writes to file.
    """

    def __init__(self, log_file_path = None, print_to_console = True):

        self._print_to_console = print_to_console

        if log_file_path is not None:
            # self._log_file_path = os.path.join(base_dir, log_file_path.replace('%T', now))
            make_file_dir(log_file_path)
            self.log = open(log_file_path, 'w')
        else:
            self.log = StringIO()
        self._log_file_path = log_file_path
        self.terminal = _ORIGINAL_STDOUT

    def get_log_file_path(self):
        return self._log_file_path

    def write(self, message):
        if self._print_to_console:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def close(self):
        if self._log_file_path is not None:
            self.log.close()

    def read(self):
        if self._log_file_path is None:
            return self.log.getvalue()
        else:
            with open(self._log_file_path) as f:
                txt = f.read()
            return txt

    def __getattr__(self, item):
        return getattr(self.terminal, item)


def capture_print(state = True, to_file = False, log_file_path = 'dump/%T-log', **print_and_store_kwargs):
    """
    :param state: True to caputure print, False to not capture print
    :param to_file: True to print to file
    :param log_file_path: Path of file to print to, if (state and to_file)
    :param print_and_store_kwargs: Passed to the PrintAndStoreLogger constructor.
    :return: The relative path to the logger.
    """

    if state:
        rel_log_file_path = format_filename(log_file_path, current_time = datetime.now(), directory='logs', ext = 'txt')
        local_log_file_path = get_local_path(rel_log_file_path)
        logger = PrintAndStoreLogger(log_file_path=local_log_file_path, **print_and_store_kwargs)
        if to_file:
            relative_link = get_relative_link_from_relative_path(rel_log_file_path)
            log_folder_link = get_relative_link_from_relative_path('logs')
            display(HTML("Writing to <a href='%s' target='_blank'>this log file</a>.  See <a href='%s' target='_blank'>all logs</a>"
                % (relative_link, log_folder_link)))
        sys.stdout = logger

        sys.stderr = logger
        return rel_log_file_path
    else:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR




def new_log_file(log_file_path = 'dump/%T-log', print_to_console = False):
    """
    Just capture-print with different defaults - intended to be called from notebooks where
    you don't want all output printed, but want to be able to see it with a link.
    :param log_file_path: Path to the log file - %T is replaced with time
    :param print_to_console: True to continue printing to console
    """
    return capture_print(state = True, to_file=True, log_file_path=log_file_path, print_to_console=print_to_console)


def read_print():
    return sys.stdout.read()


def reprint():
    assert isinstance(sys.stdout, PrintAndStoreLogger), "Can't call reprint unless you've turned on capture_print"
    # Need to avoid exponentially growing prints...
    current_stdout = sys.stdout
    sys.stdout = _ORIGINAL_STDOUT
    print read_print()
    sys.stdout = current_stdout
