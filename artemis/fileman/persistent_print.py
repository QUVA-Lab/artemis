from datetime import datetime
import sys
from StringIO import StringIO
from artemis.fileman.local_dir import get_local_path, make_file_dir


__author__ = 'peter'

"""
Save Print statements:

Useful in ipython notebooks where you lose output when printing to the browser.

On advice from:
http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class CaptureStdOut(object):
    """
    An logger that both prints to stdout and writes to file.
    """

    def __init__(self, log_file_path = None, print_to_console = True):
        """
        :param log_file_path: The path to save the records, or None if you just want to keep it in memory
        :param print_to_console:
        """
        self._print_to_console = print_to_console
        if log_file_path is not None:
            # self._log_file_path = os.path.join(base_dir, log_file_path.replace('%T', now))
            make_file_dir(log_file_path)
            self.log = open(log_file_path, 'w')
        else:
            self.log = StringIO()
        self._log_file_path = log_file_path
        self.terminal = _ORIGINAL_STDOUT

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        self.close()

    def get_log_file_path(self):
        assert self._log_file_path is not None, "You never specified a path when you created this logger, so don't come back and ask for one now"
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


def capture_print(log_file_path = 'logs/dump/%T-log.txt', print_to_console=True):
    """
    :param log_file_path: Path of file to print to, if (state and to_file).  If path does not start with a "/", it will
        be relative to the data directory.  You can use placeholders such as %T, %R, ... in the path name (see format
        filename)
    :param print_to_console:
    :param print_to_console: Also continue printing to console.
    :return: The absolute path to the log file.
    """
    local_log_file_path = get_local_path(log_file_path)
    logger = CaptureStdOut(log_file_path=local_log_file_path, print_to_console=print_to_console)
    logger.__enter__()
    sys.stdout = logger
    sys.stderr = logger
    return local_log_file_path


def stop_capturing_print():
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR


def new_log_file(log_file_path = 'dump/%T-log', print_to_console = False):
    """
    Just capture-print with different defaults - intended to be called from notebooks where
    you don't want all output printed, but want to be able to see it with a link.
    :param log_file_path: Path to the log file - %T is replaced with time
    :param print_to_console: True to continue printing to console
    """
    return capture_print(log_file_path=log_file_path, print_to_console=print_to_console)


def read_print():
    return sys.stdout.read()


def reprint():
    assert isinstance(sys.stdout, CaptureStdOut), "Can't call reprint unless you've turned on capture_print"
    # Need to avoid exponentially growing prints...
    current_stdout = sys.stdout
    sys.stdout = _ORIGINAL_STDOUT
    print read_print()
    sys.stdout = current_stdout
