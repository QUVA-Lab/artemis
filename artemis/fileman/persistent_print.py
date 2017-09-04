import sys
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.display import CaptureStdOut

__author__ = 'peter'

"""
Save Print statements:

Useful in ipython notebooks where you lose output when printing to the browser.

On advice from:
http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

** Note this is no longer being used.  Possibly delete
"""


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


def capture_print(log_file_path = 'logs/dump/%T-log.txt', print_to_console=True):
    """
    :param log_file_path: Path of file to print to, if (state and to_file).  If path does not start with a "/", it will
        be relative to the data directory.  You can use placeholders such as %T, %R, ... in the path name (see format
        filename)
    :param print_to_console:
    :param print_to_console: Also continue printing to console.
    :return: The absolute path to the log file.
    """
    local_log_file_path = get_artemis_data_path(log_file_path)
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
    print(read_print())
    sys.stdout = current_stdout
