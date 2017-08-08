import sys
import textwrap
from StringIO import StringIO
from contextlib import contextmanager

from artemis.fileman.local_dir import make_file_dir
from artemis.general.should_be_builtins import izip_equal
import numpy as np

__author__ = 'peter'


def arraystr(arr, print_threshold, summary_threshold):
    """
    :param arr: A string summary of an array.
    :param print_threshold:
    :param summary_threshold:
    :return:
    """
    if arr.size<print_threshold:
        return '<{type} with shape={shape}, dtype={dtype}, value={value}, at {id}>'.format(
            type=type(arr).__name__, shape=arr.shape, dtype=arr.dtype, value=str(arr).replace('\n', ','), id=hex(id(arr)))
    elif arr.size<summary_threshold:
        return '<{type} with shape={shape}, dtype={dtype}, in=[{min:.3g}, {max:.3g}], at {id}>'.format(
            type=type(arr).__name__, shape=arr.shape, dtype=arr.dtype, min = arr.min(), max=arr.max(), id=hex(id(arr)))
    else:
        return '<{type} with shape={shape}, dtype={dtype}, at {id}>'.format(
            type=type(arr).__name__, shape=arr.shape, dtype=arr.dtype, id=hex(id(arr)))


@contextmanager
def hold_numpy_printoptions(**kwargs):
    """
    Temporarily set the numpy print options.
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    :param kwargs: Print options (see link)
    """
    opts = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**opts)


def str_with_arrayopts(obj, float_format='.3g', threshold=8, **kwargs):
    """
    Print
    :param obj:
    :param float_format:
    :param threshold:
    :param kwargs:
    :return:
    """
    with hold_numpy_printoptions(formatter = {'float': float_format}, threshold=threshold, **kwargs):
        return str(obj)


def deepstr(obj, memo=None, array_print_threhold = 8, array_summary_threshold=10000, indent ='  ', float_format = ''):
    """
    A recursive, readable print of a data structure.
    """
    if memo is None:
        memo = set()

    if id(obj) in memo:
        return "<{type} at {loc}> already listed".format(type=type(obj).__name__, loc=hex(id(obj)))
    memo.add(id(obj))

    if isinstance(obj, np.ndarray):
        string_desc = arraystr(obj, print_threshold=array_print_threhold, summary_threshold=array_summary_threshold)
    elif isinstance(obj, (list, tuple, set, dict)):
        kwargs = dict(memo=memo, array_print_threhold=array_print_threhold, array_summary_threshold=array_summary_threshold, indent=indent, float_format=float_format)

        if isinstance(obj, (list, tuple)):
            keys, values = [str(i) for i in xrange(len(obj))], obj
        elif isinstance(obj, dict):
            keys, values = obj.keys(), obj.values()
        elif isinstance(obj, set):
            keys, values = ['- ']*len(obj), obj
        else:
            raise Exception('Should never be here')

        max_indent = max(len(k) for k in keys)

        elements = ['{k}: {v}'.format(k=k, v=' '*(max_indent-len(k)) + indent_string(deepstr(v, **kwargs), indent=' '*max_indent, include_first=False)) for k, v in izip_equal(keys, values)]
        string_desc = '<{type} at {id}>\n'.format(type = type(obj).__name__, id=hex(id(obj))) + indent_string('\n'.join(elements), indent=indent)
        return string_desc
    elif isinstance(obj, float):
        string_desc = '{{:{}}}'.format(float_format).format(obj)
    else:
        string_desc = str(obj)
    return string_desc


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


def indent_string(str, indent = '  ', include_first = True):
    if include_first:
        return indent + str.replace('\n', '\n'+indent)
    else:
        return str.replace('\n', '\n'+indent)


class IndentPrint(object):
    """
    Indent all print statements
    """

    def __init__(self, block_header=None, spacing = 4, show_line = False, show_end = False):
        self.indent = '|'+' '*(spacing-1) if show_line else ' '*spacing
        self.show_end = show_end
        self.block_header = block_header

    def __enter__(self):
        self.old_stdout = sys.stdout
        if self.block_header is not None:
            print self.block_header
        sys.stdout = self

    def flush(self):
        self.old_stdout.flush()

    def write(self, message):
        if message=='\n':
            new_message = '\n'
        else:
            new_message = indent_string(message, self.indent)
        self.old_stdout.write(new_message)

    def close(self):
        self.old_stdout.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        if self.show_end:
            print '```'


class DocumentWrapper(textwrap.TextWrapper):

    def wrap(self, text):
        split_text = text.split('\n')
        lines = [line for para in split_text for line in textwrap.TextWrapper.wrap(self, para)]
        return lines


def side_by_side(multiline_strings, gap=4, max_linewidth=None):
    """
    Return a string that displays two multiline strings side-by-side.
    :param multiline_strings: A list of multi-line strings (ie strings with newlines)
    :param gap: The gap (in spaces) to make between side-by-side display of the strings
    :param max_linewidth: Maximum width of the strings
    :return: A single string which shows all the above strings side-by-side.
    """
    if max_linewidth is not None:
        w = DocumentWrapper(width=max_linewidth, replace_whitespace=False)
        lineses = [w.wrap(mlstring) for mlstring in multiline_strings]
    else:
        lineses = [s.split('\n') for s in multiline_strings]
    longests = [max(len(line) for line in lines) for lines in lineses]

    spacer = ' '*gap
    new_lines = []
    for i in xrange(max(len(lines) for lines in lineses)):
        line = [lines[i] if i<len(lines) else '' for lines in lineses]
        final_line = [line + ' '*(max_length-len(line)) for line, max_length in zip(line, longests)]
        new_lines.append(spacer.join(final_line))
    return '\n'.join(new_lines)


def truncate_string(string, truncation, message = ''):
    """
    Truncate a string to a given length.  Optionally add a message at the end
    explaining the truncation
    :param string: A string
    :param truncation: An int
    :param message: A message, e.g. '...<truncated>'
    :return: A new string no longer than truncation
    """
    if truncation is None:
        return string
    assert isinstance(truncation, int)
    if len(string)>truncation:
        return string[:truncation-len(message)]+message
    else:
        return string


def surround_with_header(string, width, char='-'):
    """
    Surround the string by a header.  The result has length min(len(string)+2, width)
    :param string: A string
    :param width: Width of the entire header
    :param char: Character to repeat
    :return: A header, whose length will be
    """
    left = (width-len(string)-1)/2
    right = (width-len(string)-2)/2
    return char*left+' '+string+' '+char*right


def section_with_header(header, content, width=50, top_char=None, header_char='-', bottom_char=None):

    string = '' if top_char is None else top_char*width+'\n'
    string += surround_with_header(header, width=width, char=header_char) + '\n'+content+'\n'
    if bottom_char is not None:
        string += bottom_char*width
    return string