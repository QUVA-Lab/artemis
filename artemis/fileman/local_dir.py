import sys

import datetime

import os

__author__ = 'peter'


"""
For dealing with files in a uniform way between machines, we have a local directory for data.

The idea is that be able to put things in the code like:

mnist = pkl.read(open('data/mnist.pkl'))

Where the path is referenced relative to the data directory on that machine.
"""

LOCAL_DIR = \
    os.path.join(os.getenv("HOME"), 'Library', 'Application Support', 'Plato') if sys.platform == 'darwin' else \
    os.path.join(os.getenv("HOME"), '.PlatoData')


def get_local_path(relative_path = '', make_local_dir = False):
    file_path = os.path.join(LOCAL_DIR, format_filename(relative_path))
    if make_local_dir:
        make_file_dir(file_path)
    return file_path


def get_relative_path(local_path, base_path = LOCAL_DIR):
    assert local_path.startswith(base_path), '"%s" is not contained within the data directory "%s"' % (local_path, base_path)
    relative_loc = local_path[len(base_path)+1:]
    return relative_loc


def make_file_dir(full_file_path):
    """
    Make the directory containing the file in the given path, if it doesn't already exist
    :param full_file_path:
    :returns: The directory.
    """
    full_local_dir, _ = os.path.split(full_file_path)
    try:
        os.makedirs(full_local_dir)
    except OSError:
        pass
    return full_file_path


def make_dir(full_dir):

    try:
        os.makedirs(full_dir)
    except OSError:
        pass


def format_filename(file_string, current_time = 'now', base_name = None, directory = None, ext = None, allow_partial_formatting = False):
    """
    Return a formatted string with placeholders in the filestring replaced by their provided values.
    :param file_string: A string, eg '%T-%N'.  The placeholders %T, %N indicate that they should be replaced
        with the time and the provided name, respectively
    :param current_time: Current time, as returned by datetime.datetime.now() - the ISO representation of this
        time will be used to fill the %T placeholder
    :param base_name: The name to swap in for the %N placeholder
    :param rel_dir: Optionally, a directory to prepend to the file_string, relative to the local storage folder (if any).
        This can also contain placeholders
    :param local__dir: Optionally, the local folder on this machine where you store your data.  This defaults to the
        directory returned by get_local_path
    :param ext: Optionally, an extension to append to the filename.
    :param allow_partial_formatting: If True, the placeholders (%T, %N) are allowed to pass through if no values are
        specified.  This may be useful if one placeholder is not yet known at the time the rest of the info is specified.
    :return: The formatted filename.
    """

    if directory is not None:
        file_string = os.path.join(directory, file_string)

    if ext is not None:
        file_string += '.'+ext

    if current_time == 'now':
        current_time = datetime.datetime.now()
    if current_time is not None:
        iso_time = current_time.isoformat().replace(':', '.').replace('-', '.')
        file_string = file_string.replace('%T', iso_time)
    elif not allow_partial_formatting:
        assert '%T' not in file_string

    if base_name is not None:
        file_string = file_string.replace('%N', base_name)
    elif not allow_partial_formatting:
        assert '%N' not in file_string

    return file_string
