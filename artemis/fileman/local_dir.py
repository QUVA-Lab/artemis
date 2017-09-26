import datetime
from artemis.fileman.config_files import get_config_value
from artemis.config import get_artemis_config_value
import os
from six.moves import xrange

__author__ = 'peter'


"""
For dealing with files in a uniform way between machines, we have a local directory for data.

The idea is that be able to put things in the code like:

mnist = pkl.read(open('data/mnist.pkl'))

Where the path is referenced relative to the data directory on that machine.
"""


def get_default_local_path():
    return os.path.join(os.getenv("HOME"), '.artemis')

LOCAL_DIR = get_artemis_config_value(section='fileman', option='data_dir', default_generator = get_default_local_path, write_default = True)


def get_artemis_data_path(relative_path ='', make_local_dir = False):
    """
    Get the full local path of a file relative to the Data folder.  If the relative path starts with a "/", we consider
    it to be a local path already.  TODO: Make this Windows-friendly

    :param relative_path: A path relative to the data directory.  If it starts with "/", we consider it to be already
    :param make_local_dir: True to create the directory that the path points to, if it does not already exist.
    :return: The full path to the file
    """
    if not relative_path.startswith('/'):
        # Path is considered relative to data directory.
        file_path = os.path.join(LOCAL_DIR, format_filename(relative_path))
    else:
        file_path = relative_path
    if make_local_dir:
        make_file_dir(file_path)
    return file_path


get_local_path = get_artemis_data_path  # Deprecated: Here for backwards compatibility


def get_artemis_data_subdir(relative_path ='', create_dir_if_not_exist=True):
    """
    Get the full path of a directory relative to the Data folder. If the relative path starts with a "/", we consider
    it to be a local path already.
    :param relative_path: A path relative to the data directory.  If it starts with "/", we consider it to be already a local path
    :param create_dir_if_not_exist: True to create the directory that the path points to, if it does not already exist.
    :return: The full path to the directory
    """
    if not relative_path.endswith("/"):
        relative_path += "/"
    return get_artemis_data_path(relative_path=relative_path, make_local_dir=create_dir_if_not_exist)


get_local_dir = get_artemis_data_subdir  # Backwards compatibility


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
    return full_dir


_ALPHANUMERICS = [chr(a) for a in xrange(ord('a'), ord('z')+1)]+[chr(a) for a in xrange(ord('0'), ord('9')+1)]


def format_filename(file_string, current_time = 'now', base_name = None, directory = None, ext = None, allow_partial_formatting = False):
    """
    Return a formatted string with placeholders in the filestring replaced by their provided values.
    :param file_string: A string, eg '%T-%N'.  The placeholders %T, %N indicate that they should be replaced
        with the time and the provided name, respectively.
        The following placeholders can be used:
            %T: Replace with the time (will look like e.g. 2016.05.20T04.23.53.145988
            %R: Replace with a random alphanumeric character 'a'-'z', '0'-'9'
    :param current_time: Current time, as returned by datetime.datetime.now() - the ISO representation of this
        time will be used to fill the %T placeholder
    :param base_name: The name to swap in for the %N placeholder
    :param rel_dir: Optionally, a directory to prepend to the file_string, relative to the local storage folder (if any).
        This can also contain placeholders
    :param local__dir: Optionally, the local folder on this machine where you store your data.  This defaults to the
        directory returned by get_artemis_data_path
    :param ext: Optionally, an extension to append to the filename.
    :param allow_partial_formatting: If True, the placeholders (%T, %N) are allowed to pass through if no values are
        specified.  This may be useful if one placeholder is not yet known at the time the rest of the info is specified.
    :return: The formatted filename.
    """

    if directory is not None:
        file_string = os.path.join(directory, file_string)

    if ext is not None:
        file_string += '.'+ext

    if '%T' in file_string:
        if current_time == 'now':
            current_time = datetime.datetime.now()
        if current_time is not None:
            iso_time = current_time.isoformat().replace(':', '.').replace('-', '.')
            file_string = file_string.replace('%T', iso_time)
        else:
            raise Exception('You failed to specify a valid time, or string "now"')
    if '%R' in file_string:
        from random import Random
        rng = Random()
        while '%R' in file_string:
            file_string = file_string.replace('%R', rng.choice(_ALPHANUMERICS), 1)

    if '%N' in file_string:
        if base_name is None:
            assert allow_partial_formatting, 'You included "%N" in the file string but had no base_name argument'
        else:
            file_string = file_string.replace('%N', base_name)

    return file_string
