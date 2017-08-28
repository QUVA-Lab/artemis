from datetime import datetime
import sys
from artemis.fileman.local_dir import get_artemis_data_path, get_relative_path
import os

__author__ = 'peter'


def get_local_server_dir(subdir = None):
    """
    Get the directory at the root of the venv.
    :param subdir:
    :return:
    """
    figures_dir = os.path.abspath(os.path.join(sys.executable, '..', '..', '..'))
    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)
    return figures_dir


DATA_FOLDER_NAME = 'Data'
SERVER_RELATIVE_DATA_DIR = os.path.join(get_local_server_dir(), DATA_FOLDER_NAME)


def get_server_relative_data_folder_name():
    return DATA_FOLDER_NAME


def get_relative_link_from_local_path(local_path):
    """
    Given an abolulute path on the local machine, return a relative link to the file so that
    you can access it from the server.
    :param local_path: A local path.
    :return: A relative path
    """
    relative_path = get_relative_path(local_path)
    return get_relative_link_from_relative_path(relative_path)


def get_relative_link_from_relative_path(relative_path):
    """
    Given a local path to a file in the data folder, return the relative link that will access it from
    the server.

    To do this, we make a soft-link from the server directory to the data folder - this way we can
    browse our data folder from Jupyter, which is nice.

    :param relative_path: Relative path (from within Data folder)
    :return: A string representing the relative link to get to that file.
    """
    true_local_data_dir = get_artemis_data_path()

    local_path = get_artemis_data_path(relative_path)
    launcher = 'tree' if os.path.isdir(local_path) else 'files'

    if not os.path.lexists(SERVER_RELATIVE_DATA_DIR):
        os.symlink(true_local_data_dir, SERVER_RELATIVE_DATA_DIR)
    return os.path.join('/', launcher, DATA_FOLDER_NAME, relative_path)



