import os
from collections import OrderedDict


def crawl_directory(directory, ignore_hidden = True):
    """
    Given a directory, return a dict representing the tree of files under that directory.

    :param directory: A string representing a directory.
    :return: A dict<file_or_dir_name: content> where:
        file_or_dir_name is the name of the file within the parent directory.
        content is either
        - An absolute file path (for files) or
        - A dictionary containing the output of crawl_directory for a subdirectory.
    """
    contents = os.listdir(directory)
    if ignore_hidden:
        contents = [c for c in contents if not c.startswith('.')]
    this_dir = OrderedDict()
    for c in contents:
        full_path = os.path.join(directory, c)
        if os.path.isdir(full_path):
            this_dir[c] = crawl_directory(full_path)
        else:
            this_dir[c] = full_path
    return this_dir
