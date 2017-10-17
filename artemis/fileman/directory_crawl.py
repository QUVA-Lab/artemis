import os
from collections import OrderedDict

from artemis.general.should_be_builtins import izip_equal


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


class DirectoryCrawler(object):

    def __init__(self, directory, ignore_hidden=True):
        if isinstance(directory, (list, tuple)):
            directory = os.path.join(*directory)
        assert os.path.exists(directory), 'Directory "{}" does not exist.'.format(directory)
        self.directory = directory
        self._contents = None
        self.ignore_hidden = ignore_hidden

    def refresh(self):
        self._contents = None

    def listdir(self):
        if self._contents is None:
            self._contents = os.listdir(self.directory)
            if self.ignore_hidden:
                self._contents = [item for item in self._contents if not item.startswith('.')]
        return self._contents

    def __iter__(self):
        for item in self.listdir():
            yield item

    def values(self):
        for item in self:
            yield self[item]

    def items(self):
        for item, full_path in izip_equal(self, self.values()):
            yield item, full_path

    def subdirs(self):
        for item in self:
            full_path = os.path.join(self.directory, item)
            if os.path.isdir(full_path):
                yield full_path

    def __getitem__(self, item):
        full_path = os.path.join(self.directory, item)
        if os.path.isdir(full_path):
            return DirectoryCrawler(full_path, ignore_hidden=self.ignore_hidden)
        elif os.path.exists(full_path):
            return full_path
        else:
            raise Exception('No such file or directory: "{}"'.format(full_path))

    def __str__(self):
        return '{}({}, ignore_hidden={})'.format(self.__class__.__name__, self.directory, self.ignore_hidden)
