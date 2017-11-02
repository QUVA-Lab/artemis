import os
from collections import OrderedDict
from shutil import rmtree
from artemis.general.display import surround_with_header
from artemis.general.should_be_builtins import izip_equal
from six.moves import input


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

    def listdir(self, refresh=False, sortby=None, end_dirs_with_slash=True):
        if isinstance(sortby, str) and sortby.startswith('-'):
            sortby = sortby[1:]
            reverse = True
        else:
            reverse = False
        if refresh or self._contents is None:
            self._contents = os.listdir(self.directory)
            if sortby=='mtime':
                self._contents = sorted(self._contents, key = lambda item: os.path.getmtime(self.get_path(item)))
            elif sortby=='name':
                self._contents = sorted(self._contents)
            elif sortby is not None:
                raise AssertionError('Invalid value for sortby: {}'.format(sortby))
            if reverse:
                self._contents = self._contents[::-1]
            if end_dirs_with_slash:
                self._contents = [c+os.sep if self.isdir(c) else c for c in self._contents]
            if self.ignore_hidden:
                self._contents = [item for item in self._contents if not item.startswith('.')]
        return self._contents

    def isdir(self, item):
        return os.path.isdir(self.get_path(item))

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

    def get_path(self, item):
        return os.path.join(self.directory, item)

    def __getitem__(self, item):
        full_path = os.path.normpath(os.path.join(self.directory, item))
        if os.path.isdir(full_path):
            return DirectoryCrawler(full_path, ignore_hidden=self.ignore_hidden)
        elif os.path.exists(full_path):
            return full_path
        else:
            raise Exception('No such file or directory: "{}"'.format(full_path))

    def __str__(self):
        return '{}({}, ignore_hidden={})'.format(self.__class__.__name__, self.directory, self.ignore_hidden)


class DirectoryCrawlerUI(object):
    """
    Commands for Directory Crowler UI:

    <Enter>         Refresh
    del 1-4          Remove listed files 1-4 (will be prompted for confirmation)
    del all          Remove all listed files.
    cd 4            Go into the 4th directory.
    cd ..           Go back 1 directory
    sortby name     Sort by name
    sortby mtime    Sort by last modified time
    sortby -mtime   Sort by last modified time in descending order
    h               Print help
    q               Quit
    """

    def __init__(self, directory, show_num_items=False, sortby='name'):
        self.dc = DirectoryCrawler(directory)
        self.show_num_subs = show_num_items
        self.sortby = sortby

    def _list_directory_contents(self):
        self._files = self.dc.listdir(refresh=True, sortby=self.sortby)
        if self.show_num_subs:
            return '\n'.join('{}: {} ({} items)'.format(i, fname, len(self.dc[fname].listdir())) for i, fname in enumerate(self._files))
        else:
            return '\n'.join('{}: {}'.format(i, fname) for i, fname in enumerate(self._files))

    def launch(self):
        commands = {
            'h': self.help,
            'del': self.delete,
            'cd': self.cd,
            'sortby': self.sortby
            }

        redraw = True
        while True:
            if redraw:
                print(surround_with_header('Contents of {}'.format(self.dc.directory), width=100))
                print(self._list_directory_contents())
            redraw = True
            command = input('Enter Command, or h for help >>')
            cmd_and_args = command.split(' ')
            cmd, args = cmd_and_args[0], cmd_and_args[1:]
            if cmd=='':
                continue
            elif cmd=='q':
                break
            elif cmd not in commands:
                print('Command not found: {}'.format(cmd))
                redraw=False
            else:
                commands[cmd](*args)

    def help(self):
        print(self.__doc__)
        input('Press Enter to continue >>')

    def sortby(self, arg):
        self.sortby = arg

    def _get_paths_from_range(self, user_range):
        if user_range=='all':
            files = self._files
        elif '-' in user_range:
            start, end = user_range.split('-')
            files = self._files[int(start): int(end)+1]
        else:
            files = [self._files[int(user_range)]]
        return [self.dc.get_path(filename) for filename in files]

    def delete(self, user_range):
        paths = self._get_paths_from_range(user_range)
        response = input('\n'.join(paths) + '\n Will be deleted.  Type "yes" to confirm>>')
        if response=='yes':
            for p in paths:
                rmtree(p)

    def cd(self, rel_path):
        try:
            subpath = self._files[int(rel_path)]
        except ValueError:
            subpath = rel_path
        self.dc = self.dc[subpath]


if __name__ == '__main__':
    from artemis.fileman.local_dir import get_artemis_data_path
    dcu = DirectoryCrawlerUI(get_artemis_data_path(), show_num_items=True, sortby='mtime')
    dcu.launch()
