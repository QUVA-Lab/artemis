from fileman.local_dir import get_local_path
import os
from fileman.persistent_print import capture_print, read_print, new_log_file

__author__ = 'peter'


def test_persistent_print():

    test_log_path = capture_print(to_file=True)
    print 'aaa'
    print 'bbb'
    assert read_print()  == 'aaa\nbbb\n'
    capture_print(False)

    capture_print(True)
    assert read_print() == ''
    print 'ccc'
    print 'ddd'
    assert read_print()  == 'ccc\nddd\n'

    os.remove(get_local_path(test_log_path))


def test_new_log_file():
    # Just a shorthand for persistent print.

    log_file_loc = new_log_file('dump/test_file')
    print 'eee'
    print 'fff'
    capture_print(False)

    local_log_loc = get_local_path(log_file_loc)
    with open(local_log_loc) as f:
        text = f.read()

    assert text == 'eee\nfff\n'
    os.remove(local_log_loc)


if __name__ == '__main__':

    test_persistent_print()
    test_new_log_file()