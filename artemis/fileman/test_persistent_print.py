from artemis.fileman.local_dir import get_artemis_data_path
import os
from artemis.fileman.persistent_print import capture_print, read_print, new_log_file, stop_capturing_print
from artemis.general.display import CaptureStdOut

__author__ = 'peter'


def test_proper_persistent_print_usage():
    """
    Here is the best, cleanest way to use persistent print
    :return:
    """
    print('ddd')
    with CaptureStdOut() as ps:
        print('fff')
        print('ggg')
    print('hhh')
    assert ps.read() == 'fff\nggg\n'


def test_proper_persistent_print_file_logging():

    log_file_path = get_artemis_data_path('tests/test_log.txt')
    with CaptureStdOut(log_file_path) as ps:
        print('fff')
        print('ggg')
    print('hhh')
    assert ps.read() == 'fff\nggg\n'

    # You can verify that the log has also been written.
    log_path = ps.get_log_file_path()
    with open(log_path) as f:
        txt = f.read()
    assert txt == 'fff\nggg\n'


def test_persistent_print():

    test_log_path = capture_print()
    print('aaa')
    print('bbb')
    assert read_print()  == 'aaa\nbbb\n'
    stop_capturing_print()

    capture_print()
    assert read_print() == ''
    print('ccc')
    print('ddd')
    assert read_print()  == 'ccc\nddd\n'

    os.remove(get_artemis_data_path(test_log_path))


def test_new_log_file():
    # Just a shorthand for persistent print.

    log_file_loc = new_log_file('dump/test_file')
    print('eee')
    print('fff')
    stop_capturing_print()

    local_log_loc = get_artemis_data_path(log_file_loc)
    with open(local_log_loc) as f:
        text = f.read()

    assert text == 'eee\nfff\n'
    os.remove(local_log_loc)


if __name__ == '__main__':

    test_proper_persistent_print_usage()
    test_proper_persistent_print_file_logging()
    test_persistent_print()
    test_new_log_file()