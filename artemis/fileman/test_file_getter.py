from artemis.fileman.file_getter import get_file_in_archive, hold_file_root, get_file, get_file_path
from artemis.fileman.local_dir import get_artemis_data_path
import os
from six.moves import xrange


__author__ = 'peter'


def test_unpack_tar_gz():

    with hold_file_root(get_artemis_data_path('file_getter_tests'), delete_after=True, delete_before=True):

        assert not os.path.exists(get_artemis_data_path('tests/test_tar_zip'))

        for _ in xrange(2):  # (Second time to check caching)

            local_file = get_file_in_archive(
                relative_path= 'tests/test_tar_zip',
                url = 'https://drive.google.com/uc?export=download&id=0B4IfiNtPKeSAbmp6VEVJdjdSSlE',
                subpath = 'testzip/test_file.txt'
                )

            with open(local_file) as f:
                txt = f.read()

            assert txt == 'blah blah blah'


def test_unpack_zip():

    with hold_file_root(get_artemis_data_path('file_getter_tests'), delete_after=True, delete_before=True):
        assert not os.path.exists(get_artemis_data_path('tests/test_tar_zip'))

        for _ in xrange(2):  # (Second time to check caching)

            local_file = get_file_in_archive(
                relative_path= 'tests/test_zip_zip',
                url = 'https://drive.google.com/uc?export=download&id=0B4IfiNtPKeSATWZXWjEyd1FsRG8',
                subpath = 'testzip/test_file.txt'
                )

            with open(local_file) as f:
                txt = f.read()

            assert txt == 'blah blah blah'


def test_get_unnamed_file_in_archive():
    with hold_file_root(get_artemis_data_path('file_getter_tests'), delete_after=True, delete_before=True):
        path = get_file_in_archive(url='https://drive.google.com/uc?export=download&id=0B4IfiNtPKeSATWZXWjEyd1FsRG8', subpath='testzip/test_file.txt')
        with open(path) as f:
                txt = f.read()
        assert txt == 'blah blah blah'


def test_get_file():
    with hold_file_root(get_artemis_data_path('file_getter_tests'), delete_after=True, delete_before=True):
        print('Testing get_file on unnamed file')
        path = get_file(url='https://drive.google.com/uc?export=download&id=1uC9sJ04V7VjzMj32q4-OLEnRFPvQpYtp')
        with open(path) as f:
            assert f.read()=='a,b,c'

        # Should not download this time
        path = get_file(url='https://drive.google.com/uc?export=download&id=1uC9sJ04V7VjzMj32q4-OLEnRFPvQpYtp')
        with open(path) as f:
            assert f.read()=='a,b,c'

        print('Testing get_file on named file')
        path = get_file(relative_name='my-test.txt', url='https://drive.google.com/uc?export=download&id=1uC9sJ04V7VjzMj32q4-OLEnRFPvQpYtp')
        with open(path) as f:
            assert f.read()=='a,b,c'

        # Should not download this time
        path = get_file(relative_name='my-test.txt', url='https://drive.google.com/uc?export=download&id=1uC9sJ04V7VjzMj32q4-OLEnRFPvQpYtp')
        with open(path) as f:
            assert f.read()=='a,b,c'


def test_temp_file():

    with hold_file_root(get_artemis_data_path('file_getter_tests'), delete_after=True, delete_before=True):
        file_path = get_file_path(make_folder=True)
        with open(file_path, 'w') as f:
            f.write('1,2,3')
        with open(file_path) as f:
            assert f.read() == '1,2,3'


if __name__ == '__main__':
    test_temp_file()
    test_unpack_zip()
    test_unpack_tar_gz()
    test_get_unnamed_file_in_archive()
    test_get_file()
