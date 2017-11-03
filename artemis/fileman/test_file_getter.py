import shutil
from artemis.fileman.file_getter import get_file_in_archive
from artemis.fileman.local_dir import get_artemis_data_path
import os
from six.moves import xrange


__author__ = 'peter'


def test_unpack_tar_gz():

    if os.path.exists(get_artemis_data_path('tests/test_tar_zip')):
        shutil.rmtree(get_artemis_data_path('tests/test_tar_zip'))
    if os.path.exists(get_artemis_data_path('tests/test_tar_zip.tar.gz')):
        os.remove(get_artemis_data_path('tests/test_tar_zip.tar.gz'))

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

    if os.path.exists(get_artemis_data_path('tests/test_zip_zip')):
        shutil.rmtree(get_artemis_data_path('tests/test_zip_zip'))
    if os.path.exists(get_artemis_data_path('tests/test_zip_zip.zip')):
        os.remove(get_artemis_data_path('tests/test_zip_zip.zip'))

    for _ in xrange(2):  # (Second time to check caching)

        local_file = get_file_in_archive(
            relative_path= 'tests/test_zip_zip',
            url = 'https://drive.google.com/uc?export=download&id=0B4IfiNtPKeSATWZXWjEyd1FsRG8',
            subpath = 'testzip/test_file.txt'
            )

        with open(local_file) as f:
            txt = f.read()

        assert txt == 'blah blah blah'


if __name__ == '__main__':
    test_unpack_zip()
    test_unpack_tar_gz()

