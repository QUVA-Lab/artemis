import hashlib
from contextlib import contextmanager
from shutil import rmtree
from six.moves import StringIO
import gzip
import tarfile
from zipfile import ZipFile
import shutil
import os
from six.moves.urllib.request import urlopen
from artemis.fileman.local_dir import get_artemis_data_path, make_file_dir
from artemis.general.should_be_builtins import bad_value

__author__ = 'peter'


FILE_ROOT = get_artemis_data_path()


def set_file_root(path, make_dir=True):
    if make_dir:
        try:  # Best way to see if folder exists already - avoids race condition between processes
            os.makedirs(path)
        except OSError:
            pass
    else:
        assert os.path.isdir(path)
    global FILE_ROOT
    FILE_ROOT = path


@contextmanager
def hold_file_root(path, make_dir = True, delete_after = False, delete_before=False):
    """
    :param path:
    :return:
    """
    old_path = FILE_ROOT
    if delete_before and os.path.exists(path):
        rmtree(path)
    set_file_root(path, make_dir=make_dir)
    yield
    set_file_root(old_path)
    if delete_after:
        rmtree(path)


def get_file(relative_name=None, url = None, use_cache = True, data_transformation = None):
    """
    Get a file and return the full local path to that file.

    :param relative_name: The name of the local file, relative to the FILE_ROOT (by default, FILE_ROOT is ~/.artemis)
        Or if None, and a URL is specified, we give the file a temporary name.
    :param url: Optionally, a url to fetch this file from if it doesn't exist locally.
    :param use_cache: If the file exists locally and a URL is specified, use the local version.
    :param data_transformation: Optionally a function that takes the downloaded data (from response.read) and outputs
        binary data that is be written into the file.
    :return:
    """

    assert (relative_name is not None) or (url is not None), 'You must provide a local name and/or a URL to fetch from.'

    full_filename = get_file_path(relative_name=relative_name, url=url)

    if (not os.path.exists(full_filename)) or (not use_cache):
        assert url is not None, "No local copy of '%s' was found, and you didn't provide a URL to fetch it from" % (full_filename, )

        print('Downloading file from url: "%s"...' % (url, ))
        response = urlopen(url)
        data = response.read()
        print('...Done.')

        if data_transformation is not None:
            print('Processing downloaded data...')
            data = data_transformation(data)
        with open(full_filename, 'wb') as f:
            f.write(data)
    return full_filename


def get_file_in_archive(subpath, url, relative_path=None, force_extract = False, use_cache=True):
    """
    Download a zip file, unpack it, and get the local address of a file within this zip (so that you can open it, etc).

    :param relative_path: Local name for the extracted folder.  (Zip file will be named this with the appropriate zip extension)
    :param url: Url of the zip file to download
    :param subpath: Path of the file relative to the zip folder.
    :param force_extract: Force the zip file to re-extract (rather than just reusing the extracted folder)
    :return: The full path to the file on your system.
    """
    local_folder_path = get_archive(relative_path=relative_path, url=url, force_extract=force_extract, use_cache=use_cache)
    local_file_path = os.path.join(local_folder_path, subpath)
    assert os.path.exists(local_file_path), 'Could not find the file "%s" within the extracted folder: "%s"' % (subpath, local_folder_path)
    return local_file_path


def get_archive(url, relative_path=None, force_extract=False, archive_type = None, use_cache=True):
    """
    Download a compressed archive and extract it into a folder.

    :param relative_path: Local name for the extracted folder.  (Zip file will be named this with the appropriate zip extension)
    :param url: Url of the archive to download
    :param force_extract: Force the zip file to re-extract (rather than just reusing the extracted folder)
    :return: The full path to the extracted folder on your system.
    """

    if relative_path is None:
        relative_path = get_unnamed_file_hash(url)

    local_folder_path, _ = os.path.splitext(os.path.join(FILE_ROOT, relative_path))

    assert archive_type in ('.tar.gz', '.zip', None)

    if (not os.path.exists(local_folder_path)) or (not use_cache):  # If the folder does not exist, download zip and extract.
        # (We also check force download here to avoid a race condition)

        if not use_cache and os.path.exists(local_folder_path):
            shutil.rmtree(local_folder_path)

        response = urlopen(url)

        # Need to infer
        if archive_type is None:
            if url.endswith('.tar.gz'):
                archive_type = '.tar.gz'
            elif url.endswith('.zip'):
                archive_type = '.zip'
            else:
                # info = response.info()
                try:
                    # header = next(x for x in info.headers if x.startswith('Content-Disposition'))
                    header = response.headers['content-disposition']
                    original_file_name = next(x for x in header.split(';') if x.startswith('filename')).split('=')[-1].lstrip('"\'').rstrip('"\'')
                    archive_type = '.tar.gz' if original_file_name.endswith('.tar.gz') else '.zip' if original_file_name.endswith('.zip') else \
                        bad_value(original_file_name, 'Filename "%s" does not end with a familiar zip extension like .zip or .tar.gz' % (original_file_name, ))
                except StopIteration:
                    raise Exception("Could not infer archive type from user argument, url-name, or file-header.  Please specify archive type as either '.zip' or '.tar.gz'.")
        print('Downloading archive from url: "%s"...' % (url, ))
        data = response.read()
        print('...Done.')

        local_zip_path = local_folder_path + archive_type
        if os.path.exists(local_zip_path):
            if os.path.isdir(local_zip_path): # This shouldnt happen but may by accident.
                rmtree(local_zip_path)
            else:
                os.remove(local_zip_path)
        make_file_dir(local_zip_path)
        with open(local_zip_path, 'wb') as f:
            f.write(data)

        force_extract = True

    if force_extract:
        if archive_type == '.tar.gz':
            with tarfile.open(local_zip_path) as f:
                f.extractall(local_folder_path)
        elif archive_type == '.zip':
            with ZipFile(local_zip_path) as f:
                f.extractall(local_folder_path)
        else:
            raise Exception()

    return local_folder_path


# def get_file_and_cache(url, data_transformation = None, enable_cache_write = True, enable_cache_read = True):
#
#     _, ext = os.path.splitext(url)
#
#     if enable_cache_read or enable_cache_write:
#         hasher = hashlib.md5()
#         hasher.update(url.encode('utf-8'))
#         code = hasher.hexdigest()
#
#
#
#         local_cache_path = os.path.join(get_artemis_data_path('caches'), code + ext)
#
#     if enable_cache_read and os.path.exists(local_cache_path):
#         return local_cache_path
#     elif enable_cache_write:
#         full_path = get_file(
#             relative_name = os.path.join('caches', code+ext),
#             url = url,
#             data_transformation=data_transformation
#             )
#         return full_path
#     else:
#         return get_temp_file(url, data_transformation=data_transformation)
#
#
# def get_temp_file(url, data_transformation = None):
#     tmp_file = get_unnamed_file_hash(url)
#     return get_file(tmp_file, url, data_transformation=data_transformation)


def unzip_gz(data):
    return gzip.GzipFile(fileobj = StringIO(data)).read()


def get_file_path(relative_name = None, url=None, make_folder = False):

    if relative_name is None:
        relative_name = get_unnamed_file_hash(url)

    full_path = os.path.join(FILE_ROOT, relative_name) if relative_name is not None else get_unnamed_file_hash(url)

    if make_folder:
        local_folder, file_name = os.path.split(full_path)
        # local_folder = os.path.join(FILE_ROOT, relative_folder)
        try:  # Best way to see if folder exists already - avoids race condition between processes
            os.makedirs(local_folder)
        except OSError:
            pass
    return full_path


def get_unnamed_file_hash(url):
    if url is not None:
        _, ext = os.path.splitext(url)
    else:
        import random
        import string
        elements = string.ascii_uppercase + string.digits
        url = ''.join(random.choice(elements) for _ in range(256))

    hasher = hashlib.md5()
    hasher.update(url.encode('utf-8'))
    filename = os.path.join('temp', hasher.hexdigest())
    return filename
