import hashlib
import tempfile
from six.moves import StringIO
import gzip
import tarfile
from zipfile import ZipFile
from artemis.fileman.local_dir import get_artemis_data_path, make_dir, make_file_dir
from artemis.general.should_be_builtins import bad_value
import shutil
import os
from six.moves.urllib.request import urlopen

__author__ = 'peter'


def get_file(relative_name, url = None, data_transformation = None):

    relative_folder, file_name = os.path.split(relative_name)
    local_folder = get_artemis_data_path(relative_folder)

    try:  # Best way to see if folder exists already - avoids race condition between processes
        os.makedirs(local_folder)
    except OSError:
        pass

    full_filename = os.path.join(local_folder, file_name)

    if not os.path.exists(full_filename):
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


def get_file_in_archive(relative_path, subpath, url, force_extract = False):
    """
    Download a zip file, unpack it, and get the local address of a file within this zip (so that you can open it, etc).

    :param relative_path: Local name for the extracted folder.  (Zip file will be named this with the appropriate zip extension)
    :param url: Url of the zip file to download
    :param subpath: Path of the file relative to the zip folder.
    :param force_extract: Force the zip file to re-extract (rather than just reusing the extracted folder)
    :return: The full path to the file on your system.
    """
    local_folder_path = get_archive(relative_path=relative_path, url=url, force_extract=force_extract)
    local_file_path = os.path.join(local_folder_path, subpath)
    assert os.path.exists(local_file_path), 'Could not find the file "%s" within the extracted folder: "%s"' % (subpath, local_folder_path)
    return local_file_path


def get_archive(relative_path, url, force_extract=False, archive_type = None, force_download = False):
    """
    Download a compressed archive and extract it into a folder.

    :param relative_path: Local name for the extracted folder.  (Zip file will be named this with the appropriate zip extension)
    :param url: Url of the archive to download
    :param force_extract: Force the zip file to re-extract (rather than just reusing the extracted folder)
    :return: The full path to the extracted folder on your system.
    """

    local_folder_path = get_artemis_data_path(relative_path)

    assert archive_type in ('.tar.gz', '.zip', None)

    if force_download:
        shutil.rmtree(local_folder_path)

    if not os.path.exists(local_folder_path) or force_download:  # If the folder does not exist, download zip and extract.
        # (We also check force download here to avoid a race condition)
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


def get_file_and_cache(url, data_transformation = None, enable_cache_write = True, enable_cache_read = True):

    _, ext = os.path.splitext(url)

    if enable_cache_read or enable_cache_write:
        hasher = hashlib.md5()
        hasher.update(url.encode('utf-8'))
        code = hasher.hexdigest()
        local_cache_path = os.path.join(get_artemis_data_path('caches'), code + ext)

    if enable_cache_read and os.path.exists(local_cache_path):
        return local_cache_path
    elif enable_cache_write:
        full_path = get_file(
            relative_name = os.path.join('caches', code+ext),
            url = url,
            data_transformation=data_transformation
            )
        return full_path
    else:
        return get_temp_file(url, data_transformation=data_transformation)


def get_temp_file(url, data_transformation = None):
    _, ext = os.path.splitext(url)
    tmp_file = tempfile.mktemp() + ext
    return get_file(tmp_file, url, data_transformation=data_transformation)


def unzip_gz(data):
    return gzip.GzipFile(fileobj = StringIO(data)).read()
