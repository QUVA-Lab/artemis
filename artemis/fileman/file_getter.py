import hashlib
import tempfile
import urllib2
from StringIO import StringIO
import gzip

from artemis.fileman.local_dir import get_local_path
import os

__author__ = 'peter'


def get_file(relative_name, url = None, data_transformation = None):

    relative_folder, file_name = os.path.split(relative_name)
    local_folder = get_local_path(relative_folder)

    try:  # Best way to see if folder exists already - avoids race condition between processes
        os.makedirs(local_folder)
    except OSError:
        pass

    full_filename = os.path.join(local_folder, file_name)

    if not os.path.exists(full_filename):
        assert url is not None, "No local copy of '%s' was found, and you didn't provide a URL to fetch it from" % (full_filename, )

        print 'Downloading file from url: "%s"...' % (url, )
        response = urllib2.urlopen(url)
        data = response.read()
        print '...Done.'

        if data_transformation is not None:
            print 'Processing downloaded data...'
            data = data_transformation(data)
        with open(full_filename, 'w') as f:
            f.write(data)
    return full_filename


def get_file_and_cache(url, data_transformation = None, enable_cache_write = True, enable_cache_read = True):

    if enable_cache_read or enable_cache_write:
        hasher = hashlib.md5()
        hasher.update(url)
        code = hasher.hexdigest()
        local_cache_path = os.path.join(get_local_path('caches'), code)

    if enable_cache_read and os.path.exists(local_cache_path):
        return local_cache_path
    elif enable_cache_write:
        full_path = get_file(
            relative_name = os.path.join('caches', code),
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
