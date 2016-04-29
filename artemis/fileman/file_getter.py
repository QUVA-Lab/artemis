import urllib2
from StringIO import StringIO
import gzip

from fileman.local_dir import get_local_path
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
            data = data_transformation(data)
        with open(full_filename, 'w') as f:
            f.write(data)
    return full_filename


def unzip_gz(data):
    return gzip.GzipFile(fileobj = StringIO(data)).read()
