import json
from itertools import izip
from artemis.fileman.file_getter import get_file, unzip_gz
from artemis.fileman.smart_io import smart_load
import numpy as np
import os
from six.moves import xrange

__author__ = 'peter'


def get_imagenet_fall11_urls(n_images = None):

    if n_images is None:
        n_images = 14197121

    imagenet_urls = get_file(
        'data/imagnet_urls.txt',
        url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz',
        data_transformation=unzip_gz
        )

    print('Loading %s image URLs....' % (n_images, ))
    with open(imagenet_urls) as f:
        f.readline()
        lines = list(line for _, line in izip(xrange(n_images), f))
    indices = [s.index('\t') for s in lines]
    pairs = [(line[:s], line[s+1:-1]) for line, s in zip(lines, indices)]
    print('Done.')
    return pairs


def get_imagenet_images(indices):
    """
    Get imagenet images at the given indices
    :param indices:
    :return:
    """
    highest_index = np.max(indices)
    code_url_pairs = get_imagenet_fall11_urls(highest_index+1)
    files = [get_file('data/imagenet/%s%s' % (code_url_pairs[index][0], os.path.splitext(code_url_pairs[index][1])[1]), code_url_pairs[index][1]) for index in indices]
    return [smart_load(f) for f in files]


def get_imagenet_label_names():

    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
    with open(get_file('data/imagenet/labels.json', url=url)) as f:
        label_items = f.read()

    labels = [line[line.index(':')+1:].lstrip(' \'').rstrip('}, \'') for line in label_items.split('\n')]
    return labels

if __name__ == '__main__':
    # Downloads 4 random images out of the first 1000.  You may get 404 errors, etc.  So just run again and again til this works.
    import random
    from artemis.plotting.db_plotting import dbplot
    ixs = [random.randint(0, 999) for _ in xrange(4)]
    print(ixs)
    ims = get_imagenet_images(ixs)
    for i, (ix, im) in enumerate(zip(ixs, ims)):
        dbplot(im, 'Image %s' % i, hang = i==len(ims)-1)
