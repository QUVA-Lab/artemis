from artemis.fileman.file_getter import get_file_in_archive, get_archive
from artemis.fileman.smart_io import smart_load_video
from artemis.general.should_be_builtins import bad_value
import os

__author__ = 'peter'


def load_ilsvrc_video(identifier, size = None, resize_mode='scale_crop', cut_edges=True, cut_edges_thresh=5):
    """
    Load a file from the ILSVRC Dataset.  The first time this is run, it will download an 8GB file, so be patient.

    Note: If you are using the same videos repeatedly, and applying resizing, you may want to call this function as:
        memoize_to_disk(load_ilsvrc_video)(identifier, size, ...)
    This will save you time on future runs.

    :param identifier: The file-name of the video, not including the extension.  Eg: 'ILSVRC2015_train_00249001'
    :param size:
    :param cut_edges:
    :param cut_edges_thresh:
    :return:
    """

    print ('Downloading ILSVER2015... this may take a while...')
    archive_folder_path = get_archive(
        relative_path='data/ILSVRC2015',
        url='http://vision.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID_snippets_final.tar.gz'
        )
    print ('Done.')
    subpath = \
        'ILSVRC2015/Data/VID/snippets/test' if 'test' in identifier else \
        'ILSVRC2015/Data/VID/snippets/val' if 'val' in identifier else \
        'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0001/' if os.path.exists(os.path.join(archive_folder_path, 'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0001/', identifier + '.mp4')) else \
        'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0002/' if os.path.exists(os.path.join(archive_folder_path, 'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0002/', identifier + '.mp4')) else \
        'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0003/' if os.path.exists(os.path.join(archive_folder_path, 'ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0003/', identifier + '.mp4')) else \
        bad_value(identifier, 'Could not find identifier: {}'.format(identifier, ))

    print('Loading %s' % (identifier, ))
    full_path = get_file_in_archive(
        relative_path='data/ILSVRC2015',
        subpath=os.path.join(subpath, identifier+'.mp4'),
        url='http://vision.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID_snippets_final.tar.gz'
        )
    video = smart_load_video(full_path, size=size, cut_edges=cut_edges, resize_mode=resize_mode, cut_edges_thresh=cut_edges_thresh)
    print('Done.')
    return video


if __name__ == '__main__':
    import itertools
    from artemis.plotting.db_plotting import dbplot, hold_dbplots

    identifiers = ['ILSVRC2015_train_00033009', 'ILSVRC2015_train_00033010', 'ILSVRC2015_train_00763000', 'ILSVRC2015_test_00004002']
    videos = [load_ilsvrc_video(identifier, size=(224, 224), cut_edges=True) for identifier in identifiers]

    for i in itertools.count(0):
        with hold_dbplots():
            for identifier, vid in zip(identifiers, videos):
                dbplot(vid[i%len(vid)], identifier, title='%s: %s' % (identifier, i%len(vid)))
