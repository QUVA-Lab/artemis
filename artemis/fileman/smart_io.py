import os
import pickle
import re
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from artemis.fileman.file_getter import get_temp_file, get_file_and_cache
from artemis.fileman.images2gif import readGif
from artemis.fileman.local_dir import get_artemis_data_path, make_file_dir
from artemis.general.image_ops import get_dark_edge_slice, resize_image


def smart_save(obj, relative_path, remove_file_after = False):
    """
    Save an object locally.  How you save it depends on its extension.
    Extensions currently supported:
        pkl: Pickle file.
        That is all.
    :param obj: Object to save
    :param relative_path: Path to save it, relative to "Data" directory.  The following placeholders can be used:
        %T - ISO time
        %R - Current Experiment Record Identifier (includes experiment time and experiment name)
    :param remove_file_after: If you're just running a test, it's good to verify that you can save, but you don't
        actually want to leave a file behind.  If that's the case, set this argument to True.
    """
    if '%T' in relative_path:
        iso_time = datetime.now().isoformat().replace(':', '.').replace('-', '.')
        relative_path = relative_path.replace('%T', iso_time)
    if '%R' in relative_path:
        from artemis.experiments.experiment_record import get_current_experiment_id
        relative_path = relative_path.replace('%R', get_current_experiment_id())
    _, ext = os.path.splitext(relative_path)

    with smart_file(relative_path, make_dir=True) as local_path:
        print('Saved object <%s at %s> to file: "%s"' % (obj.__class__.__name__, hex(id(object)), local_path))
        if ext=='.pkl':
            with open(local_path, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ext in _IMAGE_EXTENSIONS:
            _save_image(obj, local_path)
        elif ext=='.pdf':
            obj.savefig(local_path)
        else:
            raise Exception("No method exists yet to save '%s' files.  Add it!" % (ext, ))

    if remove_file_after:
        os.remove(local_path)

    return local_path


def smart_load(location, use_cache = False):
    """
    Load a file, with the method based on the extension.  See smart_save doc for the list of extensions.
    :param location: Identifies file location.
        If it's formatted as a url, it's downloaded.
        If it begins with a "/", it's assumed to be a local path.
        Otherwise, it is assumed to be referenced relative to the data directory.
    :param use_cache: If True, and the location is a url, make a local cache of the file for future use (note: if the
        file at this url changes, the cached file will not).
    :return: An object, whose type depends on the extension.  Generally a numpy array for data or an object for pickles.
    """
    assert isinstance(location, str), 'Location must be a string!  We got: %s' % (location, )
    with smart_file(location, use_cache=use_cache) as local_path:
        ext = os.path.splitext(local_path)[1].lower()
        if ext=='.pkl':
            with open(local_path) as f:
                obj = pickle.load(f)
        elif ext=='.gif':
            frames = readGif(local_path)
            if frames[0].shape[2]==3 and all(f.shape[2] for f in frames[1:]):  # Wierd case:
                obj = np.array([frames[0]]+[f[:, :, :3] for f in frames[1:]])
            else:
                obj = np.array(readGif(local_path))
        elif ext in _IMAGE_EXTENSIONS:
            from PIL import Image
            obj = _load_image(local_path)
        elif ext in ('.mpg', '.mp4', '.mpeg'):
            obj = _load_video(local_path)
        else:
            raise Exception("No method exists yet to load '%s' files.  Add it!" % (ext, ))
    return obj


def smart_load_image(location, max_resolution = None, force_rgb=False, use_cache = False):
    """
    Load an image into a numpy array.

    :param location: Identifies file location.
        If it's formatted as a url, it's downloaded.
        If it begins with a "/", it's assumed to be a local path.
        Otherwise, it is assumed to be referenced relative to the data directory.
    :param max_resolution: Maximum resolution (size_y, size_x) of the image
    :param force_rgb: Force an RGB representation (transform greyscale and RGBA images into RGB)
    :param use_cache: If True, and the location is a url, make a local cache of the file for future use (note: if the
        file at this url changes, the cached file will not).
    :return: An object, whose type depends on the extension.  Generally a numpy array for data or an object for pickles.
    """
    with smart_file(location, use_cache=use_cache) as local_path:
        return _load_image(local_path, max_resolution = max_resolution, force_rgb=force_rgb)

_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')


def _load_image(local_path, max_resolution = None, force_rgb = False):
    """
    :param local_path: Local path to the file
    :param max_resolution: Maximum resolution (size_y, size_x) of the image
    :param force_rgb: Force an RGB representation (transform greyscale and RGBA images into RGB)
    :return: The image array, which can be:
        (size_y, size_x, 3) For a colour RGB image
        (size_y, size_x) For a greyscale image
        (size_y, size_x, 4) For an image with transperancy (Note: Disabled for now)
    """
    # TODO: Consider replacing PIL with scipy
    ext = os.path.splitext(local_path)[1].lower()
    assert ext in _IMAGE_EXTENSIONS, "Can't deal with extension: {}".format(ext)
    from PIL import Image
    pic = Image.open(local_path)

    if max_resolution is not None:
        max_width, max_height = max_resolution
        pic.thumbnail((max_width, max_height), Image.ANTIALIAS)

    pic_arr = np.asarray(pic)

    if force_rgb:
        if pic_arr.ndim==2:
            pic_arr = np.repeat(pic_arr[:, :, None], 3, axis=2)
        elif pic_arr.shape[2]==4:
            pic_arr = pic_arr[:, :, :3]
        else:
            assert pic_arr.shape[2]==3
    return pic_arr


def _save_image(image_array, local_path):
    ext = os.path.splitext(local_path)[1].lower()
    assert ext in _IMAGE_EXTENSIONS
    from PIL import Image
    pic = Image.fromarray(image_array)
    pic.save(local_path)


@contextmanager
def smart_file(location, use_cache = False, make_dir = False):
    """
    :param location: Specifies where the file is.
        If it's formatted as a url, it's downloaded.
        If it begins with a "/", it's assumed to be a local path.
        Otherwise, it is assumed to be referenced relative to the data directory.
    :param use_cache: If True, and the location is a url, make a local cache of the file for future use (note: if the
        file at this url changes, the cached file will not).
    :param make_dir: Make the directory for this file, if it does not exist.
    :yield: The local path to the file.
    """
    its_a_url = is_url(location)
    if its_a_url:
        assert not make_dir, "We cannot 'make the directory' for a URL"
        if use_cache:
            local_path = get_file_and_cache(location)
        else:
            local_path = get_temp_file(location)
    else:
        local_path = get_artemis_data_path(location)
        if make_dir:
            make_file_dir(local_path)

    yield local_path

    if its_a_url and not use_cache:
        os.remove(local_path)


def is_url(path):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return True if re.match(regex, path) else False


def smart_load_video(location, use_cache = False, resize_mode='resize_and_crop', cut_edges=False, size = None, cut_edges_thresh=0):
    """
    :param location:
    :param size: A 2-tuple of width-height, indicating the desired size of the ouput
    :param resize_mode: The mode with which to get the video to the desired size.  Can be:
        'squeeze', 'preserve_aspect', 'crop', 'scale_crop'.  See resize_image in image_ops.py for more info.
    :param cut_edges: True if you want to cut the dark edges from the video
    :param cut_edges_thresh: If cut_edges, this is the threshold at which you'd like to cut them.
    :return: A (n_frames, height, width, 3) numpy array
    """

    with smart_file(location, use_cache=use_cache) as local_path:
        return _load_video(local_path, resize_mode=resize_mode, cut_edges=cut_edges, size=size, cut_edges_thresh=cut_edges_thresh)


def _load_video(full_path, size = None, resize_mode = 'resize_and_crop', cut_edges=False, cut_edges_thresh=0):
    """
    Lead a video into a numpy array

    :param full_path: Full path to the video
    :param size: A 2-tuple of width-height, indicating the desired size of the ouput
    :param resize_mode: The mode with which to get the video to the desired size.  Can be:
        'squeeze', 'preserve_aspect', 'crop', 'scale_crop'.  See resize_image in image_ops.py for more info.
    :param cut_edges: True if you want to cut the dark edges from the video
    :param cut_edges_thresh: If cut_edges, this is the threshold at which you'd like to cut them.
    :return: A (n_frames, height, width, 3) numpy array
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError:
        raise ImportError("You need to install moviepy to read videos.  In the virtualenv, go `pip install moviepy`")
    assert os.path.exists(full_path)
    video = VideoFileClip(full_path)
    images = []
    edge_crops = None
    for frame in video.iter_frames():
        if cut_edges:
            if edge_crops is None:
                edge_crops = get_dark_edge_slice(frame, cut_edges_thresh=cut_edges_thresh)
            else:
                frame = frame[edge_crops[0], edge_crops[1]]
        if size is not None:
            width, height = size
            frame = resize_image(frame, width=width, height=height, mode=resize_mode)
        images.append(frame)
    return images
