from scipy.misc.pilutil import imresize
import numpy as np
__author__ = 'peter'


def resize_while_preserving_aspect_ratio(im, x_dim=None, y_dim=None):
    """
    Resize an image, while preserving the aspect ratio.  For this you need to specify either x_dim or y_dim.

    :param im: The image: a 2D or 3D array.
    :param x_dim: An integer indicating the desired size, or None, to leave it loose.
    :param y_dim: An integer indicating the desired size, or None, to leave it loose.
    :return: A new image whose x_dim or y_dim matches the constraint
    """
    assert not (x_dim is None and y_dim is None), 'You can not leave both constraints at None!'

    x_dim = float('inf') if x_dim is None else x_dim
    y_dim = float('inf') if y_dim is None else y_dim

    box_aspect_ratio = x_dim/float(y_dim)
    image_aspect_ratio = im.shape[1] / float(im.shape[0])
    if image_aspect_ratio > box_aspect_ratio:  # Active constraint is width
        return imresize(im, size=(int(x_dim/image_aspect_ratio+.5), x_dim))
    else:  # Active constraint is height
        return imresize(im, size=(y_dim, int(y_dim*image_aspect_ratio+.5)))


def equalize_image_dims(list_of_images, x_dim = None, y_dim = None):
    """
    Resize images so that they match roughly in size although their aspect ratio will be preserved.
    :param list_of_images: A list of numpy arrays representing images (2D or 3D arrays)
    :param size: A 2-tuple specifying the desired (y_size, x_size).
        Each of (y_size, x_size) can be:
        - An integar, meaning that this axis of the image will remain equal or smaller than this number of pixels.
        - None, meaning that there is no constraint along this axis (e.g. (224, None) just states that the image will be
          scaled to 224 pixels in the vertical direction - the horizontal will be whatever size is needed to maintain
          the aspect ratio.
        - 'max': Meaning that we take the largest image size along this axis.
        - 'min': Meaning what we take the largest image size along this axis.

        The image will then be scaled so that the image size remains inside this box (although, unless the aspect ratio
        matches exactly, one dimension will be smaller).

    :return: Another list of images.
    """
    assert not (x_dim is None and y_dim is None), 'You can not leave both constraints at None!'
    if len(list_of_images)==0:
        return []
    x_dim = max(im.shape[1] for im in list_of_images) if x_dim=='max' else \
        min(im.shape[1] for im in list_of_images) if x_dim=='min' else \
        x_dim
    y_dim = max(im.shape[0] for im in list_of_images) if y_dim=='max' else \
        min(im.shape[0] for im in list_of_images) if y_dim=='min' else \
        y_dim
    new_list_of_images = [resize_while_preserving_aspect_ratio(im, x_dim=x_dim, y_dim=y_dim) for im in list_of_images]
    return new_list_of_images


def resize_and_crop(im, width, height):
    im_aspect = float(im.shape[0])/im.shape[1]
    new_aspect = float(height)/width
    if im_aspect > new_aspect:  # Need to chop the top and bottom
        new_height = int(width*im_aspect)
        resized_im = imresize(im, (new_height, width))
        start = (new_height-height)/2
        output_im = resized_im[start:start+height, :]
    else:  # Need to chop the left and right.
        new_width = int(height/im_aspect)
        resized_im = imresize(im, (height, new_width))
        start = (new_width-width)/2
        output_im = resized_im[:, start:start+width]
    assert output_im.shape[:2] == (height, width)
    return output_im


def resize_image(im, width=None, height=None, mode='squeeze'):
    assert isinstance(im, np.ndarray) and im.ndim in (2, 3)
    if mode == 'squeeze':
        im = imresize(im, size=(height, width))
    elif mode == 'preserve_aspect':
        im = resize_while_preserving_aspect_ratio(im, x_dim=width, y_dim=height)
    elif mode == 'crop':
        current_height, current_width = im.shape[:2]
        assert height>=height and width>=width, "Crop size must be smaller than image size"
        row_start = (current_height-height)/2
        col_start = (current_width-width)/2
        im = im[..., row_start:row_start+224, col_start:col_start+224, :]
    elif mode in ('resize_and_crop', 'scale_crop'):
        assert height is not None and width is not None, "You need to specify both height and width. for 'scale_crop' mode"
        return resize_and_crop(im, width=width, height=height)
    else:
        raise Exception("Unknown resize mode: '{}'".format(mode))
    return im


def get_dark_edge_slice(im, cut_edges_thresh=0):
    vnonzero = np.flatnonzero(im.mean(axis=2).mean(axis=1)>cut_edges_thresh)
    hnonzero = np.flatnonzero(im.mean(axis=2).mean(axis=0)>cut_edges_thresh)
    edge_crops = slice(vnonzero[0], vnonzero[-1]+1), slice(hnonzero[0], hnonzero[-1]+1)
    return edge_crops


def cut_dark_edges(im, slices = None, cut_edges_thresh=0):
    if slices is None:
        slices = get_dark_edge_slice(im, cut_edges_thresh=cut_edges_thresh)
    y_slice, x_slice = slices
    return im[y_slice, x_slice]
