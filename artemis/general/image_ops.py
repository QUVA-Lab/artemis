from scipy.misc.pilutil import imresize

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
