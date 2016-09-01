from artemis.general.image_ops import resize_while_preserving_aspect_ratio, equalize_image_dims

__author__ = 'peter'
import numpy as np


def test_resize_while_preserving_aspect_ratio():

    img = np.random.randn(10, 20, 3)
    bwimg = np.random.randn(10, 20)

    img2 = resize_while_preserving_aspect_ratio(img, x_dim=30)
    assert img2.shape == (15, 30, 3)

    bwimg2 = resize_while_preserving_aspect_ratio(bwimg, x_dim=30)
    assert bwimg2.shape == (15, 30)

    img3 = resize_while_preserving_aspect_ratio(img, y_dim=5)
    assert img3.shape == (5, 10, 3)


def test_equalize_image_dims():

    imgs = [np.random.randn(12, 18, 3), np.random.randn(12, 12), np.random.randn(24, 12)]

    new_imgs = equalize_image_dims(imgs, x_dim = 'max', y_dim='max')
    assert [im.shape for im in new_imgs] == [(12, 18, 3), (18, 18), (24, 12)]

    new_imgs = equalize_image_dims(imgs, x_dim = 12, y_dim=12)
    assert [im.shape for im in new_imgs] == [(8, 12, 3), (12, 12), (12, 6)]

    new_imgs = equalize_image_dims(imgs, x_dim = 16)
    assert [im.shape for im in new_imgs] == [(11, 16, 3), (16, 16), (32, 16)]

    new_imgs = equalize_image_dims(imgs, y_dim = 16)
    assert [im.shape for im in new_imgs] == [(16, 24, 3), (16, 16), (16, 8)]

    new_imgs = equalize_image_dims(imgs, x_dim = 'max', y_dim = 16)
    assert [im.shape for im in new_imgs] == [(12, 18, 3), (16, 16), (16, 8)]


if __name__ == "__main__":
    test_resize_while_preserving_aspect_ratio()
    test_equalize_image_dims()
