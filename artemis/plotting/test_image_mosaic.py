import nptyping

from artemis.image_processing.image_utils import create_random_image
from artemis.plotting.cv2_plotting import just_show
from artemis.plotting.easy_window import ImageRow
from artemis.plotting.image_mosaic import generate_image_mosaic_and_index_grid


def test_image_mosaic(show: bool = False):

    width, height = (60, 40)
    images = {
        k: create_random_image(size_xy=(width, height)) for k in range(1, 90)
    }
    mosaic, ixs = generate_image_mosaic_and_index_grid(mosaic=images)
    assert mosaic.ndim == 3 and mosaic.shape[2] == 3 and mosaic.dtype == nptyping.UInt8
    assert ixs.ndim == 2 and ixs.shape == mosaic.shape[:2] and ixs.dtype == int

    disp = ImageRow(mosaic, ixs).render()
    if show:
        just_show(disp, hang_time=10)


if __name__ == "__main__":
    test_image_mosaic(show=True)
