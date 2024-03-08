from typing import Iterator

from artemis.general.custom_types import BGRImageArray
from artemis.plotting.easy_window import hold_just_show_capture, JustShowCapturer
from artemis.image_processing.image_utils import heatmap_to_color_image
from artemis.plotting.cv2_plotting import just_show
from artemis.plotting.scrollable_image_stream import show_scrollable_image_sequence
import numpy as np


def iter_diagonal_wave_images(size_xy = (640, 480)):
    sx, sy = size_xy
    xs, ys = np.meshgrid(np.linspace(0, 10, sx), np.linspace(0, 10, sy))
    for i in range(200):
        yield heatmap_to_color_image(np.sin(xs + ys + i / 20) ** 2)


def mock_show_func(image: BGRImageArray, name: str) -> None:
    print(f"Got command to show image with shape {image.shape} under name {name}")


def test_show_scrollable_image_sequence():

    show_scrollable_image_sequence(
        image_iterator=iter_diagonal_wave_images(),
        max_buffer_size=20,
        add_index_string=True,
        initially_pause=False
    )


def test_image_show_thing(show=False):

    with hold_just_show_capture() as render_func:

        just_show(heatmap_to_color_image(np.random.randn(100, 200)**2), 'random')
        just_show(next(iter_diagonal_wave_images()), 'waves')

    img = render_func()
    if show:
        just_show(img, hang_time=10)


def test_image_show_with_buffer():
    def iter_images() -> Iterator[BGRImageArray]:
        cap = JustShowCapturer()
        for i, wave_img in enumerate(iter_diagonal_wave_images()):
            with cap.hold_capture() as render_func:
                if i % 10 == 5:
                    just_show(heatmap_to_color_image(np.random.randn(100, 200) ** 2), 'random')
                just_show(wave_img, 'waves')
            yield render_func()

    show_scrollable_image_sequence(iter_images(), )


if __name__ == '__main__':
    # test_launch_scrollable_image_buffer()
    test_image_show_thing(show=True)
    # test_image_show_with_buffer()