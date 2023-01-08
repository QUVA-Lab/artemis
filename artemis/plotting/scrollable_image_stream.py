import time
from typing import Iterable, Optional

import cv2

from artemis.general.custom_types import BGRImageArray
from artemis.plotting.cv_keys import Keys
from artemis.plotting.easy_window import DEFAULT_WINDOW_NAME, cv_window_input
from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import BGRColors, ShowFunction, BoundingBox
from artemis.plotting.cv2_plotting import just_show
from artemis.general.sequence_buffer import SequenceBuffer


def show_scrollable_image_sequence(
        image_iterator: Iterable[BGRImageArray],
        show_func: ShowFunction = just_show,
        max_buffer_size: Optional[int] = None,
        pause_debounce_time: float = 1.,
        max_memory_size: Optional[int] = 1000000000,
        initially_pause: bool = False,
        max_fps: Optional[float] = None,
        expand_to_full_size = True,  # Expand window to full size of image.
        min_key_wait_time: float = 0.000,
        window_name: str = DEFAULT_WINDOW_NAME,
        upsample_factor: int = 1,
        add_index_string: bool = True,
        enable_zoom: bool = True,
        zoom_window_size = 100,
        zoom_scale_factor = 5,
        index_text_color = BGRColors.WHITE,
        copy_if_modifying: bool = False,  # Since we write text to the image.  If you are using it elsewhere, set this to true.
):
    image_iterator = iter(image_iterator)  # Just in case it's passed in as a list
    image_buffer: SequenceBuffer[BGRImageArray] = SequenceBuffer(max_elements=max_buffer_size, max_memory=max_memory_size)
    lookup_index = 0
    is_paused = initially_pause

    # if add_index_string:
    #     original_image_iterator = image_iterator
    #     def iter_labelled_images():
    #         for i, img in enumerate(original_image_iterator):
    #             if copy_if_modifying:
    #                 img = img.copy()
    #             put_text_in_corner(img=img, text=f'#{i}', color=index_text_color, background_color=BGRColors.BLACK)
    #             yield img
    #     image_iterator = iter_labelled_images()

    last_frame_time = -float('inf')
    period = 1/max_fps if max_fps is not None else 0.

    # for _ in iter_max_rate(max_fps):

    # cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name)

    last_pause_resume_time = -float('inf')
    last_shape = None

    zoom_position = latest_mouse_position = (0, 0)

    if enable_zoom:
        def click_callback(event, x, y, flags, param):
            nonlocal latest_mouse_position
            if toggle_zoom_overlay:
                latest_mouse_position = (x, y)

        cv2.setMouseCallback(window_name, click_callback)

    def show_image(img):
        return just_show(img, hang_time=float('inf') if is_paused else max(min_key_wait_time, last_frame_time + period - current_time), name=window_name,
                  upsample_factor=upsample_factor, enable_alternate=False)

    toggle_zoom_overlay = False
    while True:

        actual_index, image = image_buffer.lookup(lookup_index, jump_to_edge=True, new_data_source=image_iterator)

        if actual_index == lookup_index-1 and not is_paused:
            print(f'Reached final image at frame {actual_index}.  Pausing.')
            is_paused = True

        if add_index_string:
            # image = image.copy()
            title_text = f'{actual_index} ({"paused" if is_paused else "playing"}) {"(end)" if actual_index<lookup_index else "(oldest)" if actual_index==0 or actual_index>lookup_index else ""}'
            # put_text_in_corner(img=image, text=f'{"#" if is_paused else ">"}{actual_index} {"(end)" if actual_index<lookup_index else "(oldest)" if actual_index>lookup_index else ""}', color=index_text_color)
            cv2.setWindowTitle(window_name, title_text)

        print(f'Showing image at actual index {actual_index}')
        current_time = time.monotonic()
        if expand_to_full_size:
            # if expand_to_full_size and (image.shape[:2] != last_shape):
            cv2.resizeWindow(window_name, width=image.shape[1], height=image.shape[0])
            # last_shape = image.shape[:2]
        if toggle_zoom_overlay:
            box = BoundingBox.from_xywh(*zoom_position, zoom_window_size, zoom_window_size)
            image_with_inset = ImageBuilder.from_image(image).draw_zoom_inset_from_box(box, scale_factor=zoom_scale_factor).image
            key = show_image(image_with_inset)
        else:
            key = show_image(image)

        last_frame_time = current_time
        if key == Keys.P:
            if current_time - last_pause_resume_time > pause_debounce_time:
                # Debouncing is meant to solve the problem of delayed keystrokes causing you to constantly pause and resume.
                is_paused = not is_paused
                last_pause_resume_time = current_time
                print('Pausing...' if is_paused else 'Resuming...')
        elif key==Keys.COMMA:
            lookup_index = actual_index - 1
        elif key == Keys.PERIOD:
            lookup_index = actual_index + 1
        elif key == Keys.SEMICOLON:
            lookup_index = actual_index - 10
        elif key == Keys.APOSTROPHE:
            lookup_index = actual_index + 10
        elif key == Keys.LBRACE:
            lookup_index, _ = image_buffer.get_index_bounds()
        elif key == Keys.RBRACE:
            _, lookup_index = image_buffer.get_index_bounds()
        elif key == Keys.G:
            frame = cv_window_input(prompt='Enter frame to go to...')
            try:
                lookup_index = int(frame.strip())
                print(f"Going to frame {lookup_index}... If is available that is...")
            except ValueError:
                print(f'Could not parse "{frame}".  You need to enter an integer.')
        elif key == Keys.Z and enable_zoom:
            zoom_position = latest_mouse_position
            toggle_zoom_overlay = True
        elif key == Keys.X:
            toggle_zoom_overlay = False
        elif key == Keys.SPACE:
            pass
        elif key == Keys.ESC:
            return
        else:
            if key is not None:
                print(f'Unknown key {key}.  Advancing frame')
            lookup_index += 1
        lookup_index = max(0, lookup_index)





