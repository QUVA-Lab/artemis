import itertools
from contextlib import contextmanager
from functools import partial
from queue import Queue
from typing import Iterator

from tensorflow.python.distribute.multi_process_lib import multiprocessing

from artemis.general.custom_types import BGRImageArray
from artemis.plotting.cv2_plotting import hold_alternate_show_func
from artemis.plotting.scrollable_image_stream import show_scrollable_image_sequence


def iter_images_from_queue(queue: Queue) -> Iterator[BGRImageArray]:
    for i in itertools.count(0):
        print(f"Getting froma {i} from queue")
        yield queue.get(block=True)


def launch_plotting_thread(queue: Queue, initially_pause=False):
    show_scrollable_image_sequence(iter_images_from_queue(queue), initially_pause=initially_pause)


@contextmanager
def hold_just_show_in_thread(initially_pause = False):
    queue = multiprocessing.Queue(maxsize=1)

    def new_just_show(image: BGRImageArray, name: str):
        queue.put(image)

    #
    # with JustShowCapturer(
    #     callback=lambda im: queue.put(im)
    # ).hold_capture() as image_getter:
#
    with hold_alternate_show_func(new_just_show):
        thread = multiprocessing.Process(target=partial(launch_plotting_thread, queue=queue, initially_pause=initially_pause))
        thread.start()
        yield
        thread.join()
        thread.close()
    # cap = JustShowCapturer.from_row_wrap(wrap_rows).hold_capture()

    #
    # def show_func()
    #
    # with hold_alternate_show_func(lambda: )
    #     JustShowCapturer.from_row_wrap(wrap_rows).hold_capture()
    #
    # Thread.start()
    #
    #
    # yield from JustShowCapturer.from_row_wrap(wrap_rows).hold_capture()

