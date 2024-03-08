import time

from artemis.plotting.cv2_plotting import just_show
from artemis.plotting.threaded_show import hold_just_show_in_thread
import numpy as np


def test_threaded_show():
    with hold_just_show_in_thread():
        for i in range(100):
            just_show(np.random.uniform(0, 255, size=(200, 300)).astype(np.uint8))
            time.sleep(0.5)


if __name__ == "__main__":
    test_threaded_show()
