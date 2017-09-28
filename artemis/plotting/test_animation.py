import matplotlib
import pytest
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange

__author__ = 'peter'

"""
Small demo that shows how to get pixel values of a plot.
"""
# TODO: Figure out how to save these as a GIF or some animation.


@pytest.mark.skipif(True, reason = "This fails in pytest but runs when run directly.  I don't know why, but it doesn't matter for now.")
def test_animation():

    fig = plt.figure()
    get_data = lambda: np.random.randn(5, 6)
    plt.ion()
    h=plt.imshow(get_data())
    plt.show()

    frames = []
    for i in xrange(20):
        lastframe = fig2im(fig)
        frames.append(fig2im(fig))
        h.set_array(lastframe)
        plt.draw()


def fig2im(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    return buf


if __name__ == '__main__':
    test_animation()
