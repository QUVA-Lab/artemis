import time
from artemis.plotting.live_plotting import LiveStream, LivePlot, LiveCanal
from artemis.plotting.matplotlib_backend import MovingImagePlot, MovingPointPlot, LinePlot, ImagePlot, HistogramPlot
from itertools import count
from six.moves import xrange

__author__ = 'peter'
import numpy as np


def test_streaming(duration = 10):

    c = count()

    stream = LiveStream(lambda: {
        'text': ['Veni', 'Vidi', 'Vici'][next(c) % 3],
        'images': {
            'bw_image': np.random.randn(20, 20),
            'col_image': np.random.randn(20, 20, 3),
            'vector_of_bw_images': np.random.randn(11, 20, 20),
            'vector_of_colour_images': np.random.randn(11, 20, 20, 3),
            'matrix_of_bw_images': np.random.randn(5, 6, 20, 20),
            'matrix_of_colour_images': np.random.randn(5, 6, 20, 20, 3),
            },
        'line': np.random.randn(20),
        'lines': np.random.randn(20, 3),
        'moving_point': np.random.randn(),
        'moving_points': np.random.randn(3),
        })
    for i in xrange(duration):
        if i==1:
            start_time = time.time()
        elif i>1:
            print('Average Frame Rate: %.2f FPS' % (i/(time.time()-start_time), ))
        stream.update()


def test_dynamic_rebuild():

    def grab_data():
        if i < 10:
            data = {'bw_image': np.random.randn(20, 20)}
        else:
            data = {
                'bw_image': np.random.randn(20, 20),
                'lines': np.random.randn(2)
                }
        return data

    duration = 20
    stream = LiveStream(grab_data)
    for i in xrange(duration):
        if i==1:
            start_time = time.time()
        elif i>1:
            print('Average Frame Rate: %.2f FPS' % (i/(time.time()-start_time), ))
        stream.update()


def test_canaling(duration = 10):

    n_dims = 4

    # Don't be frightened by the double-lambda here - the point is just to get a callback
    # that spits out the same data when called in sequence.
    cb_constructor_1d = lambda: lambda rng = np.random.RandomState(0): rng.randn(n_dims)
    cb_image_data = lambda: lambda rng = np.random.RandomState(1): rng.rand(20, 30)
    cb_sinusoid_data = lambda: lambda c=count(): np.sin(next(c)/40.)

    canal = LiveCanal({
        'histo-mass': LivePlot(plot = HistogramPlot([-2.5, 0, 0.5, 1, 1.5, 2, 2.5], mode = 'mass'), cb = lambda: np.random.randn(np.random.randint(10))),
        'histo-density': LivePlot(plot = HistogramPlot([-2.5, 0, 0.5, 1, 1.5, 2, 2.5], mode = 'density'), cb = lambda: np.random.randn(np.random.randint(10))),
        '1d-default': cb_constructor_1d(),
        '1d-image': LivePlot(plot = MovingImagePlot(buffer_len=20), cb = cb_constructor_1d()),
        '1d-seizmic': LivePlot(plot = MovingPointPlot(), cb = cb_constructor_1d()),
        '1d-line': LivePlot(plot = LinePlot(), cb = cb_constructor_1d()),
        'image-autoscale': LivePlot(ImagePlot(), cb_image_data()),
        'image-overexposed': LivePlot(ImagePlot(clims = (0, 0.2)), cb_image_data()),
        'image-jet': LivePlot(ImagePlot(cmap='jet'), cb_image_data()),
        'trace-default': cb_sinusoid_data(),
        'trace-prescaled': LivePlot(MovingPointPlot(y_bounds=(-1, 1)), cb_sinusoid_data()),
        })

    for i in xrange(duration):
        canal.update()


if __name__ == '__main__':

    # set_test_mode(True)
    test_dynamic_rebuild()
    test_streaming(10)
    test_canaling(10)
