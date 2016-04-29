import numpy as np
from artemis.plotting.easy_plotting import ezplot
__author__ = 'peter'


class DataContainer(object):

    def __init__(self, im, line, struct, text, number):
        self._im = im
        self._line = line
        self._struct = struct
        self._text = text
        self._number = number


def test_easy_plot():

    thing = DataContainer(
        im =np.random.randn(30, 40),
        line = np.sin(np.arange(100)/10.),
        struct = {'video': np.random.randn(17, 20, 30)},
        text = 'adsagfdsf',
        number = 5
        )
    ezplot(thing, hang = False)


def test_plot_wmat():

    wmat = np.random.randn(7, 28, 28)
    ezplot(wmat, hang = False)


if __name__ == '__main__':
    test_plot_wmat()
    test_easy_plot()
