import numpy as np
from artemis.plotting.db_plotting import dbplot
from artemis.plotting.plotting_backend import LinePlot, HistogramPlot
import pytest

__author__ = 'peter'


def test_dbplot(n_steps = 3):

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            dbplot(barr, 'barr', plot_constructor=lambda: LinePlot())


@pytest.mark.skipif('True', reason = 'Need to make matplotlib backend work with scales.')
def test_dbplot_logscale(n_steps = 3):

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            kw = {"y_axis_type":"log"}
            dbplot(barr, 'barr', plot_constructor=lambda: LinePlot(y_axis_type='log'))


def test_particular_plot(n_steps = 3):

    for i in xrange(n_steps):
        r = np.random.randn(1)
        dbplot(r, plot_constructor=lambda: HistogramPlot(edges=np.linspace(-5, 5, 20)))


if __name__ == '__main__':
    test_particular_plot()
    test_dbplot()
