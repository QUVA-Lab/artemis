import numpy as np
from plotting.db_plotting import dbplot
# from plotting.matplotlib_backend import HistogramPlot
from plotting.bokeh_backend import LinePlot
__author__ = 'peter'


def test_dbplot(n_steps = 3):

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            kw = {"y_axis_type":"log"}
            dbplot(barr, 'barr', plot_constructor=lambda: LinePlot(**kw))


# def test_particular_plot(n_steps = 3):
#
#     for i in xrange(n_steps):
#         r = np.random.randn(1)
#         dbplot(r, plot_constructor=lambda: HistogramPlot(edges=np.linspace(-5, 5, 20)))


if __name__ == '__main__':
    test_particular_plot()
    test_dbplot()
