import numpy as np
from plotting.db_plotting import dbplot
# from plotting.matplotlib_backend import HistogramPlot

from plotting.bokeh_backend import set_url, LinePlot
import os
os.environ["DISPLAY"] = "localhost:10.0"
import time

def tet_dbplot(n_steps = 3):

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        # dbplot(arr, 'arr')
        for j in xrange(1):
            barr = np.array([[x**10 for x in np.arange(0,5)], [x**10 for x in np.arange(0,5)]])
            barr = barr.T
            barr2 = np.array([[x**9 for x in np.arange(0,5)], [x**8 for x in np.arange(0,5)]])

            kw = {"y_axis_type":"log"}
            dbplot(barr, 'barr', plot_constructor=lambda: LinePlot(**kw))
            dbplot(barr2, 'barr2', plot_constructor=lambda: LinePlot(**kw))

            # time.sleep(4)


if __name__ == '__main__':
    # test_particular_plot()
    set_url("http://146.50.149.168:5006")
    tet_dbplot()

