import numpy as np
from artemis.plotting.db_plotting import dbplot
from artemis.plotting.bokeh_backend import set_url, LinePlot, MovingPointPlot, HistoryPlot
# import os
# os.environ["DISPLAY"] = "localhost:10.0"
import time

def tet_dbplot(n_steps = 3):

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        barr = np.array([[x**10 for x in np.arange(0,5)], [x**10 for x in np.arange(0,5)]])
        barr = barr.T
        barr2 = np.array([[x**9 for x in np.arange(0,5)], [x**8 for x in np.arange(0,5)]])
        barr2 = barr2.T
        kw = {"y_axis_type":"log"}
        dbplot(barr, 'barr', plot_type=lambda: LinePlot(**kw))
        dbplot(barr2, 'barr2', plot_type=lambda: LinePlot(**kw))


def moving_point_plot(n_steps = 20):
    for i in xrange(n_steps):
        data = np.array([i,i**2])
        if i == 4:
            data = np.array([float('nan'), i**2])
        dbplot(data, "history", plot_type=lambda: MovingPointPlot())
        time.sleep(0.5)

if __name__ == '__main__':
    # test_particular_plot()
    set_url("http://146.50.149.168:5006")
    moving_point_plot(n_steps = 20)
    time.sleep(2)
    tet_dbplot()

