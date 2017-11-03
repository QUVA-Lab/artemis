from functools import partial
import time

import numpy as np
from artemis.plotting.demo_dbplot import demo_dbplot
from artemis.plotting.db_plotting import dbplot, clear_dbplot, hold_dbplots, freeze_all_dbplots, reset_dbplot, \
    dbplot_hang
from artemis.plotting.matplotlib_backend import LinePlot, HistogramPlot, MovingPointPlot, is_server_plotting_on
import pytest
from matplotlib import pyplot as plt
from matplotlib import gridspec
from six.moves import xrange

__author__ = 'peter'


def test_dbplot(n_steps = 3):

    reset_dbplot()

    arr = np.random.rand(10, 10)
    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            dbplot(barr, 'barr', plot_type=partial(LinePlot))


@pytest.mark.skipif('True', reason = 'Need to make matplotlib backend work with scales.')
def test_dbplot_logscale(n_steps = 3):
    reset_dbplot()

    arr = np.random.rand(10, 10)

    for i in xrange(n_steps):
        arr_sq=arr**2
        arr = arr_sq/np.mean(arr_sq)
        dbplot(arr, 'arr')
        for j in xrange(3):
            barr = np.random.randn(10, 2)
            kw = {"y_axis_type":"log"}
            dbplot(barr, 'barr', plot_type=partial(LinePlot,y_axis_type='log'))


def test_particular_plot(n_steps = 3):
    reset_dbplot()

    for i in xrange(n_steps):
        r = np.random.randn(1)
        dbplot(r, plot_type=partial(HistogramPlot,edges=np.linspace(-5, 5, 20)))


def test_history_plot_updating():
    """
    This test checks that we've fixed the bug mentioned in issue 1: https://github.com/QUVA-Lab/artemis/issues/1
    That was, when you are updating multiple plots with history in a loop, everytime any of the plots is updated, they
    all get updated with the most recent data.  You'll see this in plot 'c' - with the bug, it moves in steps, with 3
    of the same sample in a row.  If it works it should be spikey.
    """
    reset_dbplot()
    for i in xrange(10):
        dbplot(np.random.randn(20, 20), 'a')
        dbplot(np.random.randn(20, 20), 'b')
        dbplot(np.random.randn(), 'c', plot_type=partial(MovingPointPlot))
        # dbplot(np.random.randn(), 'c', plot_type=partial(MovingPointPlot, memory=2))


def test_moving_point_multiple_points():
    reset_dbplot()
    for i in xrange(5):
        dbplot(np.sin([i/10., i/15.]), 'unlim buffer', plot_type = partial(MovingPointPlot))
        dbplot(np.sin([i/10., i/15.]), 'lim buffer', plot_type = partial(MovingPointPlot,buffer_len=20))

def test_same_object():
    """
    There was a bug where when you plotted two of the same array, you got "already seen object".  This tests makes
    sure it's gotten rid of.  If it's gone, both matrices should plot.  Otherwise you'll get "Already seen object" showing
    up on one of the plots.
    """
    reset_dbplot()
    a = np.random.randn(20, 20)
    for _ in xrange(5):
        dbplot(a, 'a')
        dbplot(a, 'b')


def test_multiple_figures():
    reset_dbplot()
    for _ in xrange(2):
        dbplot(np.random.randn(20, 20), 'a', fig='1')
        dbplot(np.random.randn(20, 20), 'b', fig='1')
        dbplot(np.random.randn(20, 20), 'c', fig='2')
        dbplot(np.random.randn(20, 20), 'd', fig='2')


def test_list_of_images():
    reset_dbplot()
    for _ in xrange(2):
        dbplot([np.random.randn(12, 30), np.random.randn(10, 10), np.random.randn(15, 10)])


def test_two_plots_in_the_same_axis_version_1():
    reset_dbplot()
    # Option 1: Name the 'axis' argument to the second plot after the name of the first
    for i in xrange(5):
        data = np.random.randn(200)
        x = np.linspace(-5, 5, 100)
        with hold_dbplots():
            dbplot(data, 'histogram', plot_type='histogram')
            dbplot((x, 1./np.sqrt(2*np.pi*np.var(data)) * np.exp(-(x-np.mean(data))**2/(2*np.var(data)))), 'density', axis='histogram', plot_type='line')


def test_two_plots_in_the_same_axis_version_2():
    reset_dbplot()
    # Option 2: Give both plots the same 'axis' argument
    for i in xrange(5):
        data = np.random.randn(200)
        x = np.linspace(-5, 5, 100)
        with hold_dbplots():
            dbplot(data, 'histogram', plot_type='histogram', axis='hist')
            dbplot((x, 1./np.sqrt(2*np.pi*np.var(data)) * np.exp(-(x-np.mean(data))**2/(2*np.var(data)))), 'density', axis='hist', plot_type='line')


@pytest.mark.skipif(is_server_plotting_on(), reason = "This fails in server mode because we currently do not have an interpretation of freeze_all_dbplots")
def test_freeze_dbplot():
    reset_dbplot()
    def random_walk():
        data = 0
        for i in xrange(10):
            data += np.random.randn()
            dbplot(data, 'walk')#, plot_type=lambda: MovingPointPlot(axes_update_mode='expand'))

    random_walk()
    freeze_all_dbplots()
    random_walk()


def test_trajectory_plot():

    for i in xrange(5):
        dbplot((np.cos(i/10.), np.sin(i/11.)), 'path', plot_type='trajectory')


def test_demo_dbplot():
    demo_dbplot(n_frames=3)
    clear_dbplot()


def test_custom_axes_placement(hang=False):

    gs1 = gridspec.GridSpec(3, 1, left=0, right=0.5, hspace=0)
    dbplot(np.sin(np.linspace(0, 10, 100)), 'a', plot_type='line', axis=gs1[0, 0])
    dbplot(np.sin(np.linspace(0, 10, 100)+1), 'b', plot_type='line', axis=gs1[1, 0])
    dbplot(np.sin(np.linspace(0, 10, 100)+2), 'c', plot_type='line', axis=gs1[2, 0])

    gs2 = gridspec.GridSpec(2, 1, left=0.5, right=1, hspace=0.1)
    dbplot(np.random.randn(20, 20), 'im1', axis=gs2[0, 0])
    dbplot(np.random.randn(20, 20, 3), 'im2', axis=gs2[1, 0])

    if hang:
        dbplot_hang()


def test_cornertext():

    dbplot(np.random.randn(5, 5), 'a', cornertext='one')
    dbplot(np.random.randn(5, 5), 'a', cornertext='two')
    dbplot(np.random.randn(5, 5), 'a', cornertext='three')


def test_close_and_open():

    for _ in xrange(20):
        dbplot(np.random.randn(5), 'a')

    plt.close(plt.gcf())

    for _ in xrange(20):
        dbplot(np.random.randn(5), 'b')


if __name__ == '__main__':
    if is_server_plotting_on():
        test_cornertext()
        time.sleep(2.)
        test_trajectory_plot()
        time.sleep(2.)
        test_demo_dbplot()
        time.sleep(2.)
        test_two_plots_in_the_same_axis_version_1()
        time.sleep(2.)
        test_two_plots_in_the_same_axis_version_2()
        time.sleep(2.)
        test_moving_point_multiple_points()
        time.sleep(2.)
        test_list_of_images()
        time.sleep(2.)
        test_multiple_figures()
        time.sleep(2.)
        test_same_object()
        time.sleep(2.)
        test_history_plot_updating()
        time.sleep(2.)
        test_particular_plot()
        time.sleep(2.)
        test_dbplot()
        time.sleep(2.)
        test_custom_axes_placement()
        time.sleep(2.)
        test_close_and_open()
        time.sleep(2.)
    else:
        test_cornertext()
        test_trajectory_plot()
        test_demo_dbplot()
        test_freeze_dbplot()
        test_two_plots_in_the_same_axis_version_1()
        test_two_plots_in_the_same_axis_version_2()
        test_moving_point_multiple_points()
        test_list_of_images()
        test_multiple_figures()
        test_same_object()
        test_history_plot_updating()
        test_particular_plot()
        test_dbplot()
        test_custom_axes_placement()
        test_close_and_open()
