from artemis.general.ezprofile import EZProfiler
from artemis.plotting.db_plotting import dbplot, hold_dbplots, set_dbplot_figure_size
import numpy as np
from artemis.plotting.matplotlib_backend import MovingPointPlot


__author__ = 'peter'


def demo_dbplot(n_frames = 1000):
    """
    Demonstrates the various types of plots.

    The appropriate plot can usually be inferred from the first input data.  In cases where there are multple ways to
    display the input data, you can use the plot_type argument.
    """
    from matplotlib import pyplot as plt
    plt.ion()

    set_dbplot_figure_size(15, 10)
    with EZProfiler('plot time') as prof:
        for i in xrange(n_frames):
            with hold_dbplots():  # Sets it so that all plots update at once (rather than redrawing on each call, which is slower)
                dbplot(np.random.randn(20, 20), 'Greyscale Image')
                dbplot(np.random.randn(20, 20, 3), 'Colour Image')
                dbplot(np.random.randn(15, 20, 20), "Many Images")
                dbplot(np.random.randn(3, 6, 20, 20, 3), "Colour Image Grid")
                dbplot([np.random.randn(15, 20, 3), np.random.randn(10, 10, 3), np.random.randn(10, 30, 3)], "Differently Sized images")
                dbplot(np.random.randn(20, 2), 'Two Lines')
                dbplot((np.linspace(-5, 5, 100)+np.sin(np.linspace(-5, 5, 100)*2), np.linspace(-5, 5, 100)), '(X,Y) Lines')
                dbplot([np.sin(i/20.), np.sin(i/15.)], 'Moving Point History')
                dbplot([np.sin(i/20.), np.sin(i/15.)], 'Bounded memory history', plot_type=lambda: MovingPointPlot(buffer_len=10))
                dbplot((np.sin(i/5.)*(i+100.), np.cos(i/5.)*(i+100.)), '(X,Y) Moving Point History')
                dbplot((np.sin(np.array([i, i*1.5, i*2])/5.)*(i+10.), np.cos(np.array([i, i*1.5, i*2])/5.)*(i+10.)), 'multi (X,Y) Moving Point History', plot_type='trajectory')
                dbplot(np.random.randn(20), 'Vector History')
                dbplot(np.random.randn(50), 'Histogram', plot_type = 'histogram')
                dbplot(np.random.randn(50), 'Cumulative Histogram', plot_type = 'cumhist')
                dbplot(('Veni', 'Vidi', 'Vici')[i%3], 'text-history')
                dbplot(('Veni', 'Vidi', 'Vici')[i%3], 'text-notice', plot_type='notice')

            if i % 10 == 0:
                print 'Mean Frame Rate: %.3gFPS' % ((i+1)/prof.get_current_time(), )


def demo_debug_dbplot():

    import pdb
    for i in xrange(1000):
        dbplot(np.random.randn(50, 2), 'a')
        print 'aaa'
        pdb.set_trace()
        dbplot(np.random.randn(10, 10), 'b')
        print 'bbb'
        pdb.set_trace()


if __name__ == '__main__':
    demo_dbplot()
    demo_debug_dbplot()