from artemis.general.ezprofile import EZProfiler
from artemis.plotting.matplotlib_backend import LinePlot
from artemis.plotting.db_plotting import dbplot, hold_dbplots
import numpy as np


__author__ = 'peter'


def demo_dbplot():

    from matplotlib import pyplot as plt
    plt.ion()
    with EZProfiler('plot time') as prof:
        for i in xrange(1000):
            with hold_dbplots():
                # dbplot(np.random.randn(20, 20), 'Greyscale Image')
                # dbplot(np.random.randn(20, 20, 3), 'Colour Image')
                # dbplot(np.random.randn(20, 2), 'Two Lines')
                # dbplot(np.cos([i/10., i/10.+np.pi/2]), 'History')
                # dbplot(np.random.randn(20), 'History Image')
                # dbplot(np.random.randn(15, 20, 20), "Many Images")

                dbplot((np.linspace(-5, 5, 100)[:, None], np.sin(np.linspace(-5, 5, 100)[:, None]+[i/10., (i+1)/10., (i+2)/10.])), 'lines', plot_type=LinePlot)

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
    # demo_debug_dbplot()