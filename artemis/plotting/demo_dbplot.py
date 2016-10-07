from artemis.general.ezprofile import EZProfiler
from artemis.plotting.db_plotting import dbplot
import numpy as np


__author__ = 'peter'


def demo_dbplot():

    with EZProfiler('plot time') as prof:
        for i in xrange(1000):
            dbplot(np.random.randn(20, 20), 'Greyscale Image', draw_now=False)
            dbplot(np.random.randn(20, 20, 3), 'Colour Image', draw_now=False)
            dbplot(np.random.randn(20, 2), 'Two Lines', draw_now=False)
            dbplot(np.cos([i/10., i/10.+np.pi/2]), 'History', draw_now=False)
            dbplot(np.random.randn(20), 'History Image', draw_now=False)
            dbplot(np.random.randn(15, 20, 20), "Many Images")
            if i % 10 == 0:
                print 'Mean Frame Rate: %.3gFPS' % ((i+1)/prof.get_current_time(), )

if __name__ == '__main__':
    demo_dbplot()
