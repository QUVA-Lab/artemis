import time
from artemis.plotting.db_plotting import dbplot
import numpy as np
__author__ = 'peter'


def demo_dbplot():

    for i in xrange(1000):
        dbplot(np.random.randn(20, 20), 'Greyscale Image')
        dbplot(np.random.randn(20, 20, 3), 'Colour Image')
        dbplot(np.random.randn(20, 2), 'Two Lines')
        dbplot(np.cos([i/10., i/10.+np.pi/2]), 'History')
        dbplot(np.random.randn(20), 'History Image')
        dbplot(np.random.randn(15, 20, 20), "Many Images")
        time.sleep(0.1)


if __name__ == '__main__':
    demo_dbplot()
