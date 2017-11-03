from artemis.config import get_artemis_config_value
from artemis.general.test_mode import set_test_mode
from artemis.plotting.db_plotting import dbplot
from artemis.plotting.web_backend import setup_web_plotting
import numpy as np
from matplotlib import pyplot as plt
from six.moves import xrange


__author__ = 'peter'


def test_plotting_server():

    if get_artemis_config_value(section='plotting', option='backend') != 'matplotlib-web':
        setup_web_plotting()

    for i in xrange(5):
        dbplot(np.random.randn(10, 10, 3), 'noise')
        dbplot(np.random.randn(20, 2), 'lines')
        plt.pause(0.1)


if __name__ == '__main__':
    set_test_mode(True)
    test_plotting_server()
