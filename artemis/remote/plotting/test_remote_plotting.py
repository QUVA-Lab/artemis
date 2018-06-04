import time
import numpy as np
import sys
sys.path.extend(["/home/mreisser/PycharmProjects/artemis"])
from artemis.plotting.db_plotting import dbplot
from artemis.plotting.matplotlib_backend import set_server_plotting


def test_simple_plot():
    dbplot(np.random.randn(20, 20), 'Greyscale Image')
    time.sleep(2)
    dbplot(np.random.randn(20, 20), 'Greyscale Image')
    time.sleep(2)

if __name__ == "__main__":
    set_server_plotting(True)
    test_simple_plot()