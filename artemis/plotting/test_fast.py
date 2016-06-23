from artemis.general.test_mode import set_test_mode
from artemis.plotting.fast import find_interval_extremes, fastplot, fastloglog
import matplotlib.pyplot as plt

__author__ = 'peter'
import numpy as np


def test_fastplot():

    plt.ion()
    fastplot(np.random.randn(25000))
    plt.show()
    fastloglog(np.random.rand(25000))
    plt.show()
    fastplot(np.random.randn(1000))
    plt.show()


def test_find_interval_extremes():

    arr = np.random.RandomState(324).randn(1000)
    edges = np.linspace(0, 1000, 10)[1:]
    extreme_indices = find_interval_extremes(arr, edges)
    assert len(extreme_indices) <= len(edges)*2
    assert np.std(arr[extreme_indices]) > 2* np.std(arr)  # Good enough test


if __name__ == '__main__':
    set_test_mode(True)
    test_fastplot()
    test_find_interval_extremes()
