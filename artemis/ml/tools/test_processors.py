import numpy as np

from artemis.ml.tools.processors import RunningAverage


__author__ = 'peter'


def test_running_average():

    inp = np.arange(5)
    processor = RunningAverage()
    out = [processor(el) for el in inp]
    assert out == [0, 0.5, 1, 1.5, 2]

    inp = np.random.randn(10, 5)
    processor = RunningAverage()
    out = [processor(el) for el in inp]
    assert all(np.allclose(out[i], np.mean(inp[:i+1], axis = 0)) for i in xrange(len(inp)))


if __name__ == '__main__':

    test_running_average()
