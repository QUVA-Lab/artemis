import numpy as np
import pytest
from six.moves import xrange

from artemis.ml.tools.processors import RunningAverage, RecentRunningAverage

__author__ = 'peter'


def test_running_average():

    inp = np.arange(5)
    processor = RunningAverage()
    out = [processor(el) for el in inp]
    assert out == [0, 0.5, 1, 1.5, 2]
    assert np.array_equal(out, RunningAverage.batch(inp))

    inp = np.random.randn(10, 5)
    processor = RunningAverage()
    out = [processor(el) for el in inp]
    assert all(np.allclose(out[i], np.mean(inp[:i+1], axis = 0)) for i in xrange(len(inp)))


@pytest.mark.skipif(True, reason='Depends on weave, which is deprecated for python 3')
def test_recent_running_average():

    inp = np.arange(5)
    processor = RecentRunningAverage()
    out = [processor(el) for el in inp]
    out2 = processor.batch(inp)
    assert np.allclose(out, out2)
    assert np.allclose(out, [0.0, 0.7071067811865475, 1.4535590291019362, 2.226779514550968, 3.019787823462811])

    inp = np.random.randn(10, 5)
    processor = RunningAverage()
    out = [processor(el) for el in inp]
    out2 = processor.batch(inp)
    assert np.allclose(out, out2)


if __name__ == '__main__':

    # test_running_average()
    test_recent_running_average()