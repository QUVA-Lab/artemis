import numpy as np
import pytest
from six.moves import xrange

from artemis.general.global_vars import global_context
from artemis.ml.tools.running_averages import RunningAverage, RecentRunningAverage, get_global_running_average

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


def test_get_global_running_average():

    n_steps = 100

    rng = np.random.RandomState(1234)

    sig = 2.5*(1-np.exp(-np.linspace(0, 10, n_steps)))
    noise = rng.randn(n_steps)*0.1
    fullsig = sig + noise
    with global_context():
        for x in fullsig:
            ra = get_global_running_average(x, 'my_ra_simple', ra_type='simple')
        assert 2.24 < ra < 2.25
        for x in fullsig:
            ra = get_global_running_average(x, 'my_ra_recent', ra_type='recent')
        assert 2.490 < ra < 2.491
        for x in fullsig:
            ra = get_global_running_average(x, 'my_ra_osa', ra_type='osa')
        assert 2.44 < ra < 2.45


if __name__ == '__main__':
    test_get_global_running_average()