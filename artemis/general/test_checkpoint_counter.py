from itertools import count
from artemis.general.checkpoint_counter import CheckPointCounter, Checkpoints
import numpy as np
__author__ = 'peter'


def test_checkpoint_counter():

    points = [0, 0.2, 0.4, 0.6, 0.9]

    cpc = CheckPointCounter(points)

    checkpoints = []
    for i in count(0):
        points_passed, done = cpc.check(i/100.)
        if points_passed:
            checkpoints.append(i)
            if done:
                break

    assert len(checkpoints)==len(points)
    assert checkpoints == [0, 20, 40, 60, 90]
    # Only because all points are spaced wider than progress increments.
    # More generally len(checkpoints) will be <= len(points)


def test_checkpoints():

    is_test = Checkpoints(('exp', 10, .1))
    assert [a for a in range(100) if is_test()]==[0, 10, 22, 37, 54, 74, 97]

    is_test = Checkpoints({0: 0.25, 0.75: 0.5, 2.: 1})
    assert np.allclose([a for a in np.arange(0, 6, 0.1) if is_test(a)], [0, 0.3, 0.5, 0.8, 1.3, 1.8, 2.3, 3.3, 4.3, 5.3])

    is_test = Checkpoints({1: 0.5, 2: 1, 5: 3})
    assert np.allclose([a for a in np.arange(0, 12, 0.1) if is_test(a)], [1, 1.5, 2, 3, 4, 5, 8, 11])


if __name__ == '__main__':
    test_checkpoint_counter()
    test_checkpoints()