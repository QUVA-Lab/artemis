from itertools import count
from artemis.general.checkpoint_counter import CheckPointCounter

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


if __name__ == '__main__':
    test_checkpoint_counter()
