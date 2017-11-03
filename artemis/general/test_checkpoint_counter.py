from itertools import count
from artemis.general.checkpoint_counter import CheckPointCounter, Checkpoints

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

if __name__ == '__main__':
    test_checkpoint_counter()
    test_checkpoints()