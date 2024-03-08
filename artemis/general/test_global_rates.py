import itertools
import time

from artemis.general.global_rates import limit_rate, limit_iteration_rate
from artemis.general.global_vars import global_context


def test_limit_rate():

    with global_context():
        start = time.time()
        for t in itertools.count(0):
            limit_rate('this_rate', period=0.1)
            current = time.time()
            if current - start > 0.5:
                break
            print((t, current - start))
    assert t<6


def test_limit_rate_iterator():
    with global_context():
        start = time.time()
        for t in limit_iteration_rate(itertools.count(0), period=0.1):
            current = time.time()
            if current - start > 0.5:
                break
            print((t, current - start))
    assert t<6


if __name__ == '__main__':
    test_limit_rate()
    test_limit_rate_iterator()
