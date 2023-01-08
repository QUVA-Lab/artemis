import itertools
from pytest import raises

from artemis.general.utils_utils import tee_and_specialize_iterator


def test_tee_and_specialize_iterator():
    first_iterator = [{'a': i + 1, 'b': i + 2, 'c': i + 3} for i in range(3)]

    # Naive way - show that it doesnt work, because of lazy-binding
    teed_it = ((d[n] for d in it_copy) for n, it_copy in zip('abc', itertools.tee(iter(first_iterator), 3)))

    items = list(zip(*teed_it))
    with raises(AssertionError):
        assert items == ([1, 2, 3], [2, 3, 4], [3, 4, 5])
    assert items == [(3, 3, 3), (4, 4, 4), (5, 5, 5)]

    teed_it_2 = tee_and_specialize_iterator(iter(first_iterator), specialization_func=lambda it, arg: it[arg], args='abc')
    items = list(zip(*teed_it_2))
    assert items == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


if __name__ == '__main__':
    test_tee_and_specialize_iterator()
