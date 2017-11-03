import numpy as np
import pytest
from artemis.general.nondeterminism_hunting import delete_vars, assert_variable_matches_between_runs, variable_matches_between_runs, \
    reset_variable_tracker


def _runs_are_the_same(var_gen_1, var_gen_2, use_assert = False):

    delete_vars(['_test_random_var_32r5477w32'])
    for run, gen in [(0, var_gen_1), (1, var_gen_2)]:
        reset_variable_tracker()
        for v in gen:
            if use_assert:
                assert_variable_matches_between_runs(v, '_test_random_var_32r5477w32')
            else:
                its_a_match=variable_matches_between_runs(v, '_test_random_var_32r5477w32')
                if run==0:
                    assert its_a_match is None
                else:
                    if not its_a_match:
                        return False
    return True


def test_variable_matches_between_runs():

    rng1 = np.random.RandomState(1234)
    gen1 = (rng1.randn(3, 4) for _ in range(5))
    rng2 = np.random.RandomState(1234)
    gen2 = (rng2.randn(3, 4) for _ in range(5))
    assert _runs_are_the_same(gen1, gen2)

    rng = np.random.RandomState(1234)
    gen1 = (rng.randn(3, 4) for _ in range(5))
    gen2 = (rng.randn(3, 4) for _ in range(5))
    assert not _runs_are_the_same(gen1, gen2)

    gen1 = (i for i in range(5))
    gen2 = (i for i in range(5))
    assert _runs_are_the_same(gen1, gen2)

    gen1 = (i for i in range(5))
    gen2 = (i if i<4 else 7 for i in range(5))
    assert not _runs_are_the_same(gen1, gen2)


def test_assert_variable_matches_between_runs():

    rng1 = np.random.RandomState(1234)
    gen1 = (rng1.randn(3, 4) for _ in range(5))
    rng2 = np.random.RandomState(1234)
    gen2 = (rng2.randn(3, 4) for _ in range(5))
    _runs_are_the_same(gen1, gen2, use_assert=True)

    rng = np.random.RandomState(1234)
    gen1 = (rng.randn(3, 4) for _ in range(5))
    gen2 = (rng.randn(3, 4) for _ in range(5))
    with pytest.raises(AssertionError):
        _runs_are_the_same(gen1, gen2, use_assert=True)

    gen1 = (i for i in range(5))
    gen2 = (i for i in range(5))
    _runs_are_the_same(gen1, gen2, use_assert=True)

    gen1 = (i for i in range(5))
    gen2 = (i if i<4 else 7 for i in range(5))
    with pytest.raises(AssertionError):
        _runs_are_the_same(gen1, gen2, use_assert=True)


if __name__ == '__main__':
    test_variable_matches_between_runs()
    test_assert_variable_matches_between_runs()
