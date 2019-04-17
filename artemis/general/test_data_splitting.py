import numpy as np

from artemis.ml.tools.data_splitting import join_arrays_and_get_rebuild_func


def test_join_arrays_and_get_rebuild_function():

    n_samples = 5
    randn = np.random.RandomState(1234).randn

    struct = [
        (randn(n_samples, 3), randn(n_samples)),
        randn(n_samples, 4, 5)
    ]

    joined, rebuild_func = join_arrays_and_get_rebuild_func(struct, axis=1)

    assert joined.shape == (n_samples, 3+1+4*5)

    new_struct = rebuild_func(joined*2)

    assert np.array_equal(struct[0][0]*2, new_struct[0][0])
    assert np.array_equal(struct[0][1]*2, new_struct[0][1])
    assert np.array_equal(struct[1]*2, new_struct[1])


if __name__ == '__main__':
    test_join_arrays_and_get_rebuild_function()
