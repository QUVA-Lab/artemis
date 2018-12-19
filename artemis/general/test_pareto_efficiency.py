import numpy as np
from artemis.general.ezprofile import EZProfiler
from artemis.general.pareto_efficiency import is_pareto_efficient_dumb, \
    is_pareto_efficient_simple, is_pareto_efficient, is_pareto_efficient_reordered, \
    is_pareto_efficient_indexed_reordered

__author__ = 'peter'


def test_is_pareto_efficient(plot=False):

    for pareto_func in [is_pareto_efficient_dumb, is_pareto_efficient_simple, is_pareto_efficient]:
        for n_costs in (2, 10):

            rng = np.random.RandomState(1234)

            costs = rng.rand(1000, n_costs)
            ixs = pareto_func(costs)

            assert np.sum(ixs)>0
            for c in costs[ixs]:
                assert np.all(np.any(c<=costs, axis=1))

            if plot and n_costs==2:
                import matplotlib.pyplot as plt
                plt.plot(costs[:, 0], costs[:, 1], '.')
                plt.plot(costs[ixs, 0], costs[ixs, 1], 'ro')
                plt.show()


def test_is_pareto_efficient_integer():

    assert np.array_equal(is_pareto_efficient_dumb(np.array([[1,2], [3,4], [2,1], [1,1]])), [False, False, False, True])
    assert np.array_equal(is_pareto_efficient_simple(np.array([[1, 2], [3, 4], [2, 1], [1, 1]])), [False, False, False, True])
    assert np.array_equal(is_pareto_efficient(np.array([[1, 2], [3, 4], [2, 1], [1, 1]])), [False, False, False, True])


def profile_pareto_efficient(n_points=5000, n_costs=2, include_dumb = True):

    rng = np.random.RandomState(1234)

    costs = rng.randn(n_points, n_costs)

    print('{} samples, {} costs'.format(n_points, n_costs))

    if include_dumb:
        with EZProfiler('is_pareto_efficient_dumb'):
            base_ixs = dumb_ixs = is_pareto_efficient_dumb(costs)
    else:
        print('is_pareto_efficient_dumb: Really, really, slow')

    with EZProfiler('is_pareto_efficient_simple'):
        less_dumb__ixs = is_pareto_efficient_simple(costs)
        if not include_dumb:
            base_ixs = less_dumb__ixs
    assert np.array_equal(base_ixs, less_dumb__ixs)

    with EZProfiler('is_pareto_efficient_reordered'):
        reordered_ixs = is_pareto_efficient_reordered(costs)
    assert np.array_equal(base_ixs, reordered_ixs)

    with EZProfiler('is_pareto_efficient'):
        smart_indexed = is_pareto_efficient(costs, return_mask=True)
    assert np.array_equal(base_ixs, smart_indexed)

    with EZProfiler('is_pareto_efficient_indexed_reordered'):
        smart_indexed = is_pareto_efficient_indexed_reordered(costs, return_mask=True)
    assert np.array_equal(base_ixs, smart_indexed)


if __name__ == '__main__':
    test_is_pareto_efficient()
    test_is_pareto_efficient_integer()
    profile_pareto_efficient(n_points=10000, n_costs=2, include_dumb=True)
    profile_pareto_efficient(n_points=1000000, n_costs=2, include_dumb=False)
    profile_pareto_efficient(n_points=10000, n_costs=15, include_dumb=True)
