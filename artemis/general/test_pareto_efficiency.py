import numpy as np
from artemis.general.ezprofile import EZProfiler
from artemis.general.pareto_efficiency import is_pareto_efficient_dumb, \
    is_pareto_efficient, is_pareto_efficient_indexed

__author__ = 'peter'


def test_is_pareto_efficient(plot=False):

    for n_costs in (2, 10):

        rng = np.random.RandomState(1234)

        costs = rng.rand(1000, n_costs)
        ixs = is_pareto_efficient_indexed(costs, return_mask=True)

        assert np.sum(ixs)>0
        for c in costs[ixs]:
            assert np.all(np.any(c<=costs, axis=1))

        if plot and n_costs==2:
            import matplotlib.pyplot as plt
            plt.plot(costs[:, 0], costs[:, 1], '.')
            plt.plot(costs[ixs, 0], costs[ixs, 1], 'ro')
            plt.show()


def profile_pareto_efficient(n_points=5000, n_costs=2, include_dumb = True):

    rng = np.random.RandomState(1234)

    costs = rng.rand(n_points, n_costs)

    if include_dumb:
        with EZProfiler('is_pareto_efficient_dumb'):
            base_ixs = dumb_ixs = is_pareto_efficient_dumb(costs)

    with EZProfiler('is_pareto_efficient'):
        less_dumb__ixs = is_pareto_efficient(costs)
        if not include_dumb:
            base_ixs = less_dumb__ixs
    assert np.array_equal(base_ixs, less_dumb__ixs)

    with EZProfiler('is_pareto_efficient_indexed'):
        smart_indexed = is_pareto_efficient_indexed(costs, return_mask=True)
    assert np.array_equal(base_ixs, smart_indexed)

    with EZProfiler('is_pareto_efficient_indexed_reordered'):
        smart_indexed = is_pareto_efficient_indexed(costs, return_mask=True, rank_reorder=True)
    assert np.array_equal(base_ixs, smart_indexed)


if __name__ == '__main__':
    # test_is_pareto_efficient()
    profile_pareto_efficient(n_points=100000, n_costs=2, include_dumb=False)
