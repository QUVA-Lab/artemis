import numpy as np
from artemis.general.ezprofile import EZProfiler
from artemis.general.pareto_efficiency import is_pareto_efficient_ixs, is_pareto_efficient_dumb, \
    is_pareto_efficient

__author__ = 'peter'


def test_is_pareto_efficient(plot=False):

    for n_costs in (2, 10):

        rng = np.random.RandomState(1234)

        costs = rng.rand(1000, n_costs)
        ixs = is_pareto_efficient_ixs(costs)

        assert np.sum(ixs)>0
        for c in costs[ixs]:
            assert np.all(np.any(c<=costs, axis=1))

        if plot and n_costs==2:
            import matplotlib.pyplot as plt
            plt.plot(costs[:, 0], costs[:, 1], '.')
            plt.plot(costs[ixs, 0], costs[ixs, 1], 'ro')
            plt.show()


def profile_pareto_efficient():

    rng = np.random.RandomState(1234)

    costs = rng.rand(5000, 2)

    with EZProfiler('dumb'):
        dumb_ixs = is_pareto_efficient_dumb(costs)

    with EZProfiler('smart'):
        less_dumb__ixs = is_pareto_efficient(costs)
    assert np.array_equal(dumb_ixs, less_dumb__ixs)

    with EZProfiler('index-tracking'):
        smart_ixs = is_pareto_efficient_ixs(costs)

    assert np.array_equal(dumb_ixs, smart_ixs)


if __name__ == '__main__':
    test_is_pareto_efficient()
