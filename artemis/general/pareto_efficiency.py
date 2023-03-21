from artemis.general.should_be_builtins import all_equal

__author__ = 'peter'
import numpy as np


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def is_pareto_efficient_reordered(costs):
    ixs = np.argsort(((costs-costs.mean(axis=0))/(costs.std(axis=0)+1e-7)).sum(axis=1))
    costs = costs[ixs]
    is_efficient = is_pareto_efficient_simple(costs)
    is_efficient[ixs] = is_efficient.copy()
    return is_efficient


def is_pareto_efficient_indexed_reordered(costs, return_mask=True):
    ixs = np.argsort(((costs-costs.mean(axis=0))/(costs.std(axis=0)+1e-7)).sum(axis=1))
    costs = costs[ixs]
    is_efficient = is_pareto_efficient(costs, return_mask=return_mask)
    is_efficient[ixs] = is_efficient.copy()
    return is_efficient


def find_pareto_ixs(cost_arrays):
    """
    :param cost_arrays: A collection of nd-arrays representing a grid of costs for different indices.
    :return: A tuple of indices which can be used to index the pareto-efficient points.
    """
    assert all_equal([c.shape for c in cost_arrays])
    flat_ixs, = np.nonzero(is_pareto_efficient_simple(np.reshape(cost_arrays, (len(cost_arrays), -1)).T), )
    ixs = np.unravel_index(flat_ixs, dims=cost_arrays[0].shape)
    return ixs
