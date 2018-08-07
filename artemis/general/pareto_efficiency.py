from artemis.general.should_be_builtins import all_equal

__author__ = 'peter'
import numpy as np


def is_pareto_efficient_dumb(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs>=c, axis=1))
    return is_efficient


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient


def is_pareto_efficient_indexed(costs, return_mask = True):
    """
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
        nondominated_point_mask = np.any(costs<=costs[next_point_index], axis=1)
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def find_pareto_ixs(cost_arrays):
    """
    :param cost_arrays: A collection of nd-arrays representing a grid of costs for different indices.
    :return: A tuple of indices which can be used to index the pareto-efficient points.
    """
    assert all_equal([c.shape for c in cost_arrays])
    flat_ixs, = np.nonzero(is_pareto_efficient(np.reshape(cost_arrays, (len(cost_arrays), -1)).T), )
    ixs = np.unravel_index(flat_ixs, dims=cost_arrays[0].shape)
    return ixs
