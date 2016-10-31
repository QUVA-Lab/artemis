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


def is_pareto_efficient_ixs(costs):

    candidates = np.arange(costs.shape[0])
    for i, c in enumerate(costs):
        if 0 < np.searchsorted(candidates, i) < len(candidates):  # If this element has not yet been eliminated
            candidates = candidates[np.any(costs[candidates]<=c, axis=1)]
    is_efficient = np.zeros(costs.shape[0], dtype = bool)
    is_efficient[candidates] = True
    return is_efficient
