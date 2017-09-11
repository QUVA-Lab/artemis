import itertools

__author__ = 'peter'


def cross_dict_dicts(*dicts):
    """
    Combine two or more dictionaries of dictionaries by turning every pairwise combination of their keys, and creating a
    new dict whose keys are tuples (containing these key-combinations) and whose values are the the combined dictionionaries.

    e.g.
        cross_dict_dicts({'a':{'aa': 1}, 'b':{'bb': 2}}, {'c': {'cc': 3}, 'd': {'dd': 4}})

        returns {
            ('a','c'):{'aa':1, 'cc':3},
            ('a','d'):{'aa':1, 'dd':4},
            ('b','c'):{'bb':2, 'cc':3},
            ('b','d'):{'bb':2, 'dd':4},
            }

    This can be useful if, for example, you want to try all several combinations of different arguments to a function.

    :param dicts: Dictionaries of dictionaries.
    :return: A Dictionary of dictionaries.
    """
    cross_dict = dict((keys, merge_dicts(*[d[k] for d, k in zip(dicts, keys)])) for keys in itertools.product(*[d.keys() for d in dicts]))
    return cross_dict


def merge_dicts(*dicts):
    """
    Given a collection of dictionaries, merge them.

    e.g.
        merge_dicts({'a': 1, 'b': 2}, {'c': 3, 'd': 4})
        returns {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    Later dicts overwrite earlier ones.

    :param dicts: dictionaries.
    :return: A merged dictionary.
    """
    return dict((k, v) for d in dicts for k, v in d.items())
