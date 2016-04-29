from collections import OrderedDict

__author__ = 'peter'

all_equal = lambda *args: all(a == args[0] for a in args[1:])


def bad_value(value, explanation = None):
    """
    :param value: Raise ValueError.  Useful when doing conditional assignment.
    e.g.
    dutch_hand = 'links' if eng_hand=='left' else 'rechts' if eng_hand=='right' else bad_value(eng_hand)
    """
    raise ValueError('Bad Value: %s%s' % (value, ': '+explanation if explanation is not None else ''))


def memoize(fcn):
    """
    Use this to decorate a function whose results you want to cache.
    """
    lookup = {}

    def memoization_wrapper(*args, **kwargs):
        # arg_signature = args + ('5243643254_kwargs_start_here', ) + tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
        hashable_arg_structure = arg_signature((args, kwargs))
        if hashable_arg_structure in lookup:
            return lookup[hashable_arg_structure]
        else:
            out = fcn(*args, **kwargs)
            lookup[hashable_arg_structure]=out
            return out

    return memoization_wrapper


def arg_signature(arg):
    """
    Turn the argument into something hashable
    """
    if isinstance(arg, tuple):
        return tuple(arg_signature(a) for a in arg)
    elif isinstance(arg, list):
        return ('memoizationidentifier_list ',) + tuple(arg_signature(a) for a in arg)
    elif isinstance(arg, OrderedDict):
        return ('memoizationidentifier_ordereddict ',) + tuple((arg_signature(k), arg_signature(v)) for k, v in arg.iteritems())
    elif isinstance(arg, dict):
        return ('memoizationidentifier_dict ',) + tuple((arg_signature(k), arg_signature(arg[k])) for k in sorted(arg.keys()))
    else:
        return arg
