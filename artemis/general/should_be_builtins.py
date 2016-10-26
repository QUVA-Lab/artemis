from collections import OrderedDict
import itertools

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

    memoization_wrapper.wrapped_fcn = fcn

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


def reducemap(func, sequence, initial=None, include_zeroth = False):
    """
    A version of reduce that also returns the intermediate values.
    :param func: A function of the form x_i_plus_1 = f(x_i, params_i)
        Where:
            x_i is the value passed through the reduce.
            params_i is the i'th element of sequence
            x_i_plus_i is the value that will be passed to the next step
    :param sequence: A list of parameters to feed at each step of the reduce.
    :param initial: Optionally, an initial value (else the first element of the sequence will be taken as the initial)
    :param include_zeroth: Include the initial value in the returned list.
    :return: A list of length: len(sequence), (or len(sequence)+1 if include_zeroth is True) containing the computed result of each iteration.
    """
    if initial is None:
        val = sequence[0]
        sequence = sequence[1:]
    else:
        val = initial
    results = [val] if include_zeroth else []
    for s in sequence:
        val = func(val, s)
        results.append(val)
    return results


def itermap(func, initial, n_steps=None, stop_func = None, include_zeroth = False):
    """
    Iterively call a function with the output of the previous call.
    :param func: A function of the form x_i_plus_1 = f(x_i)
    :param n_steps: The number of times to iterate
    :param initial: An initial value
    :param stop_func: Optionally, a function returning a boolean that, if true, causes the iteration to terminate (after the value has been added)
    :param include_zeroth: Include the initial value in the returned list.
    :return:  A list of length: n_steps, (or n_steps+1 if include_zeroth is True) containing the computed result of each iteration.
    """
    assert (n_steps is not None) or (stop_func is not None), 'You must either specify a number of steps or a stopping function.'
    val = initial
    results = [val] if include_zeroth else []
    for _ in (xrange(n_steps) if n_steps is not None else itertools.count(start=0, step=1)):
        val = func(val)
        results.append(val)
        if stop_func is not None and stop_func(val):
            break
    return results


def izip_equal(*iterables):
    """
    Zip and raise exception if lengths are not equal.

    Taken from solution by Martijn Pieters, here:
    http://stackoverflow.com/questions/32954486/zip-iterators-asserting-for-equal-length-in-python

    :param iterables:
    :return:
    """
    sentinel = object()
    for combo in itertools.izip_longest(*iterables, fillvalue=sentinel):
        if any(sentinel is c for c in combo):
            raise ValueError('Iterables have different lengths')
        yield combo


def remove_duplicates(sequence):
    """
    Remove duplicates while maintaining order.
    Credit goes to Markus Jarderot from http://stackoverflow.com/a/480227/851699
    """
    seen = set()
    seen_add = seen.add
    return [x for x in sequence if not (x in seen or seen_add(x))]
