from collections import OrderedDict, Counter
import itertools
import contextlib

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


def try_key(dictionary, key, default):
    """
    Try to get the value at dict[key]
    :param dictionary: A Python dict
    :param key: A key
    :param default: The value to return if the key doesn't exist
    :return: Either dictionary[key] or default if it doesn't exist.
    """
    try:
        return dictionary[key]
    except KeyError:
        return default


def separate_common_items(list_of_lists):
    """
    Given a list of lists of items, separate out the items that are common between the sublists
    into a single lists.

    :param list_of_lists: A list of lists of items
    :return: common_items, list_of_lists_of_different_items
        common_items: A list of items that are common to all the sublists.
        different_items: A list of lists of items that are not common between all sublists
    """
    are_dicts = all(isinstance(el, dict) for el in list_of_lists)
    if are_dicts:
        list_of_lists = [el.items() for el in list_of_lists]
    all_items = [item for list_of_items in list_of_lists for item in list_of_items]
    all_identical = {k: c==len(list_of_lists) for k, c in Counter(all_items).iteritems()}
    common_items = remove_duplicates(item for item in all_items if all_identical[item])
    different_items = [[item for item in list_of_items if item not in common_items] for list_of_items in list_of_lists]
    if are_dicts:
        return dict(common_items), [dict(el) for el in different_items]
    else:
        return common_items, different_items


def get_final_args(args, kwargs, all_arg_names, default_kwargs):
    """
    Get the final arguments that python would feed into a function called as f(*args, **kwargs),
    where the function has arguments named all_arg_names, and defaults in default_kwargs

    :param args: A tuple of ordered arguments
    :param kwargs: A dict of keyword args
    :param all_arg_names: A list of all argument names
    :param default_kwargs: A dict of default kwargs, OR a list giving the values of the last len(default_kwargs) arguments
    :return: A tuple of 2-tuples of arg_name, arg_value
    """
    if isinstance(default_kwargs, (list, tuple)):
        default_kwargs = {k: v for k, v in zip(all_arg_names[-len(default_kwargs):], default_kwargs)}

    return tuple(
        zip(all_arg_names, args)  # Handle unnamed args f(1, 2)
        + [(name, kwargs[name] if name in kwargs else default_kwargs[name]) for name in
           all_arg_names[len(args):]]  # Handle named keyworkd args f(a=1, b=2)
        + [(name, kwargs[name]) for name in kwargs if name not in all_arg_names]
        )
