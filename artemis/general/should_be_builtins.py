import inspect
from collections import OrderedDict
import itertools
import os

import math
from six.moves import xrange, zip_longest

__author__ = 'peter'

all_equal_deprecated = lambda *args: all(a == args[0] for a in args[1:])


def all_equal(elements):
    """
    :param elements: A collection of things
    :return: True if all things are equal, otherwise False.  (Note that an empty list of elements returns true, just as all([]) is True
    """
    element_iterator = iter(elements)
    try:
        first = next(element_iterator) # Will throw exception
    except StopIteration:
        return True
    return all(a == first for a in element_iterator)


def all_equal_length(collection_if_collections):
    """
    :param collection_if_collections: A collection of collections
    :return: True if all collections have equal length, otherwise False
    """
    return all_equal([len(c) for c in collection_if_collections])


def is_lambda(v):
    '''
    Source http://stackoverflow.com/a/3655857/2068168
    :param v:
    :return:
    '''
    LAMBDA = lambda:0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

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
        return ('memoizationidentifier_ordereddict ',) + tuple((arg_signature(k), arg_signature(v)) for k, v in arg.items())
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
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if any(sentinel is c for c in combo):
            raise ValueError('Iterables have different lengths')
        yield combo


def remove_duplicates(sequence, hashable=True, key=None, keep_last=False):
    """
    Remove duplicates while maintaining order.
    :param sequence: The sequence you want to remove the duplicates in
    :param hashable: Set to True if your sequence contains unhashable items.  If your items are not hashable,
        you can set hashable=False.  The function will be slower, and test for dupicates based on equality rather than
        hash, which may in some cases give different results.
    :param key: Optionally, a function that extracts the identity from each element in sequence.  This allows you to be
        flexible in what you consider to be a "duplicate"
    :param keep_last: Keep the last element, rather than the first (only makes sense if key is not None)
    :returns: A list that maintains the order, but with duplicates removed
    """
    is_dup = detect_duplicates(sequence, hashable=hashable, key=key, keep_last=keep_last)
    return [x for x, is_duplicate in zip(sequence, is_dup) if not is_duplicate]


def uniquify_duplicates(sequence_of_strings):

    counts = {}
    new_strings = []
    for string in sequence_of_strings:
        if string in counts:
            new_strings.append(string+'[{}]'.format(counts[string]))
            counts[string] += 1
        else:
            counts[string]=1
            new_strings.append(string)
    return new_strings


def get_unique_name(name, taken_names):
    """
    Given a name, if it is already in the list of taken names, append with name(1), then name(2)
    :param name:
    :param taken_names:
    :return:
    """
    if name in taken_names:
        for i in itertools.count(1):
            new_name = name+'({})'.format(i) if isinstance(name, str) else name + (i, ) if isinstance(name, tuple) else bad_value(name)
            if new_name not in taken_names:
                name = new_name
                break
    return name


def detect_duplicates(sequence, hashable=True, key=None, keep_last=False):
    """
    Identify whether each element in a sequence is a duplicate of a previously existing element.

    Derived from solution by goes to Markus Jarderot from http://stackoverflow.com/a/480227/851699

    :param sequence: The sequence you want to remove the duplicates in
    :param hashable: Set to True if your sequence contains unhashable items.  If your items are not hashable,
        you can set hashable=False.  The function will be slower, and test for dupicates based on equality rather than
        hash, which may in some cases give different results.
    :param key: Optionally, a function that extracts the identity from each element in sequence.  This allows you to be
        flexible in what you consider to be a "duplicate"
    :param keep_last: Keep the last element, rather than the first.
    :returns: A list of booleans, which are True if the item is a duplicate and False otherwise.
    """
    if keep_last:
        sequence = sequence[::-1]
    if key is None:
        key = lambda x: x
    if hashable:
        seen = set()
        seen_add = seen.add
    else:
        seen = list()
        seen_add = seen.append
    if key is not None:
        sequence = [key(x) for x in sequence]
    is_dup = [(x in seen or seen_add(x) is 'This is such a hack') for x in sequence]
    if keep_last:
        is_dup = is_dup[::-1]
    return is_dup


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
    common_items = remove_duplicates([k for k, c in count_unique_items(all_items) if c==len(list_of_lists)], hashable=False)
    different_items = [[item for item in list_of_items if item not in common_items] for list_of_items in list_of_lists]
    if are_dicts:
        return dict(common_items), [dict(el) for el in different_items]
    else:
        return common_items, different_items


def count_unique_items(items):
    """
    Count the unique items in a list.  Similar to calling collections.Counter(items).items(), but it doesn't require
    that items be hashable.
    :return: A list<(item, item_count)>
    """
    unique_items = []
    unique_item_counts = []
    for item in items:
        if item in unique_items:
            unique_item_counts[unique_items.index(item)] +=1
        else:
            unique_items.append(item)
            unique_item_counts.append(1)
    return zip(unique_items, unique_item_counts)


def check(value, condition, string = ""):
    """
    Verify that the condition is true and return the value.

    Useful for conditional assignments, eg:
        xs = [x]*n_elements if not isinstance(x, (list, tuple)) else check(x, len(x)==n_elements)

    :param value:
    :param condition:
    :param string:
    :return:
    """
    assert condition, string
    return value


def remove_common_prefix(list_of_lists, max_elements=None, keep_base = True):
    """
    Remove common elements starting each list in the list of lists.

        assert remove_common_prefix([[1, 2, 3, 4], [1, 2, 5], [1, 2, 3, 5]]) == [[3, 4], [5], [3, 5]]

    :param list_of_lists: A list of lists
    :param max_elements: Maximum number of elements to delete
    :return: Truncated list of lists
    """

    count = 0

    min_len = 1 if keep_base else 0

    while min(len(parts) for parts in list_of_lists)>min_len:
        if max_elements is not None and count >= max_elements:
            break

        if all_equal_deprecated(*[parts[0] for parts in list_of_lists]):
            list_of_lists = [parts[1:] for parts in list_of_lists]
        else:
            break
        count += 1
    return list_of_lists


def remove_common_string_prefix(list_of_strings, separator = '', max_elements = None):

    list_of_lists = [list(string) for string in list_of_strings] if separator=='' else [string.split(separator) for string in list_of_strings]
    shortened_list_of_lists = remove_common_prefix(list_of_lists, max_elements=max_elements)
    return [separator.join(strlist) for strlist in shortened_list_of_lists]


def get_absolute_module(obj):
    """
    Get the abolulte path to the module for the given object.

        e.g. assert get_absolute_module(get_absolute_module) == 'artemis.general.should_be_builtins'

    :param obj: A python module, class, method, function, traceback, frame, or code object
    :return: A string representing the import path.
    """
    file_path = inspect.getfile(obj)
    return file_path_to_absolute_module(file_path)


def file_path_to_absolute_module(file_path):
    """
    Given a file path, return an import path.
    :param file_path: A file path.
    :return:
    """
    assert os.path.exists(file_path)
    file_loc, ext = os.path.splitext(file_path)
    assert ext in ('.py', '.pyc')
    directory, module = os.path.split(file_loc)
    module_path = [module]
    while True:
        if os.path.exists(os.path.join(directory, '__init__.py')):
            directory, package = os.path.split(directory)
            module_path.append(package)
        else:
            break
    path = '.'.join(module_path[::-1])
    return path


def assert_option(choice, possiblilties):
    assert choice in possiblilties, '"{}" was not in the list of possible choices: {}'.format(choice, possiblilties)


def insert_at(list1, list2, indices):
    """
    Create a new list by insert elements from list 2 into list 1 at the given indices.
    (Note: this leaves list1 and list2 unchanged, unlike list.insert)
    :param list1: A list
    :param list2: Another list
    :param indices: The indices of list1 into which elements from list2 will be inserted.
    :return: A new list with len(list1)+len(list2) elements.
    """
    list3 = []
    assert len(list2)==len(indices), 'List 2 has {} elements, but you provided {} indices.  They should have equal length'.format(len(list2), len(indices))
    index_iterator = iter(sorted(indices))
    list_2_iter = iter(list2)
    next_ix = next(index_iterator)

    iter_stopped = False
    for i in xrange(len(list1)+1):
        while i == next_ix:
            list3.append(next(list_2_iter))
            try:
                next_ix = next(index_iterator)
            except StopIteration:
                next_ix = None
                iter_stopped = True
        if i<len(list1):
            list3.append(list1[i])

    assert iter_stopped, 'Not all elements from list 2 got used!'
    return list3


try:
    from contextlib import nested  # Python 2
except ImportError:
    from contextlib import ExitStack, contextmanager

    @contextmanager
    def nested(*contexts):
        """
        Reimplementation of nested in python 3.
        """
        with ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            yield contexts


def get_shifted_element(list_of_elements, element, shift):
    key_ix = list_of_elements.index(element)
    new_key = list_of_elements[key_ix+shift]
    return new_key


def get_shifted_key_value(orderd_dict, key, shift):
    """
    Given an OrderedDict, get the value at a key which is offset from the given key by shift.
    :param orderd_dict:
    :param key:
    :param shift:
    :return: The value at the shifted key
    """
    assert isinstance(orderd_dict, OrderedDict)
    keylist = list(orderd_dict.keys())
    key_ix = keylist.index(key)
    new_key = keylist[key_ix+shift]
    return orderd_dict[new_key]


def divide_into_subsets(list_of_element, subset_size):
    """
    Given a list of elements, divide into subsets.  e.g. divide_into_subsets([1,2,3,4,5], subset_size=2) == [[1, 2], [3, 4], [5]]
    :param list_of_element:
    :param subset_size:
    :return:
    """
    element_gen = (el for el in list_of_element)
    return [[nextel for _, nextel in zip(range(subset_size), element_gen)] for _ in range(int(math.ceil(float(len(list_of_element))/subset_size)))]
