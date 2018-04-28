from collections import OrderedDict

import numpy as np
from six import string_types, next

from artemis.general.should_be_builtins import all_equal

__author__ = 'peter'

_immutible_types = (int, float, bool, type(None))+string_types


def flatten_struct(struct, primatives = (int, float, np.ndarray, type(None), bool)+string_types, custom_handlers = {},
        break_into_objects = True, detect_duplicates = True, first_dict_is_namespace=False, memo = None):
    """
    Given some nested struct, return a list<*(str, primative)>, where primative
    is some some kind of object that you don't break down any further, and str is a
    string representation of how you would access that propery from the root object.

    :param struct: Something, anything.
    :param primatives: A list of classes that will not be broken into.
    :param custum_handlers: A dict<type:func> where func has the form data = func(obj).  These
        will be called if the type of the struct is in the dict of custom handlers.
    :param break_into_objects: True if you want to break into objects to see what's inside.
    :return: list<*(str , primative)>
    """
    if memo is None:
        memo = {}

    if isinstance(struct, primatives):
        return [(None, struct)]

    if not isinstance(struct, _immutible_types):
        if id(struct) in memo:
            return [(None, memo[id(struct)])]
        elif detect_duplicates:
            memo[id(struct)] = 'Already Seen object at %s' % hex(id(struct))

    if isinstance(struct, tuple(custom_handlers.keys())):
        handler = custom_handlers[custom_handlers.keys()[[isinstance(struct, t) for t in custom_handlers].index(True)]]
        return [(None, handler(struct))]
    elif isinstance(struct, dict):
        return [
            (("[{}]{}").format(("'{}'".format(key) if isinstance(key, string_types) else key), subkey if subkey is not None else ''), v) if not first_dict_is_namespace else
            (("{}{}").format(key, subkey if subkey is not None else ''), v)
            for key in (struct.keys() if isinstance(struct, OrderedDict) else sorted(struct.keys(), key = str))
            for subkey, v in flatten_struct(struct[key], custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)
            ]
    elif isinstance(struct, (list, tuple)):
        return [("[%s]%s" % (i, subkey if subkey is not None else ''), v)
            for i, value in enumerate(struct)
            for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)
            ]
    elif break_into_objects:  # It's some kind of object, lets break it down.
        return [(".%s%s" % (key, subkey if subkey is not None else ''), v)
            for key in sorted(struct.__dict__.keys(), key = str)
            for subkey, v in flatten_struct(struct.__dict__[key], custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)
            ]
    else:
        return [(None, memo[id(struct)])]


_primitive_containers = (list, tuple, dict, set)


def _is_primitive_container(obj):
    return isinstance(obj, _primitive_containers)


def get_meta_object(data_object, is_container_func = _is_primitive_container):
    """
    Given an arbitrary data structure, return a "meta object" which is the same structure, except all non-container
    objects are replaced by their types.

    e.g.
        get_meta_obj([1, 2, {'a':(3, 4), 'b':['hey', 'yeah']}, 'woo']) == [int, int, {'a':(int, int), 'b':[str, str]}, str]

    :param data_object: A data object with arbitrary nested structure
    :param is_container_func: A callback which returns True if an object is to be considered a container and False otherwise
    :return:
    """
    if is_container_func(data_object):
        if isinstance(data_object, (list, tuple, set)):
            return type(data_object)(get_meta_object(x, is_container_func=is_container_func) for x in data_object)
        elif isinstance(data_object, dict):
            return type(data_object)((k, get_meta_object(v, is_container_func=is_container_func)) for k, v in data_object.items())
    else:
        return type(data_object)


class NestedType(object):
    """
    An object which represents the type of an arbitrarily nested data structure.  It can be constructed directly
    from a nested type descriptor, or indirectly using the NestedType.from_data(...) constructor.

    For example
        NestedType.from_data([1, 2, {'a':(3, 4.), 'b':'c'}]) == NestedType([int, int, {'a':(int, float), 'b':str}])
    """

    def __init__(self, meta_object):
        """
        :param meta_object: A nested type descriptor.  See docstring and tests for examples.
        """
        self.meta_object = meta_object

    def is_type_for(self, data_object):
        return get_meta_object(data_object)==self.meta_object

    def check_type(self, data_object):
        """
        Assert that the data_object has a format matching this NestedType.  Throw a TypeError if it does not.
        :param data_object:
        :return:
        """
        if not self.is_type_for(data_object):  # note: we'd like to switch this to isnestedinstance
            raise TypeError('The data object has type {}, which does not match this format: {}'.format(NestedType.from_data(data_object), self))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.meta_object)

    def __eq__(self, other):
        return self.meta_object == other.meta_object

    def get_leaves(self, data_object, check_types = True, is_container_func = _is_primitive_container):
        """
        :param data_object: Given a nested object, get the "leaf" values in Depth-First Order
        :return: A list of leaf values.
        """
        if check_types:
            self.check_type(data_object)
        return get_leaf_values(data_object, is_container_func=is_container_func)

    def expand_from_leaves(self, leaves, check_types = True, assert_fully_used=True, is_container_func = _is_primitive_container):
        """
        Given an iterator of leaf values, fill the meta-object represented by this type.

        :param leaves: An iteratable over leaf values
        :param check_types: Assert that the data types match those of the original object
        :param assert_fully_used: Assert that all the leaf values are used
        :return: A nested object, filled with the leaf data, whose structure is represented in this NestedType instance.
        """
        return _fill_meta_object(self.meta_object, (x for x in leaves), check_types=check_types, assert_fully_used=assert_fully_used, is_container_func=is_container_func)

    @staticmethod
    def from_data(data_object, is_container_func = _is_primitive_container):
        """
        :param data_object: A nested data object
        :param is_container_func: A callback which returns True if an object is to be considered a container and False otherwise
        :return: A NestedType object
        """
        return NestedType(get_meta_object(data_object, is_container_func=is_container_func))


def isnestedinstance(data, meta_obj):
    """
    Check if the data is
    :param data:
    :param meta_obj:
    :return:
    """
    raise NotImplementedError()


def get_leaf_values(data_object, is_container_func = _is_primitive_container):
    """
    Collect leaf values of a nested data_obj in Depth-First order.

    e.g.

        >>> get_leaf_values([6]+[{'x': 3, 'y': [i, 'aaa']} for i in xrange(4)])
        [6, 3, 0, 'aaa', 3, 1, 'aaa', 3, 2, 'aaa', 3, 3, 'aaa']

    Caution: If your data contains dicts, you may not get the same order of results when you call this function with
    different dict objects containing the same data.  Python only guarantees that a given dict will always iterate in
    the same order so long as it is not modified.  See https://docs.python.org/2/library/stdtypes.html#dict.items

    :param data_object: An arbitrary nested data object
    :param is_container_func: A callback which returns True if an object is to be considered a container and False otherwise
    :return: A list of leaf values.
    """
    leaf_values = []
    if is_container_func(data_object):
        if isinstance(data_object, (list, tuple)):
            leaf_values += [val for x in data_object for val in get_leaf_values(x, is_container_func=is_container_func)]
        elif isinstance(data_object, OrderedDict):
            leaf_values += [val for k, x in data_object.items() for val in get_leaf_values(x, is_container_func=is_container_func)]
        elif isinstance(data_object, dict):
            leaf_values += [val for k in sorted(data_object.keys(), key = str) for val in get_leaf_values(data_object[k], is_container_func=is_container_func)]
        else:
            raise Exception('Have no way to consistently extract leaf values from a {}'.format(data_object))
        return leaf_values
    else:
        return [data_object]


def _fill_meta_object(meta_object, data_iteratable, assert_fully_used = True, check_types = True, is_container_func = _is_primitive_container):
    """
    Fill the data from the iterable into the meta_object.
    :param meta_object: A nested type descripter.  See NestedType init
    :param data_iteratable: The iterable data object
    :param assert_fully_used: Assert that we actually get through all the items in the iterable
    :param is_container_func: A callback which returns True if an object is to be considered a container and False otherwise
    :return: The filled object
    """

    try:
        if is_container_func(meta_object):
            if isinstance(meta_object, (list, tuple, set)):
                filled_object = type(meta_object)(_fill_meta_object(x, data_iteratable, assert_fully_used=False, check_types=check_types, is_container_func=is_container_func) for x in meta_object)
            elif isinstance(meta_object, OrderedDict):
                filled_object = type(meta_object)((k, _fill_meta_object(val, data_iteratable, assert_fully_used=False, check_types=check_types, is_container_func=is_container_func)) for k, val in meta_object.items())
            elif isinstance(meta_object, dict):
                filled_object = type(meta_object)((k, _fill_meta_object(meta_object[k], data_iteratable, assert_fully_used=False, check_types=check_types, is_container_func=is_container_func)) for k in sorted(meta_object.keys(), key=str))
            else:
                raise Exception('Cannot handle container type: "{}"'.format(type(meta_object)))
        else:
            next_data = next(data_iteratable)
            if check_types and meta_object is not type(next_data):
                raise TypeError('The type of the data object: {} did not match type from the meta object: {}'.format(type(next_data), meta_object))
            filled_object = next_data
    except StopIteration:
        raise TypeError('The data iterable you were going through ran out before the object {} could be filled.'.format(meta_object))

    if assert_fully_used:
        try:
            next(data_iteratable)
            raise TypeError('It appears that the data object you were using to fill your meta object had more data than could fit.')
        except StopIteration:
            pass
    return filled_object


def nested_map(func, *nested_objs, **kwargs):
    """
    An equivalent of pythons built-in map, but for nested objects.  This function crawls the object and applies func
    to the leaf nodes.

    :param func: A function of the form new_leaf_val = func(old_leaf_val)
    :param nested_obj: A nested object e.g. [1, 2, {'a': 3, 'b': (3, 4)}, 5]
    :param check_types: Assert that the new leaf types match the old leaf types (False by default)
    :param is_container_func: A callback which returns True if an object is to be considered a container and False otherwise
    :return: A nested objectect with the same structure, but func applied to every value.
    """
    is_container_func = kwargs['is_container_func'] if 'is_container_func' in kwargs else _is_primitive_container
    check_types = kwargs['check_types'] if 'check_types' in kwargs else False
    assert len(nested_objs)>0, 'nested_map requires at least 2 args'

    assert callable(func), 'func must be a function with one argument.'
    nested_types = [NestedType.from_data(nested_obj, is_container_func=is_container_func) for nested_obj in nested_objs]
    assert all_equal(nested_types), "The nested objects you provided had different data structures:\n{}".format('\n'.join(str(s) for s in nested_types))
    leaf_values = zip(*[nested_type.get_leaves(nested_obj, is_container_func=is_container_func, check_types=check_types) for nested_type, nested_obj in zip(nested_types, nested_objs)])
    new_leaf_values = [func(*v) for v in leaf_values]
    new_nested_obj = nested_types[0].expand_from_leaves(new_leaf_values, check_types=check_types, is_container_func=is_container_func)
    return new_nested_obj


def get_nested_value(data_object, key_chain):
    if len(key_chain)>0:
        return get_nested_value(data_object[key_chain[0]], key_chain=key_chain[1:])
    else:
        return data_object


class ExpandingDict(dict):

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            self[key] = ExpandingDict()
            return dict.__getitem__(self, key)


class ExpandingOrderedDict(OrderedDict):

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            self[key] = ExpandingDict()
            return OrderedDict.__getitem__(self, key)


def expand_struct(struct):

    expanded_struct = ExpandingDict()

    for k in struct.keys():
        exec('expanded_struct%s = struct["%s"]' % (k, k))

    return expanded_struct


def seqstruct_to_structseq(seqstruct, as_arrays=False):
    """
    Turn a sequence of identically-structured nested objects into a nested object of sequences.
    :param seqstruct: A sequence (list or tuple) of nested objects with similar format
    :param as_arrays: Turn the output sequences into numpy arrays
    :return: A nested object with sequences

    For example, if you go:
        signal_seqs = seqstruct_to_structseq(seq_signals)
    Then
        frame_number = 5
        seq_signals[frame_number]['inputs']['camera'] == signal_seqs['inputs']['camera'][frame_number]
    """
    if len(seqstruct)==0:
        return []

    nested_type = NestedType.from_data(seqstruct[0])
    leaf_data = [nested_type.get_leaves(s) for s in seqstruct]
    batch_leaf_data = [np.array(d) for d in zip(*leaf_data)] if as_arrays else zip(*leaf_data)
    structseq = nested_type.expand_from_leaves(leaves = batch_leaf_data, check_types=False)
    return structseq


def structseq_to_seqstruct(structseq):
    """
    Turn a nested object of sequences into a sequence of identically-structured nested objects.

    This is the inverse of seqstruct_to_structseq

    :param structseq: A nested object with sequences
    :return: A sequence (list or tuple) of nested objects with similar format
    """
    nested_type = NestedType.from_data(structseq)
    leaf_data = nested_type.get_leaves(structseq, check_types=False)
    sequence = zip(*leaf_data)
    seqstruct = [nested_type.expand_from_leaves(s, check_types=False) for s in sequence]
    return seqstruct


