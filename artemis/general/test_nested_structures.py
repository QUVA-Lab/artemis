import numpy as np
import pytest
from pytest import raises
from six import string_types
from six.moves import xrange

from artemis.general.nested_structures import (flatten_struct, get_meta_object, NestedType,
                                               seqstruct_to_structseq, structseq_to_seqstruct, nested_map,
                                               get_leaf_values)


def test_flatten_struct():

    a = {
        'a': [1, 2, 3],
        'b': {
            'c': 4,
            'd': {'e': 5, 6: [7, 8]},
            },
        'c': 1
        }
    b = dict(flatten_struct(a))
    assert b["['a'][1]"]==2
    assert b["['b']['d'][6][1]"]==8

    class A(object):

        def __init__(self, other_object, b):
            self._a = other_object
            self._b = b

    thing = A(A(3, 'arr'), {'x': np.random.randn(3, 2), 'y': (3, 4, 5)})

    flat_thing = dict(flatten_struct(thing))

    assert flat_thing['._a._b']=='arr'
    assert flat_thing["._b['y'][0]"]==3


_complex_obj = [1, 2, {'a':(3, 4), 'b':['hey', 'yeah']}, 'woo']
_meta_obj = [int, int, {'a':(int, int), 'b':[str, str]}, str]


def test_get_meta_object():
    assert get_meta_object(_complex_obj) == [int, int, {'a':(int, int), 'b':[str, str]}, str]


def test_nested_type():
    # Example from docstring
    assert NestedType.from_data([1, 2, {'a': (3, 4.), 'b': 'c'}]) == NestedType([int, int, {'a': (int, float), 'b': str}])

    # Other tests
    nested_type = NestedType.from_data(_complex_obj)
    assert nested_type.is_type_for(_complex_obj)
    _complex_obj_2 = [1, 2, {'a': (3, 4, 5), 'b': ['hey', 'yeah']}, 'woo']
    _complex_obj_3 = [1, 2, {'a': (3, 5), 'b': ['hey', 'yeah']}, 'woo']
    assert not nested_type.is_type_for(_complex_obj_2)
    assert nested_type.is_type_for(_complex_obj_3)
    assert nested_type == NestedType(_meta_obj)
    leaves = [1, 2, 3, 4, 'hey', 'yeah', 'woo']
    assert nested_type.get_leaves(_complex_obj) == leaves
    assert nested_type.expand_from_leaves(leaves) == _complex_obj

    extended_leaves = leaves + ['x']
    with raises(TypeError):
        nested_type.expand_from_leaves(extended_leaves)

    # Change the type
    swapped_leaves = list(leaves)
    swapped_leaves[0] = 'eh?'
    with raises(TypeError):
        nested_type.expand_from_leaves(swapped_leaves)


def test_seqstruct_to_structseq_and_inverse():

    a = [{'x': np.random.randn(2), 'y': [i, 'aaa']} for i in xrange(4)]
    b = seqstruct_to_structseq(a, as_arrays=True)

    assert b['x'].shape==(4, 2)
    assert np.array_equal(b['x'][2], a[2]['x'])
    assert np.array_equal(b['y'][0], np.arange(4))
    assert np.array_equal(b['y'][1], ['aaa']*4)

    c = structseq_to_seqstruct(b)
    assert np.array_equal(a[2]['x'], c[2]['x'])
    assert np.array_equal(a[3]['x'], c[3]['x'])
    assert np.array_equal(a[3]['y'][0], c[3]['y'][0])
    assert np.array_equal(a[3]['y'][1], c[3]['y'][1])


def test_nested_map():
    func = lambda x: x*2 if isinstance(x, (int, float)) else x+'  Not!' if isinstance(x, string_types) else x
    assert nested_map(func, 2)==4
    assert nested_map(func, 'God is dead.')=='God is dead.  Not!'
    assert nested_map(func, (1, 2, 3)) == (2, 4, 6)
    assert nested_map(func, [1, 2, None, {'a': 3, 'b': 'It works!'}]) == [2, 4, None, {'a': 6, 'b': 'It works!  Not!'}]
    with pytest.raises(AssertionError):
        assert nested_map(lambda a, b: a+b, {'a': 1, 'b': [2, 3]}, {'a': 4, 'XXX': [5, 6]}) == {'a': 5, 'b': [7, 9]}
    with pytest.raises(AssertionError):
        assert nested_map(lambda a, b: a+b, {'a': 1, 'b': [2, 3]}, {'a': 4, 'b': [5, 6], 'c': [7]}) == {'a': 5, 'b': [7, 9]}


def test_get_leaf_values():
    assert get_leaf_values([6]+[{'x': 3, 'y': [i, 'aaa']} for i in xrange(4)]) == [6, 3, 0, 'aaa', 3, 1, 'aaa', 3, 2, 'aaa', 3, 3, 'aaa']


def test_nested_map_with_container_func():

    data = {'a': [1, 2, 3], 'b': [3, 4, 5]}
    result = nested_map(lambda x: np.array(x), data, is_container_func=lambda x: isinstance(x, dict))
    assert isinstance(result['a'], np.ndarray)
    assert np.array_equal(result['a'], [1, 2, 3])
    assert isinstance(result['b'], np.ndarray)
    assert np.array_equal(result['b'], [3, 4, 5])


def test_none_bug():

    a = {'a': 1, 'b': None, 'c': [1, 2, None]}

    fa = flatten_struct(a, first_dict_is_namespace=True)
    assert dict(fa) == {'a': 1, 'b': None, 'c[0]': 1, 'c[1]': 2, 'c[2]': None}


if __name__ == '__main__':
    test_flatten_struct()
    test_get_meta_object()
    test_nested_type()
    test_seqstruct_to_structseq_and_inverse()
    test_nested_map()
    test_get_leaf_values()
    test_nested_map_with_container_func()
    test_none_bug()
