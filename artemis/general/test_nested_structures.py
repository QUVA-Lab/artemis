from artemis.general.nested_structures import flatten_struct, get_meta_object, NestedType
import numpy as np
from pytest import raises


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


if __name__ == '__main__':
    test_flatten_struct()
    test_get_meta_object()
    test_nested_type()
