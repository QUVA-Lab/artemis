from artemis.general.nested_structures import flatten_struct
import numpy as np


def test_nested_structures():

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


if __name__ == '__main__':
    test_nested_structures()
