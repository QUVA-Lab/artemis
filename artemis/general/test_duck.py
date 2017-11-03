from collections import OrderedDict

import pytest
from pytest import raises

from artemis.general.duck import Duck, InvalidKeyError, InitializerTooShallowError
from artemis.general.hashing import compute_fixed_hash
import numpy as np

from artemis.general.should_be_builtins import izip_equal


def _get_standard_test_duck():
    a = Duck()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7
    return a


def test_so_demo():

    # Demo 1: Dynamic assignment
    a = Duck()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7

    assert list(a['b', 1, :]) == [6, 7]
    assert a['b', :, 'subfield1'] == [4, 6]
    assert a['a', 'aa2'] == 2
    assert np.array_equal(a['b'].to_array(), [[4, 5], [6, 7]])
    with pytest.raises(KeyError):  # This should raise an error because key 'a' does not have subkeys 1, 'subfield1'
        x = a[:, 1, 'subfield1']

    # Demo 2: Sequential and Sliced Assignment
    # Here we show another way to create the same structure as above
    # 1) You can assign to a slice
    # 2) You can use the "next" builtin like an index to append to the structure.
    b = Duck()
    b['a', :] = {'aa1': 1, 'aa2': 2}  # Note: when assigning with dict, keys are sorted before insertion (OrderedDict order is kept though).
    b['b', next, :] = {'subfield1': 4, 'subfield2': 5}
    b['b', next, :] = {'subfield1': 6, 'subfield2': 7}
    assert b==a


def test_dict_assignment():

    a = Duck()  # When assigning with a dict we first sort keys.  Here we just verify that keys remain sorted
    a[next, :] = {letter: number for letter, number in izip_equal('abcdefghijklmnopqrstuvwxyz', range(1, 27))}
    a[next, :] = {letter: number for letter, number in izip_equal('abcdefghijklmnopqrstuvwxyz', range(27, 27+26))}
    assert list(a[0].keys()) == [char for char in 'abcdefghijklmnopqrstuvwxyz']
    assert list(a[0].values()) == list(range(1, 27))
    assert list(a[1].keys()) == [char for char in 'abcdefghijklmnopqrstuvwxyz']
    assert list(a[1].values()) == list(range(27, 27+26))


def test_reasonable_errors_on_wrong_keys():

    a = Duck()
    a['a']=1
    with pytest.raises(InvalidKeyError):
        a[1] = 'a'
    with pytest.raises(InvalidKeyError):
        a[next]='a'

    a = Duck()
    a[next] = 'a'
    with pytest.raises(InvalidKeyError):
        a['a'] = 1
    with pytest.raises(InvalidKeyError):
        a[10] = 'a'  # Too high!


def test_string_slices():

    a = Duck()
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'[::-1]):  # Note the reverse
        a[char]=26-i
    assert list(a['q':'m'].items()) == [('q', 17), ('p', 16), ('o', 15), ('n', 14)]
    assert list(a['m':'q'].items()) == []
    assert list(a['m':'q':-1].items()) == [('m', 13), ('n', 14), ('o', 15), ('p', 16)]


def test_dictarraylist():

    # We build the same thing in four different ways

    a = Duck()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7
    a['b', 2, 'subfield2'] = 9  # Note: Out of order assignment here!
    a['b', 2, 'subfield1'] = 8

    b = Duck.from_struct(a.to_struct())
    assert a.to_struct() == b.to_struct()

    c = Duck()
    c['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    c['b', next, :] = OrderedDict([('subfield1', 4), ('subfield2', 5)])
    c['b', next, :] = OrderedDict([('subfield1', 6), ('subfield2', 7)])
    c['b', next, :] = OrderedDict([('subfield2', 9), ('subfield1', 8)])

    d = Duck()
    d['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    d['b', :, :] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    e = Duck()
    e['a', ...] = OrderedDict([('aa1', 1), ('aa2', 2)])
    e['b', ...] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    f = a[:]

    g = a[:, :]

    for i, same_struct in enumerate([a, b, c, d, e, f, g]):

        print ('Test {}'.format(i))
        assert a.to_struct()==same_struct.to_struct()
        assert a==same_struct
        assert compute_fixed_hash(a.to_struct()) == compute_fixed_hash(b.to_struct())

        assert same_struct['a', 'aa2'] == 2
        assert list(same_struct['b', 1, :].values()) == [6, 7]
        assert list(same_struct['b', :, 'subfield1'].values()) == [4, 6, 8]
        assert same_struct['b', :, :].deepvalues() == [[4, 5], [6, 7], [9, 8]]  # Note that the order-switching gets through.
        assert same_struct['b', :, ['subfield1', 'subfield2']].deepvalues() == [[4, 5], [6, 7], [8, 9]]
        assert same_struct['b', :, :].to_struct() == [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]
        assert same_struct['b', 1:, :].to_struct() == [OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

        with pytest.raises(KeyError):
            x = same_struct['a', 'aa3']

        with pytest.raises(KeyError):
            x = same_struct['a', 1, 'subfield1']

        with pytest.raises(IndexError):
            x = same_struct['b', 4, 'subfield1']

        with pytest.raises(KeyError):  # This should raise an error because key 'a' does not have subkeys 1, 'subfield1'
            x = same_struct[:, 1, 'subfield1']

        assert np.array_equal(same_struct['b'].to_array(), [[4, 5], [6, 7], [8, 9]])  # Note... the order is corrected.
        with pytest.raises(InvalidKeyError):
            assert np.array_equal(same_struct.to_array())  # Note... the order is corrected.

        new_struct = same_struct.arrayify_axis(axis=1, subkeys=('b',))
        assert np.array_equal(new_struct['b', 'subfield1'], [4, 6, 8])
        assert np.array_equal(new_struct['b', 'subfield2'], [5, 7, 9])


def test_simple_growing():
    a = Duck()
    for i in range(10):
        a[next] = i*2
    assert np.array_equal(a.to_array(), np.arange(0, 20, 2))


def test_open_key():

    a = Duck()
    a['a', :] = [3, 4]
    with pytest.raises(KeyError):
        a['b']
    c = a.open('b')
    c[next] = 5
    c[next] = 6
    assert a['b', 0]==5
    assert a['b', 1]==6


def test_open_next():

    a = Duck()
    a['a'] = Duck()
    aa = a['a'].open(next)
    aa['c'] = 4
    aa['4'] = 5
    aa = a['a'].open(next)
    aa['c'] = 6
    aa['4'] = 7


def test_to_struct():

    a = _get_standard_test_duck()

    b = a.to_struct()
    assert b==OrderedDict([
        ('a', OrderedDict([('aa1', 1), ('aa2', 2)])),
        ('b', [
            OrderedDict([('subfield1', 4), ('subfield2', 5)]),
            OrderedDict([('subfield1', 6), ('subfield2', 7)])
            ])])


def test_next_elipsis_assignment():

    a = Duck()
    a['a', next, ...] = OrderedDict([('b', [1, 2])])
    a['a', next, ...] = OrderedDict([('b', [3, 4])])
    # Should lead to form:
    # 'a'
    #   0
    #     'b'
    #       0: 1
    #       1: 2
    #   1
    #     'b'
    #       0: 3
    #       1: 4

    # If we arrayify axis, 1, we expect struct:

    # Should lead to form:
    # 'a'
    #   'b'
    #     0: [1, 3]
    #     1: [2, 4]
    assert np.array_equal(a.arrayify_axis(axis=1, subkeys='a')['a', 'b'], [[1, 3], [2, 4]])

    a = Duck()
    a['a', next, ...] = OrderedDict([('b', np.array([1, 2]))])
    a['a', next, ...] = OrderedDict([('b', np.array([3, 4]))])

    # Now, a is no longer broken into ...  So we have to be careful with these elipses!
    # 'a'
    #   0
    #     'b': [1, 2]
    #   1
    #     'b': [3, 4]

    # If we arrayify axis, 1, we expect struct:

    # Should lead to form:
    # 'a'
    #   'b'
    #     0: [1, 2]
    #     1: [3, 4]
    assert np.array_equal(a.arrayify_axis(axis=1, subkeys='a')['a', 'b'], [[1, 2], [3, 4]])


def test_slice_assignment():

    deep_struct = {'a': [1,{'c': 2, 'd': 3}], 'b': [1, 2]}

    a = Duck()
    a[:] = deep_struct
    assert list(a.items(depth='full')) == [
        (('a', ), [1,{'c': 2, 'd': 3}]),
        (('b', ), [1, 2])
        ]
    a = Duck()
    a[:, :] = deep_struct
    assert list(a.items(depth='full')) == [
        (('a', 0), 1),
        (('a', 1), {'c': 2, 'd': 3}),
        (('b', 0), 1),
        (('b', 1), 2)
        ]
    a = Duck()
    with raises(InitializerTooShallowError):
        a[:, :, :] = deep_struct

    a = Duck()
    a[...] = deep_struct
    assert list(a.items(depth='full')) == [
        (('a', 0), 1),
        (('a', 1, 'c'), 2),
        (('a', 1, 'd'), 3),
        (('b', 0), 1),
        (('b', 1), 2)
        ]

    a = Duck()
    with raises(InvalidKeyError):  # It would be nice to add this functionality, but maybe later.
        a[..., next] = deep_struct
        # This commant would mean "turn every leaf node in deep_struct into a list".  It has the nice property
        # of automatically asserting that deep_struct maintains the same structure between iterations.


def test_arrayify_empty_stuct():

    a = Duck.from_struct(OrderedDict([('a', [])]))
    b = a.arrayify_axis(axis=1, subkeys='a')
    assert isinstance(b['a'], np.ndarray)
    assert np.array_equal(b['a'], [])

    a = Duck.from_struct(OrderedDict([('a', [1, 2])]))
    b = a.arrayify_axis(axis=1, subkeys='a')
    assert isinstance(b['a'], np.ndarray)
    assert np.array_equal(b['a'], [1, 2])


def test_slice_on_start():

    a = Duck()
    a['testing'] = Duck()
    b = a.arrayify_axis(axis=1, subkeys='testing')
    assert np.array_equal(b['testing'], [])


def test_assign_tuple_keys():

    a = Duck()
    a['training'] = Duck()
    a['testing', next]=4
    b = a.arrayify_axis(axis=1, subkeys='testing')
    b = b.arrayify_axis(axis=1, subkeys='training', inplace=True).to_struct()
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 1), (('a', 'c'), 2)]))])
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 3), (('a', 'c'), 4)]))])
    a.arrayify_axis(axis=1, subkeys='training').to_struct()
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 1), (('a', 'c'), 2)]))])
    pass


def test_broadcast_bug():

    a = _get_standard_test_duck()

    assert isinstance(a['b', :], Duck)
    assert np.array_equal(a['b'].to_array(), [[4, 5], [6, 7]])
    assert a['b']==a['b', :, :]
    assert np.array_equal(a['b', :, :].to_array(), [[4, 5], [6, 7]])  # This failed, but now is fixed.


def test_key_values():

    a = _get_standard_test_duck()

    assert list(a.keys()) == ['a', 'b']
    assert list(a.values()) == [a['a'], a['b']]
    assert list(a.keys(depth=1)) == [('a', ), ('b', )]
    assert list(a.values(depth=1)) == [a['a'], a['b']]
    assert list(a.keys(depth=2)) == [('a', 'aa1'), ('a', 'aa2'), ('b', 0), ('b', 1)]
    assert list(a.values(depth=2)) == [1, 2, a['b', 0], a['b', 1]]
    assert list(a.keys(depth='full')) == [('a', 'aa1'), ('a', 'aa2'), ('b', 0, 'subfield1'), ('b', 0, 'subfield2'), ('b', 1, 'subfield1'), ('b', 1, 'subfield2')]
    assert list(a.values(depth='full')) == [1, 2, 4, 5, 6, 7]
    assert list(a.items(depth='full')) == list(zip(a.keys(depth='full'), a.values(depth='full')))


_expected_description = """<Duck with 2 keys: ['a', 'b']>
| a: 
| | aa1: 1
| | aa2: 2
| b: 
| | 0: 
| | | subfield1: 4
| | | subfield2: 5
| | 1: 
| | | subfield1: 6
| | | subfield2: 7"""


_extended_expected_description = """<Duck with 2 keys: ['a', 'b']>
| a: 
| | aa1: 1
| | aa2: 2
| b: 
| | 0: 
| | | subfield1: 4
| | | subfield2: 5
| | 1: 
| | | subfield1: 6
| | | subfield2: 7
| | 2: 
| | | subfield1: 5
| | 3: 
| | | subfield1: 5
| | 4: 
| | | subfield1: 5
| | (... Omitting 2 of 6 elements ...)"""



def test_description():
    a = _get_standard_test_duck()
    assert a.description()==_expected_description

    a['b', 2, 'subfield1']=5
    a['b', 3, 'subfield1']=5
    a['b', 4, 'subfield1']=5
    a['b', 5, 'subfield1']=5

    assert a.description()==_extended_expected_description


def test_duck_array_build():

    a = Duck()
    count = 0
    for i in range(3):
        for j in range(4):
            a[i, j] = count
            count+=1
    desired_array = np.arange(12).reshape(3, 4)
    assert a==desired_array
    assert np.array_equal(a.to_array(), desired_array)


def test_split_get_assign():

    a = Duck()
    c=0
    for i in ['a', 'b']:
        for j in [0, 1]:
            for k in ['c', 'd']:
                a[i, j, k] = c
                c+=1
    b = a[:, 1, :]
    print (a[:, 1, :].description())
    assert list(b.items(depth='full')) == [(('a', 'c'), 2), (('a', 'd'), 3), (('b', 'c'), 6), (('b', 'd'), 7)]

    with pytest.raises(InvalidKeyError):
        a[:, 1, :] = b


def test_assign_from_struct():

    a = Duck()
    a[:] = [2, 3, 4, 5]
    assert list(a.items()) == [(0, 2), (1, 3), (2, 4), (3, 5)]
    a[:] = dict(a=1, b=2, c=3)
    assert list(a.items()) == [('a', 1), ('b', 2), ('c', 3)]


def test_arrayify_axis_demo():

    a = Duck()
    a[0, 'x'] = 1
    a[0, 'y'] = 2
    a[1, 'x'] = 3
    a[1, 'y'] = 4
    b = a.arrayify_axis(axis=0)
    assert np.array_equal(b['x'], [1, 3])
    assert np.array_equal(b['y'], [2, 4])


if __name__ == '__main__':
    test_so_demo()
    test_dict_assignment()
    test_dictarraylist()
    test_simple_growing()
    test_open_key()
    test_open_next()
    test_to_struct()
    test_next_elipsis_assignment()
    test_slice_assignment()
    test_arrayify_empty_stuct()
    test_slice_on_start()
    test_assign_tuple_keys()
    test_broadcast_bug()
    test_key_values()
    test_description()
    test_duck_array_build()
    test_split_get_assign()
    test_assign_from_struct()
    test_arrayify_axis_demo()
    test_string_slices()
    test_reasonable_errors_on_wrong_keys()
