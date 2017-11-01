from collections import OrderedDict

import pytest

from artemis.general.dictarraylist import DictArrayList, InvalidKeyError
from artemis.general.hashing import compute_fixed_hash
import numpy as np


def test_so_demo():

    a = DictArrayList()

    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7

    assert a['a', 'aa2'] == 2
    assert a['b', 1, :].values() == [6, 7]
    assert a['b', :, 'subfield1'] == [4, 6]
    assert np.array_equal(a['b'].to_array(), [[4, 5], [6, 7]])

    with pytest.raises(KeyError):  # This should raise an error because key 'a' does not have subkeys 1, 'subfield1'
        x = a[:, 1, 'subfield1']


def test_dictarraylist():

    # We build the same thing in four different ways

    a = DictArrayList()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7
    a['b', 2, 'subfield2'] = 9  # Note: Out of order assignment here!
    a['b', 2, 'subfield1'] = 8

    b = DictArrayList.from_struct(a.to_struct())
    assert a.to_struct() == b.to_struct()

    c = DictArrayList()
    c['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    c['b', next, :] = OrderedDict([('subfield1', 4), ('subfield2', 5)])
    c['b', next, :] = OrderedDict([('subfield1', 6), ('subfield2', 7)])
    c['b', next, :] = OrderedDict([('subfield2', 9), ('subfield1', 8)])

    d = DictArrayList()
    d['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    d['b', :, :] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    e = DictArrayList()
    e['a', ...] = OrderedDict([('aa1', 1), ('aa2', 2)])
    e['b', ...] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    for i, same_struct in enumerate([a, b, c, d, e]):

        print 'Test {}'.format(i)
        assert a.to_struct()==same_struct.to_struct()
        assert a==same_struct
        assert compute_fixed_hash(a.to_struct()) == compute_fixed_hash(b.to_struct())

        assert same_struct['a', 'aa2'] == 2
        assert same_struct['b', 1, :].values() == [6, 7]
        assert same_struct['b', :, 'subfield1'].values() == [4, 6, 8]
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
    a = DictArrayList()
    for i in range(10):
        a[next] = i*2
    assert np.array_equal(a.to_array(), np.arange(0, 20, 2))


def test_open_key():

    a = DictArrayList()
    a['a', :] = [3, 4]
    with pytest.raises(KeyError):
        a['b']
    c = a.open('b')
    c[next] = 5
    c[next] = 6
    assert a['b', 0]==5
    assert a['b', 1]==6


def test_open_next():

    a = DictArrayList()
    a['a'] = DictArrayList()
    aa = a['a'].open(next)
    aa['c'] = 4
    aa['4'] = 5
    aa = a['a'].open(next)
    aa['c'] = 6
    aa['4'] = 7


def test_to_struct():

    a = DictArrayList()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7

    b = a.to_struct()
    assert b==OrderedDict([
        ('a', OrderedDict([('aa1', 1), ('aa2', 2)])),
        ('b', [
            OrderedDict([('subfield1', 4), ('subfield2', 5)]),
            OrderedDict([('subfield1', 6), ('subfield2', 7)])
            ])])


def test_next_elipsis_assignment():

    a = DictArrayList()
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

    a = DictArrayList()
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


def test_arrayify_empty_stuct():

    a = DictArrayList.from_struct(OrderedDict([('a', [])]))
    b = a.arrayify_axis(axis=1, subkeys='a')
    assert isinstance(b['a'], np.ndarray)
    assert np.array_equal(b['a'], [])

    a = DictArrayList.from_struct(OrderedDict([('a', [1, 2])]))
    b = a.arrayify_axis(axis=1, subkeys='a')
    assert isinstance(b['a'], np.ndarray)
    assert np.array_equal(b['a'], [1, 2])


def test_slice_on_start():

    a = DictArrayList()
    a['testing'] = DictArrayList()
    b = a.arrayify_axis(axis=1, subkeys='testing')
    assert np.array_equal(b['testing'], [])


def test_assign_tuple_keys():

    a = DictArrayList()
    a['training'] = DictArrayList()
    a['testing', next]=4
    b = a.arrayify_axis(axis=1, subkeys='testing')
    b = b.arrayify_axis(axis=1, subkeys='training', inplace=True).to_struct()
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 1), (('a', 'c'), 2)]))])
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 3), (('a', 'c'), 4)]))])
    a.arrayify_axis(axis=1, subkeys='training').to_struct()
    a['training', next, ...] = OrderedDict([('ops', OrderedDict([(('a', 'b'), 1), (('a', 'c'), 2)]))])
    pass


def test_broadcast_bug():

    a = DictArrayList()

    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7

    assert isinstance(a['b', :], DictArrayList)
    assert np.array_equal(a['b'].to_array(), [[4, 5], [6, 7]])
    assert a['b']==a['b', :, :]
    assert np.array_equal(a['b', :, :].to_array(), [[4, 5], [6, 7]])  # This failed, but now is fixed.


if __name__ == '__main__':
    test_so_demo()
    test_dictarraylist()
    test_simple_growing()
    test_open_key()
    test_open_next()
    test_to_struct()
    test_next_elipsis_assignment()
    test_arrayify_empty_stuct()
    test_slice_on_start()
    test_assign_tuple_keys()
    test_broadcast_bug()