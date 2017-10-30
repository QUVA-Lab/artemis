from collections import OrderedDict

import pytest

from artemis.general.arraystruct import ArrayStruct
from artemis.general.hashing import compute_fixed_hash
import numpy as np

def test_array_struct():

    # We build the same thing in four different ways

    a = ArrayStruct()
    a['a', 'aa1'] = 1
    a['a', 'aa2'] = 2
    a['b', 0, 'subfield1'] = 4
    a['b', 0, 'subfield2'] = 5
    a['b', 1, 'subfield1'] = 6
    a['b', 1, 'subfield2'] = 7
    a['b', 2, 'subfield2'] = 9  # Note: Out of order assignment here!
    a['b', 2, 'subfield1'] = 8

    b = ArrayStruct(a.to_struct(), recurse=True)
    assert a.to_struct() == b.to_struct()

    c = ArrayStruct()
    c['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    c['b', next, :] = OrderedDict([('subfield1', 4), ('subfield2', 5)])
    c['b', next, :] = OrderedDict([('subfield1', 6), ('subfield2', 7)])
    c['b', next, :] = OrderedDict([('subfield2', 9), ('subfield1', 8)])

    d = ArrayStruct()
    d['a', :] = OrderedDict([('aa1', 1), ('aa2', 2)])
    d['b', :, :] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    e = ArrayStruct()
    e['a', ...] = OrderedDict([('aa1', 1), ('aa2', 2)])
    e['b', ...] = [OrderedDict([('subfield1', 4), ('subfield2', 5)]), OrderedDict([('subfield1', 6), ('subfield2', 7)]), OrderedDict([('subfield2', 9), ('subfield1', 8)])]

    for i, same_struct in enumerate([a, b, c, d, e]):

        assert a.to_struct()==b.to_struct()
        assert compute_fixed_hash(a.to_struct()) == compute_fixed_hash(b.to_struct())

        print 'Test {}'.format(i)
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
        with pytest.raises(ValueError):
            assert np.array_equal(same_struct.to_array())  # Note... the order is corrected.

        new_struct = same_struct.arrayify_axis(axis=1, ixs=('b', ))
        assert np.array_equal(new_struct['b', 'subfield1'], [4, 6, 8])
        assert np.array_equal(new_struct['b', 'subfield2'], [5, 7, 9])


def test_simple_growing():
    a = ArrayStruct()
    for i in range(10):
        a[next] = i*2
    assert np.array_equal(a.to_array(), np.arange(0, 20, 2))


if __name__ == '__main__':
    test_array_struct()
    test_simple_growing()