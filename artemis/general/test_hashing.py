from artemis.general.hashing import compute_fixed_hash, fixed_hash_eq
import numpy as np
import sys

_IS_PYTHON_3 = sys.version_info > (3, 0)


def test_compute_fixed_hash():
    # Not really sure why the fixed hash differes between python 2 and 4 here (maybe something do do with changes to strings)

    complex_obj = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, list(range(10)))]
    original_code = compute_fixed_hash(complex_obj)

    expected_code = 'c9b83dd2e1099c3bcbb05e3c69327c72' if _IS_PYTHON_3 else '6c98fabc301361863f321f6149a8a12a'

    assert compute_fixed_hash(complex_obj) == original_code == expected_code
    complex_obj[2]['b'][6]=0

    expected_code = 'a783b1f098fca8ccaf977a3123be5ac4' if _IS_PYTHON_3 else '8aee1f739fc9a612ed72e14682026627'
    assert compute_fixed_hash(complex_obj) == expected_code != original_code
    complex_obj[2]['b'][6]=6  # Revert to old value
    assert compute_fixed_hash(complex_obj) == original_code


def test_compute_fixed_hash_terminates():

    a = []
    b = [a]
    a.append(b)
    code = compute_fixed_hash(a)
    assert code == 'cffaee424a62cd1893825a5811c34b8d'

    c = []
    d = [c]
    c.append(d)
    code = compute_fixed_hash(c)
    assert code == 'cffaee424a62cd1893825a5811c34b8d'


def test_fixed_hash_eq():

    obj1 = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, [1, 2, 3, 4, 5])]
    obj2 = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, [1, 2, 3, 4, 5])]
    obj3 = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, [1, 2, 3, 4, 5])]
    obj3[2]['b'][4] = 0
    assert fixed_hash_eq(obj1, obj2)
    assert not fixed_hash_eq(obj1, obj3)


if __name__ == '__main__':
    test_compute_fixed_hash()
    test_compute_fixed_hash_terminates()
    test_fixed_hash_eq()