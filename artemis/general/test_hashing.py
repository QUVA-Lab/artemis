from artemis.general.hashing import compute_fixed_hash
import numpy as np
import sys

_IS_PYTHON_3 = sys.version_info > (3, 0)


def test_compute_fixed_hash():
    # Not really sure why the fixed hash differes between python 2 and 4 here (maybe something do do with changes to strings)

    complex_obj = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, list(range(10)))]
    original_code = compute_fixed_hash(complex_obj)

    expected_code = 'c9b83dd2e1099c3bcbb05e3c69327c72' if _IS_PYTHON_3 else '285f0ff77c0cb6a5fc529ae55775f9dd'

    assert compute_fixed_hash(complex_obj) == original_code == expected_code
    complex_obj[2]['b'][6]=0

    expected_code = 'a783b1f098fca8ccaf977a3123be5ac4' if _IS_PYTHON_3 else '3124854bda236cd50f0cc04bd6f84935'
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


if __name__ == '__main__':
    test_compute_fixed_hash()
    test_compute_fixed_hash_terminates()
