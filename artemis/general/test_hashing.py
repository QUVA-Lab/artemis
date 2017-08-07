from artemis.general.hashing import compute_fixed_hash
import numpy as np


def test_compute_fixed_hash():
    complex_obj = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, range(10))]
    original_code = compute_fixed_hash(complex_obj)
    assert compute_fixed_hash(complex_obj) == original_code == '6c98fabc301361863f321f6149a8a12a'
    complex_obj[2]['b'][6]=0
    assert compute_fixed_hash(complex_obj) == '8aee1f739fc9a612ed72e14682026627' != original_code
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
