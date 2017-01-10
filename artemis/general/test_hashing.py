from artemis.general.hashing import compute_fixed_hash
import numpy as np


def test_compute_fixed_hash():
    complex_obj = [1, 'd', {'a': 4, 'b': np.arange(10)}, (7, range(10))]
    assert compute_fixed_hash(complex_obj) == '285f0ff77c0cb6a5fc529ae55775f9dd'
    complex_obj[2]['b'][6]=0
    assert compute_fixed_hash(complex_obj) == '3124854bda236cd50f0cc04bd6f84935'
    complex_obj[2]['b'][6]=6  # Revert to old value
    assert compute_fixed_hash(complex_obj) == '285f0ff77c0cb6a5fc529ae55775f9dd'


if __name__ == '__main__':
    test_compute_fixed_hash()
