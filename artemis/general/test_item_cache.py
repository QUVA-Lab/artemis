from pytest import raises
import numpy as np
from artemis.general.item_cache import CacheDict


def test_item_cache():

    cache = CacheDict(buffer_length=3)
    cache[1] = 'aaa'
    cache[4] = 'bbb'
    cache[7] = 'ccc'
    assert 1 in cache
    assert cache[1] == 'aaa'
    assert 0 not in cache
    with raises(KeyError):
        _ = cache[0]
    cache[9] = 'ddd'
    assert 1 not in cache
    with raises(KeyError):
        _ = cache[1]
    assert cache[4] == 'bbb'
    cache[9] = 'ddd'
    assert 4 not in cache

    cache = CacheDict(buffer_size_bytes=2000000)
    img = np.random.rand(240, 300)  # 8*240*300 = 576000 byes... room for 3
    cache[4] = img.copy()
    assert 4 in cache
    cache[5] = img.copy()
    assert 4 in cache
    cache[6] = img.copy()
    assert 4 in cache
    cache[7] = img.copy()
    assert 4 not in cache
    assert 5 in cache
    cache[8] = img.copy()
    assert 4 not in cache
    assert 5 not in cache

    cache = CacheDict(buffer_size_bytes=100)  # No room for even 1
    img = np.random.rand(240, 300)
    cache[1] = img.copy()
    assert 1 not in cache
    cache[2] = img.copy()
    assert 2 not in cache

    cache = CacheDict(buffer_size_bytes=100, always_allow_one_item=True)  # No room for even 1
    img = np.random.rand(240, 300)
    cache[1] = img.copy()
    assert 1 in cache
    cache[2] = img.copy()
    assert 1 not in cache
    assert 2 in cache


if __name__ == '__main__':
    test_item_cache()
