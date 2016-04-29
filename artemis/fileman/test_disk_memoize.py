import time
from fileman.disk_memoize import memoize_to_disk, clear_memo_files_for_function, DisableMemos, memoize_to_disk_and_cache
import numpy as np

__author__ = 'peter'

@memoize_to_disk
def compute_slow_thing(a, b, c):
    call_time = time.time()
    time.sleep(0.01)
    return (a+b)/float(c), call_time


def test_memoize_to_disk():

    clear_memo_files_for_function(compute_slow_thing)

    t = time.time()
    num, t1 = compute_slow_thing(1, 3, 3)
    assert t-t1 < 0.01
    assert num == (1+3)/3.

    num, t2 = compute_slow_thing(1, 3, 3)
    assert num == (1+3)/3.
    assert t2==t1

    num, t3 = compute_slow_thing(1, 3, 4)
    assert num == (1+3)/4.
    assert t3 != t1

    with DisableMemos():
        num, t4 = compute_slow_thing(1, 3, 4)
        assert num == (1+3)/4.
        assert t4 != t3

    num, t5 = compute_slow_thing(1, 3, 4)
    assert num == (1+3)/4.
    assert t5 == t3


@memoize_to_disk
def complex_arg_fcn(a, b):
    time.sleep(0.01)
    return str(a)+str(b), time.time()


def test_complex_args():
    """
    Make sure that we detect changes in nested arguments.
    :return:
    """
    arr = np.random.rand(3, 4)
    clear_memo_files_for_function(complex_arg_fcn)
    t = time.time()
    out1, t1 = complex_arg_fcn([1, 2], b={'a': 3, 'b': arr})
    assert t-t1 < 0.02
    out2, t2 = complex_arg_fcn([1, 2], b={'a': 3, 'b': arr})
    assert out1==out2
    assert t1==t2
    arr[2, 3] = -1
    out3, t3 = complex_arg_fcn([1, 2], b={'a': 3, 'b': arr})
    assert out1!=out3
    assert t1!=t3


@memoize_to_disk_and_cache
def compute_slow_thing_again(a, b, c):
    call_time = time.time()
    time.sleep(0.01)
    return (a+b)/float(c), call_time


def test_memoize_to_disk_and_cache():

    clear_memo_files_for_function(compute_slow_thing_again)

    t = time.time()
    num1, t1 = compute_slow_thing_again(1, 3, 3)
    assert t-t1 < 0.01
    assert num1 == (1+3)/3.

    num2, t2 = compute_slow_thing_again(1, 3, 3)
    assert num2 == (1+3)/3.
    assert num2 is num1
    assert t2==t1

    num, t3 = compute_slow_thing_again(1, 3, 4)
    assert num == (1+3)/4.
    assert t3 != t1

    with DisableMemos():
        num, t4 = compute_slow_thing_again(1, 3, 4)
        assert num == (1+3)/4.
        assert t4 != t3

    num, t5 = compute_slow_thing_again(1, 3, 4)
    assert num == (1+3)/4.
    assert t5 == t3


if __name__ == '__main__':
    test_memoize_to_disk_and_cache()
    test_memoize_to_disk()
    test_complex_args()
