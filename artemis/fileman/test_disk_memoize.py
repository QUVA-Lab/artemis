import time
from artemis.fileman.disk_memoize import memoize_to_disk, clear_memo_files_for_function, DisableMemos, memoize_to_disk_and_cache, \
    memoize_to_disk_test, memoize_to_disk_and_cache_test
from artemis.general.test_mode import set_test_mode
import numpy as np
from pytest import raises

__author__ = 'peter'


@memoize_to_disk_test
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


@memoize_to_disk_and_cache_test
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


@memoize_to_disk_and_cache_test
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


def test_clear_error_for_missing_arg():

    clear_memo_files_for_function(compute_slow_thing)

    with raises(AssertionError):
        compute_slow_thing(1)


def test_clear_arror_for_wrong_arg():

    clear_memo_files_for_function(compute_slow_thing)

    with raises(AssertionError):
        compute_slow_thing(a=1, b=2, c=3, d=4)


@memoize_to_disk_test
def compute_slow_thing_with_kwargs(a, **kwargs):
    call_time = time.time()
    time.sleep(0.01)
    return (a+kwargs['b'])/float(kwargs['c']), call_time


def test_unnoticed_wrong_arg_bug_is_dead():

    clear_memo_files_for_function(compute_slow_thing)
    compute_slow_thing(a=1, b=2, c=3)  # Creates a memo
    with raises(AssertionError):
        compute_slow_thing(a=1, b=2, see=3)  # Previously, this was not caught, leading you not to notice your typo


def test_catch_kwarg_error():

    clear_memo_files_for_function(compute_slow_thing_with_kwargs)

    t = time.time()
    num, t1 = compute_slow_thing_with_kwargs(1, b=2, c=4)
    assert num == (1+2)/4.
    assert t1 > t

    num, t2 = compute_slow_thing_with_kwargs(1, b=2, c=6)
    assert num == (1+2)/6.
    assert t2>t1

    num, t3 = compute_slow_thing_with_kwargs(1, b=2, c=4)
    assert num == (1+2)/4.
    assert t3 == t1


if __name__ == '__main__':
    set_test_mode(True)
    test_unnoticed_wrong_arg_bug_is_dead()
    test_catch_kwarg_error()
    test_clear_arror_for_wrong_arg()
    test_clear_error_for_missing_arg()
    test_memoize_to_disk_and_cache()
    test_memoize_to_disk()
    test_complex_args()
