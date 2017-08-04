import sys
from contextlib import contextmanager

__author__ = 'peter'

"""
A global shared "test_mode".  Allows all functions to modify their behaviour
in "test_mode".  Test mode should be set whenever the variable is set or when
things are beign run from pytest.

We do this to replace the previous solution of passing around a "test_mode" argumet
everywhere.

This is used in conjunction with conftest.py, which adds the _called_from_test flag.
"""

_TEST_MODE = False


def is_test_mode():
    return _TEST_MODE or (hasattr(sys, '_called_from_test') and sys._called_from_test)


def set_test_mode(state):
    global _TEST_MODE
    _TEST_MODE = state


@contextmanager
def hold_test_mode(test_mode = True):
    """
    Execute a block of code in test mode.  That is, any call to "is_test_mode" within that
    block will return True.  Usage:

    with hold_test_mode():
        if is_test_mode():
            dataset = get_my_big_dataset(n_samples = 10)  # Shorten the dataset just to run test
            n_epochs = 2
        else:
            dataset = get_my_big_dataset()
            n_epochs = 20
        ...
    """

    old_test_mode = _TEST_MODE
    set_test_mode(test_mode)
    yield
    set_test_mode(old_test_mode)


class UseTestContext(object):
    """
    DEPRECATED.... use hold_test_mode
    """

    def __init__(self, new_test_mode=True):
        self.new_test_mode = new_test_mode

    def __enter__(self):
        self._old_test_mode = _TEST_MODE
        set_test_mode(self.new_test_mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_test_mode(self._old_test_mode)
