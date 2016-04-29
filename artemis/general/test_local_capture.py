import sys
from artemis.general.local_capture import execute_and_capture_locals

__author__ = 'peter'


def test_execute_and_capture_locals():

    def func():
        a=3
        return a+4

    def outer_func():
        b=4
        def nested():
            return func()+b+5
        return nested

    out, local_vars = execute_and_capture_locals(func)
    assert out == 3+4
    assert local_vars == {'a': 3}
    assert sys.getprofile() is None

    out, local_vars = execute_and_capture_locals(outer_func())
    assert out == 7+4+5
    assert local_vars == {'b': 4, 'func': func}
    assert sys.getprofile() is None


if __name__ == '__main__':

    test_execute_and_capture_locals()