from artemis.general.deferred_defaults import default
from pytest import raises


def test_deferred_defaults():

    def subfunction_1(a=2, b=3):
        return a+b

    def subfunction_2(c=4):
        return c**2

    def main_function(a=default(subfunction_1, 'a'), b=default(subfunction_1, 'b'), c=default(subfunction_2, 'c')):
        return subfunction_1(a=a, b=b) * subfunction_2(c=c)

    assert main_function()==(2+3)*4**2
    assert main_function(b=5)==(2+5)*4**2
    assert main_function(b=5, c=1)==(2+5)*1**2


def check_that_errors_caught():

    def g(a=4):
        return a*2

    with raises(AssertionError):
        def f(a = default(g, 'b')):
            return a


if __name__ == '__main__':
    test_deferred_defaults()
    check_that_errors_caught()
