from functools import partial

import sys
from pytest import raises
from artemis.general.functional import infer_arg_values, get_partial_chain, infer_derived_arg_values


def test_get_full_args():

    def func(a, b):
        return a+b

    assert list(infer_arg_values(func, kwargs=dict(a=1, b=3)).items()) == [('a', 1), ('b', 3)]
    assert list(infer_arg_values(func, (1, ), dict(b=3)).items()) == [('a', 1), ('b', 3)]
    assert list(infer_arg_values(func, (1, 3)).items()) == [('a', 1), ('b', 3)]
    with raises(AssertionError):  # AssertionError: Arguments ('a', 'b') require values but are not given any.
        infer_arg_values(func, )
    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_arg_values(func, kwargs=dict(b=3))
    with raises(AssertionError):  # AssertionError: The set of argument names to the function: ('a', 'b') must match the set of arguments given: ('a', 'b', 'c')
        infer_arg_values(func, kwargs=dict(a=1, b=3, c=5))
    with raises(AssertionError):  # AssertionError: Arguments ('b',) require values but are not given any.
        infer_arg_values(func, (1, ), kwargs=dict(c=5))
    with raises(AssertionError):  # AssertionError: You provided 3 arguments, but the function only takes 2
        print(infer_arg_values(func, args=(1, 2, 5)))
    with raises(AssertionError):  # AssertionError: Arguments ('b',) have been defined multiple times: (('a', 1), ('b', 2), ('b', 5))
        infer_arg_values(func, args=(1, 2), kwargs=dict(b=5))

    def func_with_defaults(a, b=4):
        return a+b
    assert list(infer_arg_values(func_with_defaults, kwargs=dict(a=1)).items()) == [('a', 1), ('b', 4)]
    assert list(infer_arg_values(func_with_defaults, kwargs=dict(a=1, b=3)).items()) == [('a', 1), ('b', 3)]
    assert list(infer_arg_values(func_with_defaults, args=(1, )).items()) == [('a', 1), ('b', 4)]
    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_arg_values(func_with_defaults, )
    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_arg_values(func_with_defaults, kwargs=dict(b=3))
    with raises(AssertionError):  # AssertionError: The set of argument names to the function: ('a', 'b') must match the set of arguments given: ('a', 'b', 'c')
        infer_arg_values(func_with_defaults, kwargs=dict(a=1, b=3, c=5))
    with raises(AssertionError):  # AssertionError: The set of argument names to the function: ('a', 'b') must match the set of arguments given: ('a', 'b', 'c')
        infer_arg_values(func_with_defaults, args=(1, ), kwargs=dict(c=5))
    with raises(AssertionError):  # AssertionError: You provided 3 arguments, but the function only takes 2
        print(infer_arg_values(func_with_defaults, args=(1, 2, 5)))
    with raises(AssertionError):  # AssertionError: Arguments ('b',) have been defined multiple times: (('a', 1), ('b', 2), ('b', 5))
        infer_arg_values(func_with_defaults, args=(1, 2), kwargs=dict(b=5))

    def func_with_kwargs(a, b=4, **kwargs):
        return a+b
    assert list(infer_arg_values(func_with_kwargs, kwargs=dict(a=1)).items()) == [('a', 1), ('b', 4)]
    assert list(infer_arg_values(func_with_kwargs, kwargs=dict(a=1, b=3)).items()) == [('a', 1), ('b', 3)]
    assert list(infer_arg_values(func_with_kwargs, args=(1, )).items()) == [('a', 1), ('b', 4)]
    assert list(infer_arg_values(func_with_kwargs, kwargs=dict(a=1, b=3, c=5)).items()) == [('a', 1), ('b', 3), ('c', 5)]
    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_arg_values(func_with_kwargs, )
    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_arg_values(func_with_kwargs, kwargs=dict(b=3))
    with raises(AssertionError):  # AssertionError: You provided 3 arguments, but the function only takes 2
        print(list(infer_arg_values(func_with_kwargs, args=(1, 2, 5))))
    with raises(AssertionError):  # AssertionError: Arguments ('b',) have been defined multiple times: (('a', 1), ('b', 2), ('b', 5))
        infer_arg_values(func_with_kwargs, args=(1, 2), kwargs=dict(b=5))


def test_get_partial_chain():
    def f(a, b):
        return a+b
    g = partial(f, b=3)
    h = partial(g, a=2)

    if sys.version_info < (3, 0):  # Python 2.X
        assert [f, g, h] == get_partial_chain(h)
    else: # Python 3.X
        base, part = get_partial_chain(h)
        assert base is f
        assert part.keywords == {'a': 2, 'b': 3}


def test_get_derived_function_args():
    def f(a, b=1):
        return a+b
    g = partial(f, a=2)
    h = partial(g, b=3)
    j = partial(h, c=4)

    with raises(AssertionError):  # AssertionError: Arguments ('a',) require values but are not given any.
        infer_derived_arg_values(f)
    assert list(infer_derived_arg_values(g).items()) == [('a', 2), ('b', 1)]
    assert list(infer_derived_arg_values(h).items()) == [('a', 2), ('b', 3)]
    with raises(AssertionError):
        infer_derived_arg_values(j)

    assert f(**infer_derived_arg_values(g)) == g()
    assert f(**infer_derived_arg_values(h)) == h()


if __name__ == '__main__':
    test_get_full_args()
    test_get_partial_chain()
    test_get_derived_function_args()
