import inspect
from collections import OrderedDict
from functools import partial
import collections
from artemis.general.should_be_builtins import separate_common_items


def get_partial_chain(f):
    """
    Given a partial function f, the chain of functions from which it is derived.
    eg:
        def f(a, b):
            return a+b
        g = partial(f, b=3)
        h = partial(g, a=2)
        assert [f, g, h] == get_partial_chain(h)

    WARNING: Python 3 automatically collapses partials - so in Python 3 this chain will never be longer than 2!

    :param f: A function, possibly a partial
    :return: A list of functions, starting with the root
    """
    return get_partial_chain(f.func) + [f] if isinstance(f, partial) else [f]


def get_partial_root(f):
    return get_partial_chain(f)[0]


def infer_function_and_derived_arg_values(f):
    """
    Given a function f, which may be a partial version of some other function, going down to some root, standard python
    function, get the full set of arguments that this final function will end up being called with.  This function will
    raise an AssertionError if not arguments are defined by the partial chain.

    :param f: A function, or partial function
    :return: root_func, kwargs     ... where:
        root_func is the root function.
        kwargs is An OrderedDict(arg_name->arg_value)
    """
    partial_chain = get_partial_chain(f)
    root, partials = partial_chain[0], partial_chain[1:]
    assert all(len(pf.args)==0 for pf in partials), "We don't handle unnamed arguments for now.  Add this functionality if necessary"
    overrides = dict((argname, argval) for pf in partials for argname, argval in pf.keywords.items())  # Note that later updates on the same args go into the dict
    full_arg_list = infer_arg_values(root, **overrides)
    return root, full_arg_list


def infer_derived_arg_values(f):
    """
    Given a function f, which may be a partial version of some other function, going down to some root, standard python
    function, get the full set of arguments that this final function will end up being called with.  This function will
    raise an AssertionError if not arguments are defined by the partial chain.
    e.g.

        def f(a, b=1):
            return a+b
        g = partial(f, a=2)
        h = partial(g, b=3)
        assert f(**get_derived_function_args(g)) == g()
        assert f(**get_derived_function_args(h)) == h()

    :param f: A function, or partial function
    :return: An OrderedDict(arg_name->arg_value)
    """
    _, full_arg_list = infer_function_and_derived_arg_values(f)
    return full_arg_list


def infer_arg_values(f, *args, **kwargs):
    """
    :param f: Get the full list of arguments to a function, or throw an error if the function cannot be called by the
        given arguments.
    :param args: A list of args
    :param kwargs: A dict of keyword args
    :return: An OrderedDict(arg_name->arg_value)
    """
    all_arg_names, varargs_name, kwargs_name, defaults = inspect.getargspec(f)
    assert varargs_name is None, "This function doesn't work with unnamed args"
    default_args = {k: v for k, v in zip(all_arg_names[len(all_arg_names)-(len(defaults) if defaults is not None else 0):], defaults if defaults is not None else [])}
    args_with_values = set(all_arg_names[:len(args)]+list(default_args.keys())+list(kwargs.keys()))
    assert set(all_arg_names).issubset(args_with_values), "Arguments {} require values but are not given any.  ".format(tuple(set(all_arg_names).difference(args_with_values)))
    assert len(args) <= len(all_arg_names), "You provided {} arguments, but the function only takes {}".format(len(args), len(all_arg_names))
    full_args = tuple(
        list(zip(all_arg_names, args))  # Handle unnamed args f(1, 2)
        + [(name, kwargs[name] if name in kwargs else default_args[name]) for name in all_arg_names[len(args):]]  # Handle named keyworkd args f(a=1, b=2)
        + [(name, kwargs[name]) for name in kwargs if name not in all_arg_names[len(args):]]  # Need to handle case if f takes **kwargs
        )
    duplicates = tuple(item for item, count in collections.Counter([a for a, _ in full_args]).items() if count > 1)
    assert len(duplicates)==0, 'Arguments {} have been defined multiple times: {}'.format(duplicates, full_args)

    common_args, (different_args, different_given_args) = separate_common_items([tuple(all_arg_names), tuple(n for n, _ in full_args)])
    if kwargs_name is None:  # There is no **kwargs
        assert len(different_given_args)==0, "Function {} was given args {} but didn't ask for them".format(f, different_given_args)
    assert len(different_args)==0, "Function {} needs values for args {} but didn't get them".format(f, different_args)
    return OrderedDict(full_args)
