import inspect
from abc import abstractmethod
from collections import OrderedDict
from functools import partial
import collections
from artemis.general.should_be_builtins import separate_common_items
import sys
import types


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
    return get_partial_chain(f.func) + [f] if isinstance(f, (partial, PartialReparametrization)) else [f]


def get_partial_root(f):
    return get_partial_chain(f)[0]


class PartialReparametrization(object):
    """
    A reparametrization of a function.
    """

    def __init__(self, func, arg_constructors):
        arg_to_sub_arg = {}
        all_arg_names, _, _, _ = advanced_getargspec(func)
        for arg_name, arg_constructor in arg_constructors.items():
            assert arg_name in all_arg_names, "Function {} has no argument named '{}'".format(func, arg_name)
            assert callable(arg_constructor), "The configuration for argument '{}' must be a function which constructs the argument.  Got a {}".format(arg_name, type(arg_constructor).__name__)
            # assert isinstance(arg_constructor, types.FunctionType) or inspect.isclass(arg_constructor), "The constructor '{}' appeared not to be a pure function.  It is probably an instance of a callable class, and you probably meant to give a either a constructor for that instance, or a class object.".format(arg_constructor)
            assert not inspect.isclass(arg_constructor),  "'{}' is a class object.  You must instead pass a function to construct an instance of this class.  You can use lambda for this.".format(arg_constructor.__name__)
            assert isinstance(arg_constructor, types.FunctionType),  "The constructor '{}' appeared not to be a pure function.  If it is an instance of a callable class, you probably meant to give a either a constructor for that instance.".format(arg_constructor)
            sub_arg_names, _, _, _ = advanced_getargspec(arg_constructor)
            for a in sub_arg_names:
                if a != arg_name:  # If the name of your reparemetrizing argument is not the same as the argument you are replacing....
                    assert a not in all_arg_names, "You use argument '{}' to construct argument '{}', but the function {} already has an argument named '{}'.  You need to come up with a new name.".format(a, arg_name, func, a)
            arg_to_sub_arg[arg_name] = sub_arg_names
        self._arg_to_sub_arg = arg_to_sub_arg
        self._arg_constructors = arg_constructors
        self._func = func

    @property
    def func(self):
        return self._func

    @property
    def keyword_constructors(self):
        return self._arg_constructors

    def _put_constructed_args_into_kwargs(self, kwargs):
        # This function constructs all arguments using the constructors defined in add_config_variant using the metaargument
        # It then removes the metaarguments that were used to construct those arguments

        # Construct argument specified in config
        constructed_args = {}
        for arg_name, arg_constructor in self._arg_constructors.items():
            input_args = {k: kwargs[k] for k, v in kwargs.items() if k in self._arg_to_sub_arg[arg_name]}
            try:
                constructed_args[arg_name] = arg_constructor(**input_args)
            except TypeError as err:
                print('Error while trying to construct argument: "{}": {}'.format(arg_name, str(err)))
                raise
        # Remove config args from args that are passed down.
        for k in set(argname for args in self._arg_to_sub_arg.values() for argname in args if argname in kwargs):
            del kwargs[k]

        kwargs.update(constructed_args)

    def __str__(self):
        return '<PartialReparametrization of {} with args {} redefined>'.format(self._func, list(self._arg_to_sub_arg.keys()))

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class PartialReparametrizationFunction(PartialReparametrization):

    def __call__(self, **kwargs):
        self._put_constructed_args_into_kwargs(kwargs)
        return self._func(**kwargs)


class PartialGeneratorReparametrization(PartialReparametrization):

    def __call__(self, **kwargs):
        self._put_constructed_args_into_kwargs(kwargs)
        for result in self._func(**kwargs):
            yield result


def partial_reparametrization(func, **arg_constructors):
    """
    Reparameterize the function func, by adding constructors for the arguments to func, which themselves have arguemnts.

    This can be useful when arguments are objects and you wish to define a function

    e.g.

        def add(a, b):
            return a+b

        f = partial_reparametrization(add, b=lambda c, d: c*d)

        assert f(a=1, c=2, d=3) == add(a=1, b=2*3) == 1 + 2*3 == 7

    :param func:
    :param arg_constructors:
    :return:
    """
    if inspect.isgeneratorfunction(get_partial_root(func)):
        return PartialGeneratorReparametrization(func, arg_constructors=arg_constructors)
    else:
        return PartialReparametrizationFunction(func, arg_constructors=arg_constructors)
    #
    # get_arg_names = lambda f: inspect.getargspec(f) if sys.version_info < (3, 0) else inspect.getfullargspec(f)[0]
    # arg_to_sub_arg = {}
    # all_arg_names = get_arg_names(func)
    # for arg_name, arg_constructor in arg_constructors.items():
    #     # assert arg_name in all_arg_names, "Function {} has no argument named '{}'".format(self.function, arg_name)
    #     assert callable(arg_constructor), "The configuration for argument '{}' must be a function which constructs the argument.  Got a {}".format(arg_name, type(arg_constructor).__name__)
    #     # assert isinstance(arg_constructor, types.FunctionType) or inspect.isclass(arg_constructor), "The constructor '{}' appeared not to be a pure function.  It is probably an instance of a callable class, and you probably meant to give a either a constructor for that instance, or a class object.".format(arg_constructor)
    #     assert not inspect.isclass(arg_constructor),  "'{}' is a class object.  You must instead pass a function to construct an instance of this class.  You can use lambda for this.".format(arg_constructor.__name__)
    #     assert isinstance(arg_constructor, types.FunctionType),  "The constructor '{}' appeared not to be a pure function.  If it is an instance of a callable class, you probably meant to give a either a constructor for that instance.".format(arg_constructor)
    #     sub_arg_names = get_arg_names(arg_constructor)
    #     for a in sub_arg_names:
    #         assert a not in all_arg_names, "An argument with name '{}' already exists.  You need to come up with a new name.".format(a)
    #     arg_to_sub_arg[arg_name] = sub_arg_names
    #
    # def put_constructed_args_into_kwargs(kwargs):
    #     # This function constructs all arguments using the constructors defined in add_config_variant using the metaargument
    #     # It then removes the metaarguments that were used to construct those arguments
    #
    #     # Construct argument specified in config
    #     constructed_args = {}
    #     for arg_name, arg_constructor in arg_constructors.items():
    #         input_args = {k: kwargs[k] for k, v in kwargs.items() if k in arg_to_sub_arg[arg_name]}
    #         try:
    #             constructed_args[arg_name] = arg_constructor(**input_args)
    #         except TypeError as err:
    #             print('Error while trying to construct argument: "{}": {}'.format(arg_name, str(err)))
    #             raise
    #     # Remove config args from args that are passed down.
    #     for k in set(argname for args in arg_to_sub_arg.values() for argname in args if argname in kwargs):
    #         del kwargs[k]
    #
    #     kwargs.update(constructed_args)
    #
    # if inspect.isgeneratorfunction(get_partial_root(func)):
    #     def configured_function(**kwargs):
    #         put_constructed_args_into_kwargs(kwargs)
    #         for result in func(**kwargs):
    #             yield result
    # else:
    #     def configured_function(**kwargs):
    #         put_constructed_args_into_kwargs(kwargs)
    #         return func(**kwargs)
    #

    # return configured_function

#
# if inspect.isgeneratorfunction(get_partial_root(self.function)):
#     def configured_function(**kwargs):
#         put_constructed_args_into_kwargs(kwargs)
#         for result in self.function(**kwargs):
#             yield result
# else:
#     def configured_function(**kwargs):
#         put_constructed_args_into_kwargs(kwargs)
#         return self.function(**kwargs)


#
# def infer_function_and_derived_arg_values(f):
#     """
#     Given a function f, which may be a partial version of some other function, going down to some root, standard python
#     function, get the full set of arguments that this final function will end up being called with.  This function will
#     raise an AssertionError if not all arguments are defined by the partial chain.
#
#     :param f: A function, or partial function
#     :return: root_func, kwargs     ... where:
#         root_func is the root function.
#         kwargs is An OrderedDict(arg_name->arg_value)
#     """
#     partial_chain = get_partial_chain(f)
#     all_arg_names, varargs_name, kwargs_name, defaults = advanced_getargspec(f)
#     assert all(a in defaults)
#     return partial_chain[0],
#
#     root, partials = partial_chain[0], partial_chain[1:]
#     assert all(len(pf.args)==0 for pf in partials), "We don't handle unnamed arguments for now.  Add this functionality if necessary"
#     overrides = dict((argname, argval) for pf in partials for argname, argval in pf.keywords.items())  # Note that later updates on the same args go into the dict
#     full_arg_list = infer_arg_values(root, kwargs=overrides)
#     return root, full_arg_list
#
#
# def infer_derived_arg_values(f):
#     """
#     Given a function f, which may be a partial version of some other function, going down to some root, standard python
#     function, get the full set of arguments that this final function will end up being called with.  This function will
#     raise an AssertionError if not all arguments are defined by the partial chain.
#     e.g.
#
#         def f(a, b=1):
#             return a+b
#         g = partial(f, a=2)
#         h = partial(g, b=3)
#         assert f(**get_derived_function_args(g)) == g()
#         assert f(**get_derived_function_args(h)) == h()
#
#     :param f: A function, or partial function
#     :return: An OrderedDict(arg_name->arg_value)
#     """
#     _, full_arg_list = infer_function_and_derived_arg_values(f)
#     return full_arg_list


def advanced_getargspec(f):
    """
    A more intelligent version of getargspec which is able to handle partial functions and PartialReparametrizations.
    :param f: A function, partial function, or PartialReparametrization
    :return: (all_arg_names, varargs_name, kwargs_name, defaults)
        all_arg_names a list of all arguments that can be fed to the function
        varargs_name: the name of the *arg object, if any, else None
        kwargs_name: the name of the **kwarg object, if any, else None
        defaults: A OrderedDict of default argument values.
    """
    chain = get_partial_chain(f)
    if sys.version_info < (3, 4):
        all_arg_names, varargs_name, kwargs_name, defaults = inspect.getargspec(chain[0])
    else:
        all_arg_names, varargs_name, kwargs_name, defaults, _, _, _ = inspect.getfullargspec(chain[0])
    if defaults is None:
        defaults = OrderedDict()
    else:
        defaults = OrderedDict((name, val) for name, val in zip(all_arg_names[-len(defaults):], defaults))
    assert varargs_name is None, "This function doesn't work with unnamed args.  Add this functionality if necessary"

    for f in chain[1:]:
        if isinstance(f, partial):
            assert len(f.args)==0, "We don't handle unnamed arguments for now.  Add this functionality if necessary"
            for k, v in f.keywords.items():
                if kwargs_name is None:
                    assert k in all_arg_names, "Partial Argument '{}' appears not to exist in function {}".format(k, chain[0])
                defaults[k] = v
        elif isinstance(f, PartialReparametrization):
            current_layer_arg_names = list(all_arg_names)
            for k, constructor in f.keyword_constructors.items():
                if kwargs_name is None:
                    assert k in all_arg_names, "Constructed Argument '{}' appears not to exist in function {}".format(k, chain[0])
                sub_all_arg_names, sub_varargs_name, sub_kwargs_name, sub_defaults = advanced_getargspec(constructor)
                assert sub_varargs_name is None, "Currently can't handle unnamed arguments for argument constructor {}={}".format(k, constructor)
                assert sub_kwargs_name is None, "Currently can't handle unnamed keyword arguments for argument constructor {}={}".format(k, constructor)
                all_arg_names.remove(k)  # Since the argument has been reparameterized, it is removed from the  list of constructor signature
                current_layer_arg_names.remove(k)
                assert not any(a in current_layer_arg_names for a in sub_all_arg_names), "The constructor for argument '{}' has name '{}', which us already used by the function '{}'.  Rename it.".format(k, next(a for a in sub_all_arg_names if a in current_layer_arg_names), chain[0])
                all_arg_names.extend(sub_all_arg_names)  # Add the reparametrizing args to the constructor signature
                defaults.update(sub_defaults)
                if k in defaults:
                    del defaults[k]
        else:
            raise Exception('Unexpected element of partial chain: {}'.format(f))

    return all_arg_names, varargs_name, kwargs_name, defaults


def get_defined_and_undefined_args(func):
    """
    :param func: A function (including partial or PartialReparametrization)
    :return: defined_args, undefined_arg_names
        defined_args: A dict[arg_name, arg_val]
        undefined_arg_names: A Sequence[arg_name] of all arg names which are still undefined
    """
    undefined_arg_names, varargs_name, kwargs_name, defined_args = advanced_getargspec(func)
    assert varargs_name is None
    assert kwargs_name is None
    for k in defined_args.keys():
        undefined_arg_names.remove(k)
    return defined_args, undefined_arg_names



def infer_arg_values(f, args=(), kwargs={}):
    """
    DEPRECATED!  Use advanced_getargspec instead.
    
    Get the full list of arguments to a function f called with args, kwargs, or throw an error if the function cannot be
    called by the given arguments (e.g. if the provided args, kwargs do not provide all required arguments to the function).

    :param f: A function
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
