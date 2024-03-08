import sys
import inspect
_CACHE = {}


def default(function, arg):
    """
    Get the default value to the argument named 'arg' for function "function"
    :param Callable function: The function from which to get the default value.
    :param str arg: The name of the argument
    :return Any: The default value
    """
    if function not in _CACHE:
        if sys.version_info < (3, 4):
            all_arg_names, varargs_name, kwargs_name, defaults = inspect.getargspec(function)
        else:
            all_arg_names, varargs_name, kwargs_name, defaults, _, _, _ = inspect.getfullargspec(function)
        _CACHE[function] = dict(zip(all_arg_names[-len(defaults):], defaults))
    assert arg in _CACHE[function], 'Function {} has no default argument "{}"'.format(function, arg)
    return _CACHE[function][arg]
