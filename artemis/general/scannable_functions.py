from collections import namedtuple
from artemis.general.functional import advanced_getargspec


def _normalize_formats(func, returns, state, output, outer_kwargs):
    """

    :param func:
    :param returns:
    :param state:
    :param output:
    :return:
    """

    if returns is None:
        assert isinstance(state, str), "If there is more than one state variable, you must specify the return variables!"
        returns = state

    single_return_format = isinstance(returns, str)

    if isinstance(state, str):
        state = (state, )
    assert isinstance(state, (list, tuple)), 'State should be a list of state names.  Got a {}'.format(state.__class__)

    arg_names, _, _, initial_state_dict = advanced_getargspec(func)

    for s in state:
        assert s in arg_names, "The state '{}' is not a parameter to the function '{}'".format(s, func.__name__)
        assert s==returns if single_return_format else s in returns, "The state variable '{}' is not updated by the returns '{}'".format(s, returns)

    if output is None:
        output = returns

    if isinstance(state, dict):
        initial_state_dict = state
    else:
        for name in state:
            assert name in initial_state_dict, "Argument '{}' is part of your state, but it does not have an initial value.  Provide one either by passing in state as a dict or adding a default in the function signature".format(name)

    parameter_kwargs = {}
    for k, v in outer_kwargs.items():
        if k in state:
            initial_state_dict[k] = v
        else:
            parameter_kwargs[k] = v

    if isinstance(state, (list, tuple, dict)):
        return_state_indices = None if single_return_format else [returns.index(s) for s in state]
    else:
        raise Exception('State must be a list, tuple, dict, string.  Not {}'.format(output))

    if single_return_format:
        return_state_names = returns
    else:
        return_state_names = [returns[ix] for ix in return_state_indices]

    if isinstance(output, (list, tuple)):
        output_type = namedtuple('ScanOutputs_of_{}'.format(func.__name__), output)
        return_output_indices = [returns.index(o) for o in output]
    elif isinstance(output, str):
        output_type = None
        return_output_indices = returns.index(output)
    else:
        raise Exception('Output must be a list/tuple or string.  Not {}'.format(output))

    if single_return_format:
        return_output_indices = None

    return single_return_format, return_output_indices, return_state_indices, return_state_names, initial_state_dict, output_type, parameter_kwargs


def immutable_scan(func, returns, state, output, outer_kwargs = {}):
    """
    Create a StatelessUpdater object from a function.

    A StatelessUpdater is an Immutable callable object, which stores state.  When called, it returns a new Statelessupdater,
    containing the new state, and the specified return value.

        e.g.

            def moving_average(avg, x, t=0):
                return avg*t/(t+1)+x/(t+1), t+1

            sup = immutable_scan(moving_average, state=['avg', 't'], returns = ['avg', 't'], output='avg')

            sup2, avg = sup(3)
            assert avg==3
            sup3, avg = sup2(4)
            assert avg == 3.5

        Note, you may choose to use the @scannable decorator instead:

            @scannable(state=['avg', 't'], returns = ['avg', 't'], output='avg')
            def moving_average(avg, x, t=0):
                return avg*t/(t+1)+x/(t+1), t+1

            sup = moving_average.immutable_scan()

    :param Callable func: A function which defines a step in an iterative process
    :param Union[Sequence[str], str] state: A list of names of state variables.
    :param Union[Sequence[str], str] returns: A list of names of variables returned from the function.
    :param Union[Sequence[str], str] output: A list of names of "output" variables
    :return Callable[[...], Tuple[Callable, Any]]: An immutable callable of the form:
        new_object_state, outputs = old_object_state(**inputs)
    """

    single_return_format, return_output_indices, return_state_indices, return_state_names, initial_state_dict, output_type, parameter_kwargs = _normalize_formats(func=func, returns=returns,  state=state, output=output, outer_kwargs=outer_kwargs)
    single_output_format = not isinstance(return_output_indices, (list, tuple))

    class ImmutableScan(namedtuple('ImmutableScan_of_{}'.format(func.__name__), state)):

        def __call__(self, *args, **kwargs):
            """
            :param args:
            :param kwargs:
            :return StatelessUpdater, Any:
                Where the second output is an arbitrary value if output is specified as a string, or a namedtuple if outputs is specified as a list/tuple
            """
            arguments = self._asdict()
            arguments.update(**parameter_kwargs)
            arguments.update(**kwargs)
            return_values = func(*args, **arguments)

            if single_return_format:
                if single_output_format:
                    output_values = return_values
                else:
                    output_values = output_type(return_values)
                new_state = ImmutableScan(return_values)
            else:
                try:
                    assert len(return_values) == len(returns), 'The number of return values: {}, does not match the length of the specified return variables: {} ({})'.format(len(return_values), len(returns), returns)
                except TypeError:
                    raise TypeError('{} should have returned an iterable of length {} containing variables {}, but got a non-iterable: {}'.format(func.__name__, len(returns), returns, return_values))
                if single_output_format:
                    output_values = return_values[return_output_indices]
                else:
                    output_values = output_type(*(return_values[i] for i in return_output_indices))
                new_state = ImmutableScan(*(return_values[ix] for ix in return_state_indices))

            return new_state, output_values

    return ImmutableScan(**initial_state_dict)


def mutable_scan(func, state, returns, output, outer_kwargs = {}):

    single_return_format, return_output_indices, return_state_indices, return_state_names, initial_state_dict, output_type, parameter_kwargs = _normalize_formats(func=func, returns=returns,  state=state, output=output, outer_kwargs=outer_kwargs)
    single_output_format = not isinstance(return_output_indices, (list, tuple))

    try:
        from recordclass import recordclass
    except:
        raise ImportError('Stateful Updaters require recordclass to be installed.  Run "pip install recordclass".')

    class MutableScan(recordclass('MutableScan_of_{}'.format(func.__name__), state)):

        def __call__(self, *args, **kwargs):
            """
            :param args:
            :param kwargs:
            :return StatelessUpdater, Any:
                Where the second output is an arbitrary value if output is specified as a string, or a namedtuple if outputs is specified as a list/tuple
            """
            arguments = self._asdict()
            arguments.update(**parameter_kwargs)
            arguments.update(**kwargs)
            return_values = func(*args, **arguments)

            if single_return_format:
                if single_output_format:
                    output_values = return_values
                else:
                    output_values = output_type(return_values)
                setattr(self, return_state_names, return_values)
            else:
                try:
                    assert len(return_values) == len(returns), 'The number of return values: {}, does not match the length of the specified return variables: {} ({})'.format(len(return_values), len(returns), returns)
                except TypeError:
                    raise TypeError('{} should have returned an iterable of length {} containing variables {}, but got a non-iterable: {}'.format(func.__name__, len(returns), returns, return_values))
                if single_output_format:
                    output_values = return_values[return_output_indices]
                else:
                    output_values = output_type(*(return_values[i] for i in return_output_indices))
                for ix, name in zip(return_state_indices, return_state_names):
                    setattr(self, name, return_values[ix])

            return output_values

    return MutableScan(**initial_state_dict)


def scannable(state, returns=None, output=None):
    """
    A decorator for turning functions into stateful objects.  The decorator attaches a "scan" method to the given function,
    which can be called to create an object which stores the state that gets fed back into the next function call.  This
    removes the need to create single-method classes.

    e.g. To compute a simple moving average of a sequence:

        seq = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

        @scannable(state=dict(avg=0, n=0), output=['avg', 'n'], returns='avg')
        def simple_moving_average(x, avg, n):
            return (n/(1.+n))*avg + (1./(1.+n))*x, n+1

        f = simple_moving_average.scan()
        averaged_signal = [f(x=x) for t, x in enumerate(seq)]
        assert np.allclose(f.state['avg'], np.mean(seq))

    :param dict state: A dictionary whose keys are the names of the arguments to feed back, and whose values are the
        default initial state.  These initial values can be overridden when calling function.scan(arg_name, initial_value)
    :param  Optional[Union[str, Sequence[str]]] returns: If there is more than one state variable or more than one output,
        include the list of output names, so that the scan knows which outputs to use to update the state.
    :param Optional[Union[str, Sequence[str]]] output: If there is more than one output and you only wish to return a
        subset of the outputs, indicate here which variables you want to return.
    :return: func, but with a "scan" function attched.
    """
    def wrapper(func):

        def make_mutable_scan(**kwargs):
            return mutable_scan(func=func, state=state, returns=returns, output=output, outer_kwargs=kwargs)
        func.mutable_scan = make_mutable_scan

        def make_immutable_scan(**kwargs):
            return immutable_scan(func=func, state=state, returns=returns, output=output, outer_kwargs=kwargs)

        func.immutable_scan = make_immutable_scan
        return func

    return wrapper
