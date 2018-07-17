from collections import namedtuple

from artemis.general.should_be_builtins import izip_equal


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

        def create_scannable(**kwargs):
            return Scannable(func=func, state=state, returns=returns, output=output, kwargs=kwargs)
        func.scan = create_scannable




        class StatelessUpdater(namedtuple('StatelessUpdater for {}'.format(func.__name__))):

            def __call__(self, *args, **kwargs):




                return StatelessUpdater(*(return_value for return_name, return_value in izip_equal(returns, return_values) if return_name in state))


        state_object = None if isinstance(state, str) else namedtuple('State of {}'.format(func.__name__), state)
        output_object = None if isinstance(output, str) else namedtuple('Output of {}'.format(func.__name__, output))

        def standard_form(input_values, state_values):

            kwargs = input_value


            return output_values, state_values


        func.standard_form = standard_form

        return func

    return wrapper


class Scannable(object):

    SINGLE_OUTPUT_FORMAT = object()
    TUPLE_OUTPUT_FORMAT = object()

    def __init__(self, func, state, returns, output, kwargs = None):
        """
        See scannable docstring
        """
        if isinstance(state, str):
            state = (state, )
        assert isinstance(state, (list, tuple)), 'State should be a list of state names.  Got a {}'.format(state.__class__)
        state_names = state
        state = {}
        if kwargs is not None:
            state.update(kwargs)

        if returns is None:
            assert len(state_names)==1, "If there is more than one state variable, you must specify the output!"
            returns = next(iter(state_names))
        if isinstance(returns, str):
            assert returns in state_names, 'Output name "{}" was not provided not included in the state dict: "{}"'.format(returns, state_names)
            self._output_format = Scannable.SINGLE_OUTPUT_FORMAT
            self._state_names = returns
            output_names = [returns]
            self._output_format = Scannable.SINGLE_OUTPUT_FORMAT
        else:
            assert isinstance(returns, (list, tuple)), "output must be a string, a list/tuple, or None"
            assert all(sn in returns for sn in state_names), 'Variabels name(s) {} were listed as state variables but not included in the list of outputs: {}'.format([sn for sn in state_names if sn not in returns], returns)
            output_names = returns
            self._output_format = Scannable.TUPLE_OUTPUT_FORMAT
            self._state_names = tuple(state_names)
            self._state_indices_in_output = [output_names.index(state_name) for state_name in state_names]
        if isinstance(output, str):
            assert returns is not None, 'If you specify returns, you must specify output'
            if isinstance(returns, str):
                assert output == output_names
                return_index = None
            else:
                assert isinstance(returns, (list, tuple))
                return_index = returns.index(output)
        elif isinstance(output, (list, tuple)):
            return_index = tuple(output_names.index(r) for r in output)
        else:
            assert output is None
            return_index = None

        self.func = func
        self._state = state
        self._return_index = return_index
        self._output_names = output_names

    def __str__(self):
        output = self._output_names[0] if self._output_format is Scannable.SINGLE_OUTPUT_FORMAT else self._output_names
        returns = None if self._return_index is None else repr(self._output_names[self._return_index]) if isinstance(self._return_index, int) else tuple(self._output_names[i] for i in self._return_index)
        self._strrep = '{}(func={}, state={}, output={}, returns={})'.format(self.__class__.__name__, self.func.__name__, self._state, output, returns)
        return self._strrep

    def __call__(self, *args, **kwargs):
        kwargs.update(self._state)
        values_returned = self.func(*args, **kwargs)
        if self._output_format is Scannable.SINGLE_OUTPUT_FORMAT:
            self._state[self._state_names] = values_returned
        else:
            try:
                assert len(values_returned) == len(self._output_names), 'The number of outputs: {}, does not match the length of the specified outputs: {} ({})'.format(len(values_returned), len(self._output_names), self._output_names)
            except TypeError:
                raise TypeError('{} should have returned an iterable of length {} containing variables {}, but got a non-iterable: {}'.format(self.func.__name__, len(self._output_names), self._output_names, values_returned))
            self._state.update((state_name, values_returned[ix]) for state_name, ix in zip(self._state_names, self._state_indices_in_output))
        return values_returned if self._return_index is None else values_returned[self._return_index] if isinstance(self._return_index, int) else tuple(values_returned[i] for i in self._return_index)

    @property
    def state(self):
        return self._state.copy()

#
# class ScannableStateLess(object):
#
#     def __init__(self, func, inputs, returns, outputs):
#         pass
#
#     def __call__(self, *args, **kwargs):
#         """
#         :param args:
#         :param kwargs:
#         :return:
#         """


