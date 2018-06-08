class Scannable(object):

    SINGLE_OUTPUT_FORMAT = object()
    TUPLE_OUTPUT_FORMAT = object()



    def __init__(self, func, state, output, returns):
        assert isinstance(state, dict)

        if output is None:
            assert len(state)==1, "If there is more than one state variable, you must specify the output!"
            output = next(iter(state.keys()))
        if isinstance(output, str):
            assert output in state, 'Output name "{}" was not provided not included in the state dict: "{}"'.format(output, list(self.state.keys()))
            self._output_format = Scannable.SINGLE_OUTPUT_FORMAT
            self._state_names = output
            output_names = [output]
            self._output_format = Scannable.SINGLE_OUTPUT_FORMAT
        else:
            assert isinstance(output, (list, tuple)), "output must be a string, a list/tuple, or None"
            output_names = output
            self._output_format = Scannable.TUPLE_OUTPUT_FORMAT
            self._state_names = tuple(state.keys())
            self._state_indices_in_output = [output_names.index(state_name) for state_name in state.keys()]
        if isinstance(returns, str):
            assert output is not None, 'If you specify returns, you must specify output'
            if isinstance(output, str):
                assert returns==output_names
                return_index = None
            else:
                assert isinstance(output, (list, tuple))
                return_index = output.index(returns)
        elif isinstance(returns, (list, tuple)):
            return_index = tuple(output_names.index(r) for r in returns)
        else:
            assert returns is None
            return_index = None

        self._strrep = '{}(func={}, state={}, output={}, returns={})'.format(self.__class__.__name__, func, state, output, returns)
        self.func = func
        self._state = state
        self._return_index = return_index
        self._output_names = output_names

    def __str__(self):
        return self._strrep

    def __call__(self, *args, **kwargs):
        kwargs.update(self._state)
        values_returned = self.func(*args, **kwargs)
        if self._output_format is Scannable.SINGLE_OUTPUT_FORMAT:
            self._state[self._state_names] = values_returned
        else:
            assert len(values_returned) == len(self._output_names), 'The number of outputs: {}, does not match the length of the specified outputs: {} ({})'.format(len(values_returned), len(self._output_names), self._output_names)
            self._state.update((state_name, values_returned[ix]) for state_name, ix in zip(self._state_names, self._state_indices_in_output))
        return values_returned if self._return_index is None else values_returned[self._return_index] if isinstance(self._return_index, int) else tuple(values_returned[i] for i in self._return_index)

    @property
    def state(self):
        return self._state.copy()


def scannable(state, output=None, returns=None):
    """
    A decorator for turning functions into simple classes.  The decorator attaches a "scan" method to the given function,
    which

    e.g.

        @scannable(state=dict(avg=0))
        def moving_average(x, avg, decay):
            return (1-decay)*avg + decay*x

        exponential_moving_average = simple_scannable.scan()
        smooth_signal = [exponential_moving_average(x, decay=0.1) for x in seq]


    :param func: A function
    :return: func, but with a "scan" function attched.
    """
    def wrapper(func):

        def create_scannable(**kwargs):
            override_state = state.copy()
            override_state.update(**kwargs)
            return Scannable(func=func, state=override_state, output=output, returns=returns)

        func.scan = create_scannable
        return func

    return wrapper
