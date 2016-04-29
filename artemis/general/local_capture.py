import logging
import sys


def execute_and_capture_locals(fcn, *args, **kwargs):
    """
    Execute the function with the provided arguments, and return both the output and
    the LOCAL VARIABLES of the function.  This is crazy.  It actually looks inside the
    function at the time that return is called and grabs all the local variables.
    """
    cap = CaptureLocals()

    with cap:
        out = fcn(*args, **kwargs)

    local_vars = cap.get_captured_locals()

    return out, local_vars


class CaptureLocals(object):
    """
    Use this object to capture the locals of a function that you call.
    """
    _profiler_stack = []

    def __init__(self):

        self._locals = None

        def tracer(frame, event, arg):
            if event == 'return':
                # Note - this is called for every return of every function called within.  The only
                # reason it's ok is that the final return is always that of the decorated function.
                # Still, we're often doing thousands of unnecessary copies.
                local_variables = frame.f_locals.copy()
                self._locals = local_variables

        self._profile_fcn = tracer

    def __enter__(self):
        CaptureLocals._profiler_stack.append(self._profile_fcn)
        sys.setprofile(self._profile_fcn)
        return self

    def __exit__(self, _, _1, _2):
        CaptureLocals._profiler_stack.pop()
        new_profiler = None if len(CaptureLocals._profiler_stack)==0 else CaptureLocals._profiler_stack[-1]
        sys.setprofile(new_profiler)

    def get_captured_locals(self):
        if self._locals is None:
            raise LocalsNotCapturedError()
        return self._locals


class LocalsNotCapturedError(Exception):

    def __init__(self):
        Exception.__init__(self, "Locals not captured.  This can happen if you're calling a function from the debugger or "
            "something else that messes with the profiler.")
