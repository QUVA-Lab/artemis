from contextlib import contextmanager

from artemis.general.global_vars import get_global, set_global, has_global
import time


class _RateMeasureSingleton:
    pass


def measure_global_rate(name):
    this_time = time.time()
    key = (_RateMeasureSingleton, name)
    n_calls, start_time = get_global(key, constructor=lambda: (0, this_time))
    set_global(key, (n_calls+1, start_time))
    return n_calls / (this_time - start_time) if this_time!=start_time else float('inf')


class _ElapsedMeasureSingleton:
    pass


@contextmanager
def measure_rate_context(name):
    start = time.time()
    key = (_ElapsedMeasureSingleton, name)
    n_calls, elapsed = get_global(key, constructor=lambda: (0, 0.))
    yield n_calls / elapsed if elapsed > 0 else float('nan')
    end = time.time()
    set_global(key, (n_calls+1, elapsed+(end-start)))


@contextmanager
def measure_runtime_context(name):
    start = time.time()
    key = (_ElapsedMeasureSingleton, name)
    n_calls, elapsed = get_global(key, constructor=lambda: (0, 0.))
    yield elapsed / n_calls if n_calls > 0 else float('nan')
    end = time.time()
    set_global(key, (n_calls+1, elapsed+(end-start)))


class _LastTimeMeasureSingleton:
    pass


def is_elapsed(identifier, period, current = None, count_initial = True):
    """
    Return True if the given span has elapsed since this function last returned True
    :param identifier: A string, or anything identifier
    :param period: The span which should have elapsed for this to return True again.  This is measured in time in seconds
        if no argument is provided for "current" or for whatever the unit of "current" is otherwise.
    :param current: Optionally, the current state of progress.  If ommitted, this defaults to the current time.
    :param count_initial: Count the initial point
    :return bool: True if first call or at least "span" units of time have elapsed.
    """
    if current is None:
        current = time.time()
    key = (_LastTimeMeasureSingleton, identifier)

    if not has_global(key):
        set_global(key, current)
        return count_initial
    else:
        last = get_global(key)
        assert current>=last, f"Current value ({current}) must be greater or equal to the last value ({last})"
        has_elapsed = current - last >= period
        if has_elapsed:
            set_global(key, current)
        return has_elapsed
