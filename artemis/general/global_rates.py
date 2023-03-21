from contextlib import contextmanager

from artemis.general.global_vars import get_global, set_global, has_global
import time


class _RateMeasureSingleton:
    pass


def measure_global_rate(name, n_steps = None):
    this_time = time.time()
    key = (_RateMeasureSingleton, name)
    n_calls, start_time = get_global(key, constructor=lambda: (0, this_time))
    if n_steps is None:
        n_steps = n_calls
    set_global(key, (n_steps+1, start_time))
    return n_steps / (this_time - start_time) if this_time!=start_time else float('inf')


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


def elapsed_time(identifier, current = None):
    """
    Return the time that has elapsed since this function was called with the given identifier.
    """
    if current is None:
        current = time.time()
    key = (_LastTimeMeasureSingleton, identifier)

    if not has_global(key):
        set_global(key, current)
        return float('inf')
    else:
        last = get_global(key)
        assert current>=last, "Current value ({}) must be greater or equal to the last value ({})".format(current, last)
        elapsed = current - last
        set_global(key, current)
        return elapsed


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
        assert current>=last, "Current value ({}) must be greater or equal to the last value ({})".format(current, last)
        has_elapsed = current - last >= period
        if has_elapsed:
            set_global(key, current)
        return has_elapsed


def limit_rate(identifier, period):
    """
    :param identifier: Any python object to uniquely identify what you're limiting.
    :param period: The minimum period
    :param current: The time measure (if None, system time will be used)
    :return: Whether the rate was exceeded (True) or not (False)
    """

    enter_time = time.time()
    key = (_LastTimeMeasureSingleton, identifier)
    if not has_global(key):  # First call
        set_global(key, enter_time)
        return False
    else:
        last = get_global(key)
        assert enter_time>=last, "Current value ({}) must be greater or equal to the last value ({})".format(enter_time, last)
        elapsed = enter_time - last
        if elapsed < period:  # Rate has been exceeded
            time.sleep(period - elapsed)
            set_global(key, time.time())
            return False
        else:
            set_global(key, enter_time)
            return True


def limit_iteration_rate(iterable, period):
    for x in iterable:
        limit_rate(id(iterable), period=period)
        yield x
