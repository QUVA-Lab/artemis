from contextlib import contextmanager

from artemis.general.global_vars import get_global, set_global
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
