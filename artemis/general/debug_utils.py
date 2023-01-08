from builtins import Exception
from contextlib import contextmanager
from time import monotonic
from logging import Logger
PROFILE_LOG = Logger('easy_profile')

PROFILE_DEPTH = 0


@contextmanager
def easy_profile(name: str, log_entry: bool = False, enable = True, time_unit='ms'):
    if not enable:
        yield
        return
    global PROFILE_DEPTH
    tstart = monotonic()
    try:
        if log_entry:
            PROFILE_LOG.warn(f"Starting block '{name}...")
        PROFILE_DEPTH += 1
        yield
    finally:
        PROFILE_DEPTH -= 1
        elapsed = monotonic()-tstart
        time_str = f"{elapsed:.5f}s" if time_unit == 's' else \
            f"{elapsed*1000:.2f}ms" if time_unit == 'ms' else \
            f"{elapsed*1000000:.1f}us" if time_unit == 'us' else \
            f"{elapsed:.5f}s <unknown unit: '{time_unit}' requested so defaulting to seconds>"
        PROFILE_LOG.warn(f'EasyProfile: {"| "*PROFILE_DEPTH} {name} took {time_str}')
