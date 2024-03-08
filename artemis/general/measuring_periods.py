
import time
from typing import Optional, Callable

from dataclasses import dataclass

_last_time_dict = {}


def measure_period(identifier):
    """
    You can call this in a loop to get an easy measure of how much time has elapsed since the last call.
    On the first call it will return NaN.
    :param Any identifier:
    :return float: Elapsed time since last measure
    """
    if identifier not in _last_time_dict:
        _last_time_dict[identifier] = time.time()
        return float('nan')
    else:
        now = time.time()
        elapsed = now - _last_time_dict[identifier]
        _last_time_dict[identifier] = now
        return elapsed


@dataclass
class PeriodicChecker:
    interval: float  # Call interval in seconds
    call_at_start: bool = True
    callback: Optional[Callable[[], None]] = None
    _last_time = -float('inf')

    def is_time_for_update(self, time_now: Optional[float] = None) -> bool:
        if time_now is None:
            time_now = time.monotonic()
        call_now = self.call_at_start if self._last_time == -float('inf') else time_now-self._last_time > self.interval
        if call_now:
            if self.callback is not None:
                self.callback()
            self._last_time = time_now
        return call_now
