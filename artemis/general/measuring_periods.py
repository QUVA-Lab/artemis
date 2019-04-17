
import time

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
