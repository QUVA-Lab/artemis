from typing import Optional

from artemis.general.custom_types import TimeIntervalTuple


def parse_time_delta_str_to_sec(time_delta_str: str) -> Optional[float]:
    if time_delta_str in ('start', 'end'):
        return None
    else:
        start_splits = time_delta_str.split(':')
        if len(start_splits) == 1:
            return float(time_delta_str)
        elif len(start_splits) == 2:
            return 60 * float(start_splits[0]) + float(start_splits[1])
        elif len(start_splits) == 3:
            return 3600*float(start_splits[0]) + 60*float(start_splits[1]) + float(start_splits[2])
        else:
            raise Exception(f"Bad format: {time_delta_str}")


def parse_interval(interval_str: str) -> TimeIntervalTuple:

    start, end = (s.strip('') for s in interval_str.split('-'))
    return parse_time_delta_str_to_sec(start), parse_time_delta_str_to_sec(end)
