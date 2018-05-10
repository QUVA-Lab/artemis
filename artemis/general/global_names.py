import itertools

_NAME_GENERATORS = {}


def create_global_name(format_str):
    if format_str not in _NAME_GENERATORS:
        _NAME_GENERATORS[format_str] = (format_str.format(i) for i in itertools.count(0))
    return next(_NAME_GENERATORS[format_str])
