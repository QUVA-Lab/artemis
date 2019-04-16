from contextlib import contextmanager

_GLOBALS = {}


@contextmanager
def global_context(context_dict = None):
    global _GLOBALS
    if context_dict is None:
        context_dict = {}
    old_globals = _GLOBALS
    _GLOBALS = context_dict
    yield context_dict
    _GLOBALS = old_globals


def get_global(identifier, constructor=None):

    if identifier not in _GLOBALS:
        if constructor is not None:
            _GLOBALS[identifier] = constructor()
        else:
            raise KeyError('No global variable with key: {}'.format(identifier))
    return _GLOBALS[identifier]


def has_global(identifier):
    return identifier in _GLOBALS


def set_global(identifier, value):

    _GLOBALS[identifier] = value
