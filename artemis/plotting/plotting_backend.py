from artemis.config import get_artemis_config_value

__author__ = 'peter'

_PLOTTING_SERVER = get_artemis_config_value(section='plotting', option='plotting_server', default_generator="")
_USE_SERVER = _PLOTTING_SERVER != ""


BACKEND = get_artemis_config_value(section='plotting', option='backend')
assert BACKEND in ('matplotlib', 'matplotlib-web', 'bokeh'), 'Your config file ~/.artimisrc lists "%s" as the backend.  Valid backends are "matplotlib" and "bokeh".  Change the file.' % (BACKEND, )

if BACKEND in ('matplotlib', 'matplotlib-web'):
    from matplotlib.pyplot import *
    from artemis.plotting.matplotlib_backend import *
elif BACKEND == 'bokeh':
    from artemis.plotting.bokeh_backend import *


def is_server_plotting_on():
    return _USE_SERVER


def set_server_plotting(state):
    global _USE_SERVER
    _USE_SERVER = state


def get_plotting_server_address():
    return _PLOTTING_SERVER
