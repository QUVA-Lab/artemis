from artemis.config import get_artemis_config
from artemis.remote.utils import is_valid_ip

__author__ = 'peter'

config = get_artemis_config()
BACKEND = config.get('plotting', 'backend')
if config.has_option('plotting','plotting_server'):
    _USE_SERVER = True
    _PLOTTING_SERVER = config.get('plotting','plotting_server')
    assert is_valid_ip(_PLOTTING_SERVER), "Please specify a valid ip-address for the plotting server. You provided: %s"%_PLOTTING_SERVER
else:
    _USE_SERVER = False
    _PLOTTING_SERVER = ""

assert BACKEND in ('matplotlib', 'matplotlib-web', 'bokeh'), 'Your config file ~/.artimisrc lists "%s" as the backend.  Valid backends are "matplotlib" and "bokeh".  Change the file.' % (BACKEND, )

if BACKEND in ('matplotlib', 'matplotlib-web'):
    from matplotlib.pyplot import *
    from artemis.plotting.matplotlib_backend import *
elif BACKEND == 'bokeh':
    from artemis.plotting.bokeh_backend import *
