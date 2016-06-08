from artemis.config import get_artemis_config

__author__ = 'peter'

config = get_artemis_config()
BACKEND = config.get('plotting', 'backend')

assert BACKEND in ('matplotlib', 'matplotlib-web', 'bokeh'), 'Your config file ~/.artimisrc lists "%s" as the backend.  Valid backends are "matplotlib" and "bokeh".  Change the file.' % (BACKEND, )

if BACKEND in ('matplotlib', 'matplotlib-web'):
    from matplotlib.pyplot import *
    from artemis.plotting.matplotlib_backend import *
elif BACKEND == 'bokeh':
    from artemis.plotting.bokeh_backend import *
