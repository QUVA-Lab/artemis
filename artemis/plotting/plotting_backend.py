import ConfigParser
import os

__author__ = 'peter'

CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.artemisrc')

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        f.write('[plotting]\nbackend: matplotlib')


config = ConfigParser.ConfigParser()
config.read(os.path.join(os.path.expanduser('~'), '.artemisrc'))

BACKEND = config.get('plotting', 'backend')

assert BACKEND in ('matplotlib', 'bokeh'), 'Your config file ~/.artimisrc lists "%s" as the backend.  Valid backends are "matplotlib" and "bokeh".  Change the file.' % (BACKEND, )

if BACKEND == 'matplotlib':
    from artemis.plotting.matplotlib_backend import *
elif BACKEND == 'bokeh':
    from artemis.plotting.bokeh_backend import *
