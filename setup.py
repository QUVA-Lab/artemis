

from setuptools import setup
import os

CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.artemisrc')

setup(name='artemis-ml',
      author='Peter & Matthias',
      author_email='poconn4@gmail.com',
      url='https://github.com/quva-lab/artemis',
      long_description='A Library for plotting and managing experiments.',
      version=0,
      packages=['artemis-ml'],
      scripts=[])
