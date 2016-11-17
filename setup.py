

from setuptools import setup
import os

CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.artemisrc')

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        f.write('[plotting]\nbackend: matplotlib')

setup(name='artemis',
      author='Peter & Matthias',
      author_email='poconn4@gmail.com',
      url='https://github.com/quva-lab/artemis',
      long_description='A Library for plotting and managing experiments.',
      install_requires=['numpy', 'scipy', 'matplotlib', 'pytest', 'pillow'],
      version=0,
      packages=['artemis'],
      scripts=[])
