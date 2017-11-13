from setuptools import setup, find_packages

setup(
    name='artemis-ml',
    author='Peter & Matthias',
    author_email='poconn4@gmail.com',
    url='https://github.com/quva-lab/artemis',
    long_description='Artemis aims to get rid of all the boring, bureaucratic coding (plotting, file management, etc) involved in machine learning projects, so you can get to the good stuff quickly.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pytest', 'pillow', 'tabulate', 'si-prefix', 'enum34'],
    extras_require = {
        'remote_plotting': ["paramiko", "netifaces"]
        },
    version='2.0.0',
    packages=find_packages(),
    scripts=[])
