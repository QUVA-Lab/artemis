from setuptools import setup, find_packages

setup(
    name='artemis-ml',
    author='Peter & Matthias',
    author_email='poconn4@gmail.com',
    url='https://github.com/quva-lab/artemis',
    long_description='Artemis aims to get rid of all the boring, bureaucratic coding (plotting, file management, etc) involved in machine learning projects, so you can get to the good stuff quickly.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pytest', 'pillow', 'tabulate'],
    version='1.2',
    packages=find_packages(),
    scripts=[])
