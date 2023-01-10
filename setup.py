from setuptools import setup, find_packages
import re
# Get the version, following advice from https://stackoverflow.com/a/7071358/851699

VERSIONFILE="artemis/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name='artemis-ml',
    author='Peter & Matthias',
    author_email='poconn4@gmail.com',
    url='https://github.com/quva-lab/artemis',
    long_description='Artemis aims to get rid of all the boring, bureaucratic coding (plotting, file management, etc) involved in machine learning projects, so you can get to the good stuff quickly.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pytest', 'pillow', 'tabulate', 'si-prefix', 'rectangle-packer'],
    extras_require = {
        'remote_plotting': ["paramiko", "netifaces"]
        },
    version=verstr,
    packages=find_packages(),
    scripts=[])
