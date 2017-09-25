import sys
import os
# content of conftest.py

def pytest_configure(config):
    sys._called_from_test = True
    if os.environ.get('TRAVIS'):
        import matplotlib
        matplotlib.use('Agg')


def pytest_unconfigure(config):
    if hasattr(sys, '_called_from_test'):
        del sys._called_from_test
