import sys
# content of conftest.py

def pytest_configure(config):
    import sys
    sys._called_from_test = True
    import matplotlib
    matplotlib.use('Agg')


def pytest_unconfigure(config):
    if hasattr(sys, '_called_from_test'):
        del sys._called_from_test
