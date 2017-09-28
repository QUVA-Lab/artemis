import sys
import pytest
from artemis.experiments.decorators import experiment_function


def is_matplotlib_imported():
    return 'matplotlib' in sys.modules


@pytest.mark.skipif(True, reason='Test only works in isolation.  Matplotlib may be imported already when running with other tests in pytest.')
def test_matplotlib_dependency_is_soft():

    assert not is_matplotlib_imported()

    @experiment_function
    def my_test_exp(a=1, b=3):
        print('aaa')
        return a*b

    assert not is_matplotlib_imported()
    my_test_exp.run(save_figs = False)  # When Running an experiment

    my_test_exp.browse(command='q')  # Also in UI

    assert not is_matplotlib_imported()
    from matplotlib import pyplot as plt
    assert is_matplotlib_imported()


if __name__ == '__main__':
    test_matplotlib_dependency_is_soft()
