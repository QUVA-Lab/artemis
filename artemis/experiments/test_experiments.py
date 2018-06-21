import pytest

from artemis.experiments import experiment_root, experiment_function
from artemis.experiments.experiments import experiment_testing_context
import numpy as np

def test_unpicklable_args():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_root
        def my_parametrizable_exp(f, x):
            return f(x)

        X = my_parametrizable_exp.add_variant(f = (lambda x: 2*x), x=3)
        X.run()
        X.browse(command='q')


class _ExponentialMovingAverageForTestingPurposes(object):

    def __init__(self, decay):
        self.decay = decay
        self.avg = 0

    def __call__(self, x):
        self.avg = (1.-self.decay) * self.avg + self.decay*x
        return self.avg


def test_config_variant():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def demo_smooth_out_signal(smoother, seed = 1234):
            signal = np.sin(np.linspace(0, 10, 100)) + 0.1*np.random.RandomState(seed).randn(100)
            y = np.array([smoother(xt) for xt in signal])
            return y

        X = demo_smooth_out_signal.add_config_variant('exp_smooth', smoother = lambda decay=0.1: _ExponentialMovingAverageForTestingPurposes(decay))
        answer = X.run().get_result()
        assert answer.shape==(100, )
        assert np.array_equal(X.run().get_result(), answer)  # Check that we're definitely making a new one each time

        # Make sure we can still configure new args
        X2 = X.add_variant(seed=1235)
        answer2 = X2.run().get_result()
        assert answer2.shape == (100, )
        assert not np.array_equal(answer, answer2)

        # Just check for no bugs in UI
        demo_smooth_out_signal.browse(command='q')


def test_config_bug_catching():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def demo_smooth_out_signal_testing(smoother, seed = 1234):
            signal = np.sin(np.linspace(0, 10, 100)) + 0.1*np.random.RandomState(seed).randn(100)
            y = np.array([smoother(xt) for xt in signal])
            return y

        # The right way
        X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth', smoother = lambda decay=0.1: _ExponentialMovingAverageForTestingPurposes(decay))

        # Also acceptable, so long as arg names do not conflict
        X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth2', smoother = _ExponentialMovingAverageForTestingPurposes)

        with pytest.raises(AssertionError):  # Arg name already used!
            X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth3', smoother = lambda seed=0.1: _ExponentialMovingAverageForTestingPurposes(decay=seed))

        with pytest.raises(AssertionError):  # Make sure we catch when we do not give a callable
            X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth4', smoother = 0.1)

        with pytest.raises(AssertionError):  # Make sure we catch when we give the wrong name
            X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth5', averager = lambda decay: _ExponentialMovingAverageForTestingPurposes(decay=decay))

        with pytest.raises(AssertionError):  # Catch when we accidentally give an instance:
            X = demo_smooth_out_signal_testing.add_config_variant('exp_smooth', smoother = _ExponentialMovingAverageForTestingPurposes(decay=0.1))


if __name__ == '__main__':
    test_unpicklable_args()
    test_config_variant()
    test_config_bug_catching()
