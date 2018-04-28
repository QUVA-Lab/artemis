import numpy as np
from artemis.experiments.decorators import experiment_function
from matplotlib import pyplot as plt
from six.moves import xrange

__author__ = 'peter'


"""
This file demonstates Artemis's "Experiments"

When you run an experiment, all figures and console output, as well as some metadata such as total run time, arguments,
etc are saved to disk.

This demo illustrates how you can create an experiment, create variations on that experiment, and view the results.
"""


class OnlineLinearRegressor:

    def __init__(self, n_in, n_out, learning_rate = 0.01):
        self.w = np.zeros((n_in, n_out))
        self.learning_rate = learning_rate

    def train(self, x, targ):  # x: (n_samples, n_in), targ: (n_samples, n_out)
        y = self.predict(x)
        self.w -= self.learning_rate * (x.T.dot(y-targ))

    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)


@experiment_function
def demo_linear_regression(
        n_in = 100,
        n_out = 4,
        n_training_samples = 500,
        n_test_samples = 500,
        noise = .1,
        n_epochs = 10,
        eta = 0.001,
        random_seed = 1234,
        score_report_period = 100,
        ):
    """
    Generate a random linear regression problem and train an online predictor to solve it with Stochastic gradient descent.
    Log the scores and plot the resulting learning curves.

    :param n_in: Number of inputs
    :param n_out: Number of outputs
    :param n_training_samples: Number of training samples in generated dataset.
    :param n_test_samples: Number of test samples in generated dataset.
    :param noise: Noise to add to generated dataset
    :param n_epochs: Number of epochs to run for
    :param eta: Learning rate for SGD
    :param random_seed: Random seed (for generating data)
    :param score_report_period: Report score every X training iterations.
    """

    # Setup data
    rng = np.random.RandomState(random_seed)
    w_true = rng.randn(n_in, n_out)*.1  # (n_in, n_out)
    training_data = rng.randn(n_training_samples, n_in)  # (n_training_samples, n_in)
    training_target = training_data.dot(w_true) + noise*rng.randn(n_training_samples, n_out)  # (n_training_samples, n_out)
    test_data = rng.randn(n_test_samples, n_in)  # (n_test_samples, n_in)
    test_target = test_data.dot(w_true) + noise*rng.randn(n_test_samples, n_out)  # (n_test_samples, n_out)
    predictor = OnlineLinearRegressor(n_in=n_in, n_out=n_out, learning_rate=eta)

    # Train and periodically record scores.
    epoch_scores = []
    for i in xrange(n_training_samples*n_epochs+1):
        if i % score_report_period == 0:
            training_out = predictor.predict(training_data)
            training_cost = ((training_target-training_out)**2).sum(axis=1).mean(axis=0)
            test_out = predictor.predict(test_data)
            test_cost = ((test_target-test_out)**2).sum(axis=1).mean(axis=0)
            print('Epoch {epoch}: Test Cost: {test}, Training Cost: {train}'.format(epoch=float(i)/n_training_samples, test=test_cost, train=training_cost))
            epoch = float(i) / n_training_samples
            epoch_scores.append((epoch, training_cost, test_cost))
        predictor.train(training_data[[i % n_training_samples]], training_target[[i % n_training_samples]])

    # Plot
    epochs, training_costs, test_costs = zip(*epoch_scores)
    plt.plot(epochs, np.array([training_costs, test_costs]).T)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(['Training Cost', 'Test Cost'])
    plt.title("Learning Curve")
    plt.ion()
    plt.show()

    return {'training_cost': training_cost, 'test_cost': test_cost}


demo_linear_regression.add_variant('fast-learn', eta=0.01)
demo_linear_regression.add_variant('large_input_space', n_in=1000)


if __name__ == "__main__":
    # Open a menu that allows you to run experiments and view old ones.
    demo_linear_regression.browse(display_format="flat")


