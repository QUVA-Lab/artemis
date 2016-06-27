from artemis.fileman.experiment_record import register_experiment, run_experiment, get_latest_record_identifier, \
    get_latest_experiment_record
import numpy as np
from matplotlib import pyplot as plt
__author__ = 'peter'


"""
This file demonstates you you can use experiments.

After running, all results will be saved
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


def demo_linear_regression(
        n_in = 100,
        n_out = 4,
        n_training_samples = 500,
        n_test_samples = 500,
        n_epochs = 10,
        noise = .1,
        eta = 0.001,
        random_seed = 1234,
        score_report_period = 100,
        ):

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
            print 'Epoch {epoch}: Test Cost: {test}, Training Cost: {train}'.format(epoch=float(i)/n_training_samples, test=test_cost, train=training_cost)
            epoch = float(i) / n_training_samples
            epoch_scores.append((epoch, training_cost, test_cost))
        predictor.train(training_data[[i % n_training_samples]], training_target[[i % n_training_samples]])
        # predictor.train(training_data, training_target)

    # Plot
    epochs, training_costs, test_costs = zip(*epoch_scores)
    plt.plot(epochs, np.array([training_costs, test_costs]).T)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(['Training Cost', 'Test Cost'])
    plt.title("Learning Curve")
    plt.show()

    
register_experiment(
    name = 'demo_linear_regression_experiment',
    description = "Run linear regression.",
    function = demo_linear_regression,
    conclusion = "Linear Regression Works"
    )


if __name__ == "__main__":

    # First, run the experiment
    run_experiment('demo_linear_regression_experiment')

    # After this, you can show the saved results.  You can run the file "experiment_record.py" to
    # browse through all past experiments.
    # They are stored in <Your project folder>/Data/experiments/
    get_latest_experiment_record('demo_linear_regression_experiment').show()
