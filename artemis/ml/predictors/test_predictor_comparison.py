
import numpy as np
import pytest
from artemis.ml.predictors.learning_curve_plots import plot_learning_curves
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from artemis.ml.predictors.i_predictor import IPredictor
from artemis.general.mymath import sqrtspace
from artemis.ml.predictors.perceptron import Perceptron
from artemis.ml.predictors.predictor_comparison import compare_predictors, assess_online_predictor


__author__ = 'peter'


@pytest.mark.skip(True, reason = 'Test uses sklearn, which should not be a requirement for artemis.  It does work though.')
def test_compare_predictors(hang_plot = False):
    from sklearn.svm import SVC
    dataset = get_synthetic_clusters_dataset()

    w_constructor = lambda rng = np.random.RandomState(45): .1*rng.randn(dataset.input_shape[0], dataset.n_categories)
    records = compare_predictors(
        dataset = dataset,
        offline_predictors={
            'SVM': SVC()
            },
        online_predictors={
            'fast-perceptron': Perceptron(alpha = 0.1, w = w_constructor()).to_categorical(),
            'slow-perceptron': Perceptron(alpha = 0.001, w = w_constructor()).to_categorical()
            },
        minibatch_size = 10,
        test_epochs = sqrtspace(0, 10, 20),
        evaluation_function='percent_correct'
        )

    assert 99 < records['SVM'].get_scores('Test') <= 100
    assert 20 < records['slow-perceptron'].get_scores('Test')[0] < 40 and 95 < records['slow-perceptron'].get_scores('Test')[-1] <= 100
    assert 20 < records['fast-perceptron'].get_scores('Test')[0] < 40 and 98 < records['fast-perceptron'].get_scores('Test')[-1] <= 100

    plot_learning_curves(records, hang = hang_plot)


def test_stretch_minibatches():
    """
    Assert that the 'stretch' minibatch size works as expected.
    """
    class _SingleSampleRegressor(IPredictor):

        def __init__(self, n_in, n_out, eta = 0.1):
            self.w = np.zeros((n_in, n_out))
            self.eta = eta
            self.n_train_calls = 0

        def _predict_one(self, x):
            return x.dot(self.w)

        def _train_one(self, x, y):
            self.w += self.eta * np.outer(x, y)

        def predict(self, x):
            return np.array([self._predict_one(xi) for xi in x])

        def train(self, x, y):
            for xi, yi in zip(x, y):
                self._train_one(xi, yi)
            self.n_train_calls += 1

    dataset = get_synthetic_clusters_dataset().to_onehot()

    p1 = _SingleSampleRegressor(dataset.input_size, dataset.target_size)
    p1_scores = assess_online_predictor(p1, dataset, evaluation_function='percent_argmax_correct', test_epochs=[0, .5, 1], minibatch_size=1, test_on = 'test')
    p1_trained_out = p1.predict(dataset.test_set.input)
    assert p1.n_train_calls == dataset.training_set.n_samples
    scores = p1_scores.get_scores()
    assert len(scores) == 3 and scores[0] < 28 and scores[2] > 99

    p2 = _SingleSampleRegressor(dataset.input_size, dataset.target_size)
    assess_online_predictor(p2, dataset, evaluation_function='percent_argmax_correct', test_epochs=[0, 0.5, 1], minibatch_size='stretch',  test_on = 'test')
    p2_trained_out = p2.predict(dataset.test_set.input)
    assert p2.n_train_calls == 2

    assert np.array_equal(p1_trained_out, p2_trained_out)


if __name__ == '__main__':
    test_compare_predictors(hang_plot=True)
    test_stretch_minibatches()
