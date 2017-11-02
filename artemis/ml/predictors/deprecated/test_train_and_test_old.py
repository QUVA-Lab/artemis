from artemis.general.mymath import sigm
from artemis.ml.tools.costs import mean_squared_error, percent_argmax_correct
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from artemis.ml.datasets.synthetic_linear import get_synthethic_linear_dataset
from artemis.ml.predictors.deprecated.train_and_test_old import assess_prediction_functions, \
    train_and_test_online_predictor
from artemis.ml.predictors.logistic_regressor import LogisticRegressor
import numpy as np


def test_assess_prediction_functions(print_results=True):

    x_tr, y_tr, x_ts, y_ts = get_synthethic_linear_dataset(n_input_dims=80, n_output_dims=4).xyxy

    w = np.random.RandomState(1234).randn(80, 4)

    results = assess_prediction_functions(
        test_pairs=[('train', (x_tr, y_tr)), ('test', (x_ts, y_ts))],
        functions=[('prediction', lambda x: sigm(x.dot(w)))],
        costs = [mean_squared_error, percent_argmax_correct],
        print_results=print_results
        )

    assert results['train', 'prediction', 'percent_argmax_correct'] == 23.1


def test_train_online_predictor():
    """
    Demonstrates how to train the online predictor.
    """
    predictor = LogisticRegressor.from_init(n_in=20, n_out=4)
    info_score_pairs = train_and_test_online_predictor(
        dataset = get_synthetic_clusters_dataset(n_clusters=4, n_dims=20),
        predict_fcn=predictor.predict,
        train_fcn = predictor.train,
        score_measure = 'percent_argmax_correct',
        minibatch_size=10,
        test_epochs=[0, 0.2, 0.5, 0.8, 1],
        )
    assert [info.epoch for info, score in info_score_pairs] == [0, 0.2, 0.5, 0.8, 1]
    infos, scores = zip(*info_score_pairs)
    first_score, last_score = scores[0], scores[-1]
    assert first_score['test', 'predict', 'percent_argmax_correct'] < 40 and last_score['test', 'predict', 'percent_argmax_correct'] > 99
    info_score_pairs.get_best_value('test', 'predict', 'percent_argmax_correct') > 99


if __name__ == '__main__':
    test_assess_prediction_functions()
    test_train_online_predictor()
