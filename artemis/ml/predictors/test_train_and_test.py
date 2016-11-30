from artemis.general.mymath import sigm
from artemis.ml.datasets.synthetic_linear import get_synthethic_linear_dataset
from artemis.ml.predictors.train_and_test import assess_prediction_functions, mean_squared_error, percent_argmax_correct
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

    assert results['train', 'prediction', 'percent_argmax_correct'] == 22.3


if __name__ == '__main__':
    test_assess_prediction_functions()
