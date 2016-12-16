from collections import OrderedDict

import numpy as np
from artemis.general.should_be_builtins import remove_duplicates
from artemis.general.tables import build_table
from artemis.ml.datasets.datasets import DataSet
from artemis.ml.tools.iteration import zip_minibatch_iterate_info

__author__ = 'peter'

"""
All evaluation functions in here are of the form

score = evaluation_fcn(actual, target)

Where:
    score is a scalar
    actual is an (n_samples, ...) array
    target is an (n_samples, ....) array
"""


def train_online_predictor(predictor, training_set, minibatch_size, n_epochs = 1):
    """
    Train a predictor on the training set
    :param predictor: An IPredictor object
    :param training_set: A DataCollection object
    :param minibatch_size: An integer, or 'full' for full batch training
    :param n_epochs: Number of passes to make over the training set.
    """
    print 'Training Predictor %s...' % (predictor, )
    for (_, data, target) in training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_epochs, single_channel = True):
        predictor.train(data, target)
    print 'Done.'


def evaluate_predictor(predictor, test_set, evaluation_function):
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)
    output = predictor.predict(test_set.input)
    score = evaluation_function(actual = output, target = test_set.target)
    return score


def get_evaluation_function(name):
    return {
        'mse': mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'mean_l1_error': mean_l1_error,
        'percent_argmax_correct': percent_argmax_correct,
        'percent_argmax_incorrect': percent_argmax_incorrect,
        'percent_correct': percent_correct,
        }[name]


def mean_l1_error(actual, target):
    return np.mean(np.sum(np.abs(actual-target), axis=-1), axis=-1)


def mean_squared_error(actual, target):
    return np.mean(np.sum((actual-target)**2, axis = -1), axis = -1)


def fraction_correct(actual, target):
    return np.mean(actual == target)


def percent_correct(actual, target):
    return 100*fraction_correct(actual, target)


def percent_argmax_correct(actual, target):
    """
    :param actual: An (n_samples, n_dims) array
    :param target: An (n_samples, ) array of indices OR an (n_samples, n_dims) array
    :return:
    """
    actual = collapse_onehot_if_necessary(actual)
    target = collapse_onehot_if_necessary(target)
    return 100*fraction_correct(actual, target)


def percent_binary_incorrect(actual, target):
    return 100.-percent_binary_correct(actual, target)


def percent_binary_correct(actual, target):
    """
    :param actual:  A (n_samples, ) array of floats between 0 and 1
    :param target: A (n_samples, ) array of True/False
    :return: The percent of times the "actual" was closes to the correct.
    """
    assert len(actual) == len(target)
    assert target.ndim==1
    if actual.ndim>1:
        assert actual.shape == (len(target), 1)
        actual = actual[:, 0]
    if np.array_equal(np.unique(target), (0, 1)):
        assert np.all(actual)>=0 and np.all(actual)<=1
        assert np.all((target==0)|(target==1))
        return 100*np.mean(np.round(actual) == target)
    elif np.array_equal(np.unique(target), (-1, 1)):
        assert np.all((target==-1)|(target==1))
        return 100*np.mean((actual>0)*2-1 == target)
    else:
        raise Exception("Go away I'm tired.")


def percent_argmax_incorrect(actual, target):
    return 100 - percent_argmax_correct(actual, target)


def collapse_onehot_if_necessary(output_data):
    """
    Given an input that could either be in onehot encoding or not, return it in onehot encoding.

    :param output_data: Either an (n_samples, n_dims) array, or an (n_samples, ) array of labels.
    :return: An (n_samples, ) array.
    """

    output_data = np.squeeze(output_data)

    if output_data.ndim == 2:
        return np.argmax(output_data, axis = 1)
    else:
        assert output_data.ndim == 1 and output_data.dtype in (int, 'int32', bool)
        return output_data


def assess_prediction_functions(test_pairs, functions, costs, print_results=False):
    """

    :param test_pairs: A list<pair_name, (x, y)>, where x, y are equal-length vectors representing the samples in a dataset.
        Eg. [('training', (x_train, y_train)), ('test', (x_test, y_test))]
    :param functions: A list<function_name, function> of functions for computing the forward pass.
    :param costs: A list<cost_name, cost_function> of cost functions, where cost_function has the form:
        cost = cost_fcn(guess, y), where cost is a scalar, and guess is the output of the prediction function given one
            of the inputs (x) in test_pairs.
    :return: An OrderedDict: (test_pair_name, function_name, cost_name) -> cost
    """
    if isinstance(test_pairs, DataSet):
        test_pairs = [
            ('train', (test_pairs.training_set.input, test_pairs.training_set.target)),
            ('test', (test_pairs.test_set.input, test_pairs.test_set.target)),
            ]
    assert isinstance(test_pairs, list)
    assert all(len(_)==2 for _ in test_pairs)
    assert all(len(pair)==2 for name, pair in test_pairs)
    if callable(functions):
        functions = [(functions.__name__ if hasattr(functions, '__name__') else None, functions)]
    else:
        assert all(callable(f) for name, f in functions)
    if callable(costs):
        costs = [(costs.__name__, costs)]
    else:
        costs = [(cost, get_evaluation_function(cost)) if isinstance(cost, basestring) else (cost.__name__, cost) if callable(cost) else cost for cost in costs]
    assert all(callable(cost) for name, cost in costs)

    results = OrderedDict()
    for test_pair_name, (x, y) in test_pairs:
        for function_name, function in functions:
            for cost_name, cost_function in costs:
                results[test_pair_name, function_name, cost_name] = cost_function(function(x), y)

    if print_results:
        print_score_results(results)

    return results


def print_score_results(results):
    """
    :param results: An OrderedDict in the format returned by assess_prediction_functions.
    :return:
    """
    test_pair_names, function_names, cost_names = [remove_duplicates(k) for k in zip(*results.keys())]
    rows = build_table(
        lookup_fcn=lambda (test_pair_name_, function_name_), cost_name_: results[test_pair_name_, function_name_, cost_name_],
        row_categories = [[test_pair_name for test_pair_name in test_pair_names], [function_name for function_name in function_names]],
        column_categories = [cost_name for cost_name in cost_names],
        row_header_labels=['Subset', 'Function'],
        clear_repeated_headers = False,
        remove_unchanging_cols=True
        )
    import tabulate
    print tabulate.tabulate(rows)


def training_iterator(dataset, train_fcn, predict_fcn, minibatch_size, n_epochs=None, test_epochs=None,
        score_measure='percent_argmax_correct', bigger_is_better=True, enter_on='test'):
    """
    Takes care of the common tasks of training.
    :param dataset: A DataSet object
    :param train_fcn: A function of the form train_fcn(x, y) which updates the parameters
    :param predict_fcn: A function of the form y=predict_fcn(x) which makes a prediction giben inputs
    :param minibatch_size: Minibatch size
    :param n_epochs: Number of epoch
    :param test_epochs: Test epcohs
    :param enter_on:
    :return: Yields:
        score: The score for the current iteration
        best_score: The best score
    """
    assert enter_on in ('every', 'test')
    best_score = None
    for (x_mini, y_mini), info in zip_minibatch_iterate_info(dataset.training_set.xy, minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=test_epochs):
        if info.test_now:
            print 'Epoch {}'.format(info.epoch)
            score = assess_prediction_functions(dataset, functions=predict_fcn, costs=percent_argmax_correct, print_results=True)
            best_score = score if best_score is None or score['test', None, score_measure] > best_score['test', None, score_measure] else best_score
        if enter_on=='every' or enter_on=='test' and info.test_now:
            yield score, best_score
        if not info.done:
            train_fcn(x_mini, y_mini)
    print 'Best Score:'
    print_score_results(best_score)
