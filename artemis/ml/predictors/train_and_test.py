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
    elif isinstance(costs, basestring):
        costs = [(costs, get_evaluation_function(costs))]
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


def print_score_results(score, info=None):
    """
    :param results: An OrderedDict in the format returned by assess_prediction_functions.
    :return:
    """
    if info is not None:
        print 'Epoch {} (after {:.3g}s)'.format(info.epoch, info.time)
    test_pair_names, function_names, cost_names = [remove_duplicates(k) for k in zip(*score.keys())]
    rows = build_table(
        lookup_fcn=lambda (test_pair_name_, function_name_), cost_name_: score[test_pair_name_, function_name_, cost_name_],
        row_categories = [[test_pair_name for test_pair_name in test_pair_names], [function_name for function_name in function_names]],
        column_categories = [cost_name for cost_name in cost_names],
        row_header_labels=['Subset', 'Function'],
        clear_repeated_headers = False,
        remove_unchanging_cols=True
        )
    import tabulate
    print tabulate.tabulate(rows)


def train_online_predictor(dataset, train_fcn, predict_fcn, minibatch_size, n_epochs=None, test_epochs=None,
        score_measure='percent_argmax_correct', test_callback=None, training_callback = None):
    """
    Train an online predictor.  Return a data structure with info about the training.
    :param dataset: A DataSet object
    :param train_fcn: A function of the form train_fcn(x, y) which updates the parameters
    :param predict_fcn: A function of the form y=predict_fcn(x) which makes a prediction giben inputs
    :param minibatch_size: Minibatch size
    :param n_epochs: Number of epoch
    :param test_epochs: Test epcohs
    :param score_measure: String or function of the form:
        score = score_measure(guess, ground_truth)
        To be used in testing.
    :param test_callback: Function to be called on test.  It has the form: f(info, score)
    :param training_callback: Function to be called after every training iteration.  It takes no arguments.
    :return: A list<info, scores>  where...
        IterationInfo object (see artemis.ml.tools.iteration.py) with fields:
            'iteration', 'epoch', 'sample', 'time', 'test_now', 'done'
        scores is dict<(subset, prediction_function, cost_function) -> score>  where:
            subset is a string identifying the subset (eg 'train', 'test')
            prediction_function is identifies the prediction function (usually None, but can be used if you specify multiple prediction functions)
            cost_function is identifiers the cost function.
    """
    info_score_pairs = []
    for (x_mini, y_mini), info in zip_minibatch_iterate_info(dataset.training_set.xy, minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=test_epochs):
        if info.test_now:
            rate = (info.time-last_time)/(info.epoch - last_epoch) if info.epoch>0 else float('nan')
            print 'Epoch {}.  Rate: {:.3g}s/epoch'.format(info.epoch, rate)
            last_epoch = info.epoch
            last_time = info.time
            score = assess_prediction_functions(dataset, functions=predict_fcn, costs=score_measure, print_results=True)
            info_score_pairs.append((info, score))
            if test_callback is not None:
                test_callback(info, score)
        if not info.done:
            train_fcn(x_mini, y_mini)
            if training_callback is not None:
                training_callback()
    return info_score_pairs


def get_best_score(score_info_pairs, subset = 'test', prediction_function = None, score_measure = None, lower_is_better = False):
    """
    Given a list of (info, score) pairs which represet the progress over training, find the best score and return it.
    :param score_info_pairs: A list<(IterationInfo, dict)> of the type returned in train_online_predictor
    :param subset: 'train' or 'test' ... which subset to use to look for the best score
    :param prediction_function: Which prediction function (if there are multiple prediction functions, otherwise leave blank)
    :param score_measure: Which score measure (if there are multiple score measures, otherwise leave blank)
    :param lower_is_better: True if a lower score is better for the chosen score_measure
    :return: best_info, best_score
        best_info is an InterationInfo object
        best_score is a dict<(subset, prediction_function, score_measure) -> score>
    """
    assert len(score_info_pairs)>0, "You need to have at least one score to determine the best one."
    _, first_score = score_info_pairs[0]
    all_subsets, all_functions, all_measures = [remove_duplicates(s) for s in zip(*first_score.keys())]
    if prediction_function is None:
        assert len(all_functions)==1, "You did not specify prediction_function... options are: {}".format(all_functions)
        prediction_function = all_functions[0]
    if score_measure is None:
        assert len(all_measures)==1, "You did not specify a score_measure... options are: {}".format(all_measures)
        score_measure = all_measures[0]
    best_info = None
    best_score = None
    for info, score in score_info_pairs:
        this_is_the_best = best_score is None or \
            (score[subset, prediction_function, score_measure]<best_score[subset, prediction_function, score_measure]) == lower_is_better
        if this_is_the_best:
            best_score = score
            best_info = info
    return best_info, best_score


def print_best_score(score_info_pairs, **best_score_kwargs):
    best_info, best_score = get_best_score(score_info_pairs, **best_score_kwargs)
    print_score_results(score=best_score, info=best_info)
