import numpy as np

from artemis.ml.predictors.predictor_comparison import assess_online_predictor
from artemis.ml.predictors.train_and_test import percent_argmax_correct
from plato.tools.common.bureaucracy import multichannel
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from artemis.ml.tools.processors import OneHotEncoding


__author__ = 'peter'


def assert_online_predictor_not_broken(predictor_constructor, initial_score_under = 35, final_score_over = 95, n_epochs = 1,
        minibatch_size = 'full', categorical_target = False, accumulator = None, n_extra_tests = 0):
    """
    Assert that your predictor is not a total embarrassment.  (Note that it still may pass this test and be a terrible
    predictor, this at least makes clear that it's not completely broken.)

    :param predictor_constructor: A constructor that returns an IPredictor object given (n_dims_in, n_dims_out) as
        arguments.
    :param initial_score_under: Asser that the initial score on the 4-cluster dataset (where chance is 25%) is worse than
        this (mainly just makes sure you're not cheating somehow)
    :param final_score_over: Assert that the final score is over this - Solving this dataset isn't rocket science.  It
        is not hard to get a final score of 100.
    :param n_epochs: Now many epochs should you run?
    :param minibatch_size: Minibatch size.  By default, do full-batch training.
    :param categorical_target: If True, your predictor expects an integer as a target, where the integer indicates the
        correct label.  Otherwise, it expects a one-hot encoding vector - with the unit corresponding to the label being
        True.
    :param n_extra_tests: Number of extra tests - you may set this to non-zero to see the progress of your predictor
        over training.
    """
    dataset = get_synthetic_clusters_dataset(dtype = 'float32')

    if not categorical_target:
        dataset = dataset.process_with(targets_processor=multichannel(OneHotEncoding()))
        out_shape = dataset.target_size
    else:
        out_shape = dataset.n_categories

    predictor = predictor_constructor(dataset.input_size, out_shape)
    record = assess_online_predictor(predictor, dataset, evaluation_function=percent_argmax_correct,
        test_epochs=np.linspace(0, n_epochs, 2+n_extra_tests), minibatch_size=minibatch_size,
        accumulator = accumulator, test_on = 'test')
    scores = record.get_scores()
    assert scores[0] <= initial_score_under, "Initial score was %.2f%%, which was greater than expected (<%.2f%%).  That's odd." % (scores[0], initial_score_under)
    assert scores[-1] >= final_score_over, 'Achieved a final score of %.2f%%, which was less than the threshold of %.2f%%' % (scores[-1], final_score_over)
