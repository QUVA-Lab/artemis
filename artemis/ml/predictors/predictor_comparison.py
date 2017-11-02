import time
from collections import OrderedDict

import numpy as np

from artemis.general.checkpoint_counter import CheckPointCounter
from artemis.general.should_be_builtins import bad_value
from artemis.ml.tools.costs import get_evaluation_function
from artemis.ml.tools.iteration import checkpoint_minibatch_index_generator
from artemis.general.mymath import sqrtspace
from artemis.ml.tools.processors import RunningAverage


def compare_predictors(dataset, online_predictors={}, offline_predictors={}, minibatch_size = 'full',
        evaluation_function = 'mse', test_epochs = sqrtspace(0, 1, 10), report_test_scores = True,
        test_on = 'training+test', test_batch_size = None, accumulators = None, online_test_callbacks = {}):
    """
    DEPRECATED: use train_and_test_online_predictor instead.

    Compare a set of predictors by running them on a dataset, and return the learning curves for each predictor.

    :param dataset: A DataSet object
    :param online_predictors: A dict<str:IPredictor> of online predictors.  An online predictor is
        sequentially fed minibatches of data and updates its parameters with each minibatch.
    :param offline_predictors: A dict<str:object> of offline predictors.  Offline predictors obey sklearn's
        Estimator/Predictor interfaces - ie they methods
            estimator = object.fit(data, targets) and
            prediction = object.predict(data)
    :param minibatch_size: Size of the minibatches to use for online predictors.  Can be:
        An int, in which case it represents the minibatch size for all classifiers.
        A dict<str: int>, in which case you can set the minibatch size per-classifier.
        In place of the int, you can put 'all' if you want to train on the whole dataset in each iteration.
    :param test_epochs: Test points to use for online predictors.  Can be:
        A list of integers - in which case the classifier is after seeing this many samples.
        A list of floats - in which case the classifier is tested after seeing this many epochs.
        'always' - In which case a test is performed after every training step
        The final test point determines the end of training.
    :param evaluation_function: Function used to evaluate output of predictors
    :param report_test_scores: Boolean indicating whether you'd like to report results online.
    :param test_on: 'training', 'test', 'training+test'
    :param test_batch_size: When the test set is too large to process in one step, use this to break it
        up into chunks.
    :param accumulators: A dict<str: accum_fcn>, where accum_fcn is a stateful-function of the form:
        accmulated_output = accum_fcn(this_output)
        Special case: accum_fcn can be 'avg' to make a running average.
    :param online_test_callbacks: A dict<str: fcn> where fcn is a callback that takes an online
        predictor as an argument.  Useful for logging/plotting/debugging progress during training.
    :return: An OrderedDict<LearningCurveData>
    """

    all_keys = online_predictors.keys()+offline_predictors.keys()
    assert len(all_keys) > 0, 'You have to give at least one predictor.  Is that too much to ask?'
    assert len(all_keys) == len(np.unique(all_keys)), "You have multiple predictors using the same names. Change that."
    type_constructor_dict = OrderedDict(
        [(k, ('offline', offline_predictors[k])) for k in sorted(offline_predictors.keys())] +
        [(k, ('online', online_predictors[k])) for k in sorted(online_predictors.keys())]
        )

    minibatch_size = _pack_into_dict(minibatch_size, expected_keys=online_predictors.keys())
    accumulators = _pack_into_dict(accumulators, expected_keys=online_predictors.keys())
    online_test_callbacks = _pack_into_dict(online_test_callbacks, expected_keys=online_predictors.keys(), allow_subset=True)
    test_epochs = np.array(test_epochs)
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)

    records = OrderedDict()

    # Run the offline predictors
    for predictor_name, (predictor_type, predictor) in type_constructor_dict.items():
        print('%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20))
        records[predictor_name] = \
            assess_offline_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                report_test_scores = report_test_scores,
                test_on = test_on,
                test_batch_size = test_batch_size
                ) if predictor_type == 'offline' else \
            assess_online_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                test_epochs = test_epochs,
                accumulator = accumulators[predictor_name],
                minibatch_size = minibatch_size[predictor_name],
                report_test_scores = report_test_scores,
                test_on = test_on,
                test_batch_size = test_batch_size,
                test_callback=online_test_callbacks[predictor_name] if predictor_name in online_test_callbacks else None
                ) if predictor_type == 'online' else \
            bad_value(predictor_type)

    print('Done!')

    return records


def _pack_into_dict(value_or_dict, expected_keys, allow_subset = False):
    """
    Used for when you want to either
        a) Distribute some value to all predictors
        b) Distribute different values to different predictors and check that the names match up.
    :param value_or_dict: Either
        a) A value
        b) A dict<predictor_name: value_for_predictor>
    :param expected_keys: Names of predictors
    :return: A dict<predictor_name: value_for_predictor>
    """
    if not isinstance(value_or_dict, dict):
        output_dict = {predictor_name: value_or_dict for predictor_name in expected_keys}
    else:
        output_dict = value_or_dict
        if allow_subset:
            assert set(value_or_dict.keys()).issubset(expected_keys), 'Expected a subset of: %s.  Got %s' % (expected_keys, value_or_dict.keys())
        else:
            assert set(expected_keys) == set(value_or_dict.keys()), 'Expected keys: %s.  Got %s' % (expected_keys, value_or_dict.keys())
    return output_dict


def dataset_to_testing_sets(dataset, test_on = 'training+test'):
    return \
        {'Training': (dataset.training_set.input, dataset.training_set.target), 'Test': (dataset.test_set.input, dataset.test_set.target)} if test_on == 'training+test' else \
        {'Test': (dataset.test_set.input, dataset.test_set.target)} if test_on == 'test' else \
        {'Training': (dataset.training_set.input, dataset.training_set.target)} if test_on == 'training' else \
        bad_value(test_on)


def assess_offline_predictor(predictor, dataset, evaluation_function, test_on = 'training+test', report_test_scores=True,
        test_batch_size = None):
    """
    Train an offline predictor and return the LearningCurveData (which will not really be a curve,
    but just a point representing the final score.

    :param predictor:  An object with methods fit(X, Y), predict(X)
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: LearningCurveData containing the score on the test sets
    """
    record = LearningCurveData()
    predictor.fit(dataset.training_set.input, dataset.training_set.target)
    testing_sets = dataset_to_testing_sets(dataset, test_on)
    scores = [(k, evaluation_function(process_in_batches(predictor.predict, x, test_batch_size), y)) for k, (x, y) in testing_sets.items()]
    record.add(None, scores)
    if report_test_scores:
        print('Scores: %s' % (scores, ))
    return record


def assess_online_predictor(predictor, dataset, evaluation_function, test_epochs, minibatch_size, test_on = 'training+test',
        accumulator = None, report_test_scores=True, test_batch_size = None, test_callback = None):
    """
    DEPRECATED: use assess_prediction_functions_on_generator in train_and_test_old.py

    Train an online predictor and return the LearningCurveData.

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param test_epochs: List of epochs to test at.  Eg. [0.5, 1, 2, 4]
    :param minibatch_size: Number of samples per minibatch, or:
        'full' to do full-batch.
        'stretch': to stretch the size of each batch so that we make just one call to "train" between each test.  Use
            this, for instance, if your predictor trains on one sample at a time in sequence anyway.
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :param test_callback: A callback which takes the predictor, and is called every time a test
        is done.  This can be useful for plotting/debugging the state.
    :return: LearningCurveData containing the score on the test sets

    """
    # TODO: Remove this class, as it is deprecated

    record = LearningCurveData()

    testing_sets = dataset_to_testing_sets(dataset, test_on)
    if accumulator is None:
        prediction_functions = {k: predictor.predict for k in testing_sets}
    else:
        accum_constructor = {'avg': RunningAverage}[accumulator] \
            if isinstance(accumulator, str) else accumulator
        accumulators = {k: accum_constructor() for k in testing_sets}
        prediction_functions = {k: lambda inp, kp=k: accumulators[kp](predictor.predict(inp)) for k in testing_sets}
        # Bewate the in-loop lambda - but I think we're ok here.

    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)

    def do_test(current_epoch):
        scores = [(k, evaluation_function(process_in_batches(prediction_functions[k], x, test_batch_size), y)) for k, (x, y) in testing_sets.items()]
        if report_test_scores:
            print('Scores at Epoch %s: %s, after %.2fs' % (current_epoch, ', '.join('%s: %.3f' % (set_name, score) for set_name, score in scores), time.time()-start_time))
        record.add(current_epoch, scores)
        if test_callback is not None:
            record.add(current_epoch, ('callback', test_callback(predictor)))

    start_time = time.time()
    if minibatch_size == 'stretch':
        test_samples = (np.array(test_epochs) * dataset.training_set.n_samples).astype(int)
        i=0
        if test_samples[0] == 0:
            do_test(i)
            i += 1
        for indices in checkpoint_minibatch_index_generator(n_samples=dataset.training_set.n_samples, checkpoints=test_samples, slice_when_possible=True):
            predictor.train(dataset.training_set.input[indices], dataset.training_set.target[indices])
            do_test(test_epochs[i])
            i += 1
    else:
        checker = CheckPointCounter(test_epochs)
        last_n_samples_seen = 0
        for (n_samples_seen, input_minibatch, target_minibatch) in \
                dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = float('inf'), single_channel = True):
            current_epoch = (float(last_n_samples_seen))/dataset.training_set.n_samples
            last_n_samples_seen = n_samples_seen
            time_for_a_test, done = checker.check(current_epoch)
            if time_for_a_test:
                do_test(current_epoch)
            if done:
                break
            predictor.train(input_minibatch, target_minibatch)

    return record


def process_in_batches(func, data, batch_size):
    """
    Sometimes a function requires too much internal memory, so you have to process things in batches.
    """
    if batch_size is None:
        return func(data)

    n_samples = len(data)
    chunks = np.arange(int(np.ceil(float(n_samples)/batch_size))+1)*batch_size
    assert len(chunks)>1
    out = None
    for ix_start, ix_end in zip(chunks[:-1], chunks[1:]):
        x = data[ix_start:ix_end]
        y = func(x)
        if out is None:
            out = np.empty((n_samples, )+y.shape[1:], dtype = y.dtype)
            out[ix_start:ix_end] = y
    return out


class LearningCurveData(object):
    """
    A container for the learning curves resulting from running a predictor
    on a dataset.  Use this object to incrementally write results, and then
    retrieve them as a whole.
    """
    def __init__(self):
        self._times = {}
        self._scores = OrderedDict()
        self._latest_score = None

    def add(self, time, scores):
        """
        :param time: Something representing the time at which the record was taken.
        :param scores: A list of 2-tuples of (score_name, score).  It can also be a scalar score.
            Eg: [('training', 0.104), ('test', 0.119)]
        :return:
        """
        if np.isscalar(scores):
            scores = [('Score', scores)]
        elif isinstance(scores, tuple):
            scores = [scores]
        else:
            assert isinstance(scores, list) and all(len(s) == 2 for s in scores)

        for k, v in scores:
            if k not in self._scores:
                self._times[k] = []
                self._scores[k] = []
            self._times[k].append(time)
            self._scores[k].append(v)

    def get_results(self):
        """
        :return: (times, results), where:
            times is a length-N vector indicating the time of each test
            scores is a (length_N, n_scores) array indicating the each score at each time
                OR a (length_N, n_scores, n_reps) array where n_reps indexes each repetition or the same experiment
        """
        return {k: np.array(t) for k, t in self._times.items()}, OrderedDict((k, np.array(v)) for k, v in self._scores.items())

    def get_scores(self, which_test_set = None):
        """
        :return: scores for the given test set.
            For an offline predictor, scores'll be float
            For an online predictor, scores'll by a 1-D array representing the score at each test point.
        """
        _, results = self.get_results()

        if which_test_set is None:
            assert len(results)==1, 'You failed to specify which test set to use, which would be fine if there was only ' \
                "one, but there's more than one.  There's %s" % (results.keys(), )
            return next(v for v in results.values())
        else:
            assert which_test_set in results, 'You asked for results for the test set %s, but we only have test sets %s' \
                % (which_test_set, results.keys())
            return results[which_test_set]
