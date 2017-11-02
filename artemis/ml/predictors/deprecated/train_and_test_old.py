# coding=utf-8
from collections import OrderedDict

import numpy as np
from six import string_types

from artemis.general.should_be_builtins import remove_duplicates
from artemis.general.tables import build_table
from artemis.ml.tools.costs import get_evaluation_function
from artemis.ml.datasets.datasets import DataSet
from artemis.ml.tools.iteration import zip_minibatch_iterate_info, IterationInfo, minibatch_process

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
    print('Training Predictor %s...' % (predictor, ))
    for (_, data, target) in training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_epochs, single_channel = True):
        predictor.train(data, target)
    print('Done.')


def evaluate_predictor(predictor, test_set, evaluation_function):
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)
    output = predictor.predict(test_set.input)
    score = evaluation_function(actual = output, target = test_set.target)
    return score


class ModelTestScore(object):
    """
    An object representing the evaluation of a model on
    - one or more test sets
    - one or more prediction functions
    - one or more costs
    """

    def __init__(self, ):
        self.scores = OrderedDict()

    def __getitem__(self, data_predfunc_cost):
        data_subset, prediction_function_name, cost_name = data_predfunc_cost
        return self.scores[data_subset, prediction_function_name, cost_name]

    def __setitem__(self, data_predfunc_cost, value):
        data_subset, prediction_function_name, cost_name = data_predfunc_cost
        self.scores[data_subset, prediction_function_name, cost_name] = value

    def keys(self):
        return self.scores.keys()

    def values(self):
        return self.scores.values()

    def items(self):
        return self.scores.items()

    def iteritems(self):
        return self.scores.items()

    def get_score(self, subset=None, prediction_function=None, cost_name=None):
        if subset is None:
            subset = self.get_only_data_subset()
        if prediction_function is None:
            prediction_function = self.get_only_prediction_function()
        if cost_name is None:
            cost_name = self.get_only_cost()
        return self[subset, prediction_function, cost_name]

    def get_data_subsets(self):
        return remove_duplicates([s for s, _, _ in self.scores.keys()])

    def get_prediction_functions(self):
        return remove_duplicates([f for _, f, _ in self.scores.keys()])

    def get_costs(self):
        return remove_duplicates([c for _, c, c in self.scores.keys()])

    def get_only_data_subset(self):
        return self._get_only_element(self.get_data_subsets(), 'Data Subset')

    def get_only_prediction_function(self):
        return self._get_only_element(self.get_prediction_functions(), 'Prediction Function')

    def get_only_cost(self):
        return self._get_only_element(self.get_costs(), 'Cost')

    @staticmethod
    def _get_only_element(elements, category_name):
        if len(elements)==1:
            result, = elements
            return result
        else:
            raise Exception("You need to specify the {}.  Options are: {}".format(category_name, elements))

    def __str__(self):
        sections = [u'({}{}{})={}'.format(
            subset,
            '' if pred_fcn is None else ','+str(pred_fcn),
            ','+str(cost_fcn),
            self.get_score(subset, pred_fcn, cost_fcn))
            for subset, pred_fcn, cost_fcn in self.keys()
            ]
        return '{}:{}'.format(self.__class__.__name__, ','.join(sections))

    def get_table(self):
        # test_pair_names, function_names, cost_names = [remove_duplicates(k) for k in zip(*self.scores.keys())]

        def lookup( test_pair_name_and_function_name_, cost_name_):
            test_pair_name_, function_name_ = test_pair_name_and_function_name_
            return self[test_pair_name_, function_name_, cost_name_]

        rows = build_table(
            lookup_fcn=lookup,
            row_categories=[[test_pair_name for test_pair_name in self.get_data_subsets()], [function_name for function_name in self.get_prediction_functions()]],
            column_categories=[cost_name for cost_name in self.get_costs()],
            row_header_labels=['Subset', 'Function'],
            clear_repeated_headers=False,
            remove_unchanging_cols=False
        )
        import tabulate
        return tabulate.tabulate(rows)


class InfoScorePair(object):

    def __init__(self, info, score):
        """
        :param info: An IterationInfo Object
        :param v: A ModelTestScore object
        """
        assert isinstance(info, IterationInfo)
        assert isinstance(score, ModelTestScore)
        self.info = info
        self.score = score

    def __iter__(self):
        return iter([self.info, self.score])

    def __str__(self):
        return 'Epoch {} (after {:.3g}s)\n{}'.format(self.info.epoch, self.info.time, self.score)

    def get_table_headers(self):
        iterfields = IterationInfo._fields
        subset_names, prediction_funcs, cost_funcs = zip(*self.score.keys())
        return [
            (' ', )*len(iterfields) + subset_names,
            (' ', )*len(iterfields) + prediction_funcs,
            iterfields + cost_funcs
            ]

    def get_table_row(self):
        return [v for v in self.info] + self.score.values()

    def get_table(self, remove_headers = False):
        from tabulate import tabulate
        iterfields = IterationInfo._fields
        subset_names, prediction_funcs, cost_funcs = zip(*self.score.keys())
        headers = [
            (' ', )*len(iterfields) + subset_names,
            (' ', )*len(iterfields) + prediction_funcs,
            iterfields + cost_funcs
            ]
        data = [v for v in self.info] + self.score.values()
        table = tabulate(headers+[data], tablefmt='plain')
        if remove_headers:
            table = table[table.rfind('\n'):]

        return table


class InfoScorePairSequence(object):
    """
    An object representing a sequence of pairs of IterationInfo, Score objects.
    """

    def __init__(self):
        self._pairs = []

    def __iter__(self):
        return iter(self._pairs)

    def __getitem__(self, ix):
        return self._pairs[ix]

    def __len__(self):
        return len(self._pairs)

    def append(self, info_score_pair):
        self._pairs.append(info_score_pair)

    def get_oneliner(self, subset = 'test', prediction_function = None, score_measure = None, lower_is_better = False):
        """
        Return a 1-liner descibing the best score.
        """
        subset, prediction_function, score_measure = self._fill_fields(subset, prediction_function, score_measure)
        best_pair = self.get_best(subset, prediction_function, score_measure, lower_is_better=lower_is_better)
        return 'Best: Epoch {} of {}, {}: {}'.format(best_pair.info.epoch, self._pairs[-1].info.epoch, score_measure, best_pair.score[subset, prediction_function, score_measure])

    def get_values(self, subset = 'test', prediction_function = None, score_measure = None):
        return [score.get_score(subset, prediction_function, score_measure) for _, score in self]

    def get_best_value(self, subset = 'test', prediction_function = None, score_measure = None, lower_is_better = False):

        subset, prediction_function, score_measure = self._fill_fields(subset, prediction_function, score_measure)
        best_pair = self.get_best(subset, prediction_function, score_measure, lower_is_better=lower_is_better)
        try:
            return best_pair.score[subset, prediction_function, score_measure]
        except KeyError as err:
            raise KeyError(str(err.message) + '.  Indices must be ({{{}}}, {{{}}}, {{{}}}'.format(self.get_data_subsets(), self.get_prediction_functions(), self.get_costs()))

    def get_best(self, subset = 'test', prediction_function = None, score_measure = None, lower_is_better = False):
        """
        Given a list of (info, score) pairs which represet the progress over training, find the best score and return it.
        :param score_info_pairs: A list<(IterationInfo, dict)> of the type returned in train_and_test_online_predictor
        :param subset: 'train' or 'test' ... which subset to use to look for the best score
        :param prediction_function: Which prediction function (if there are multiple prediction functions, otherwise leave blank)
        :param score_measure: Which score measure (if there are multiple score measures, otherwise leave blank)
        :param lower_is_better: True if a lower score is better for the chosen score_measure
        :return: A InfoScorePair object
        """
        assert len(self._pairs) > 0, "You need to have at least one score to determine the best one."
        subset, prediction_function, score_measure = self._fill_fields(subset, prediction_function, score_measure)
        best_pair = self._pairs[0]
        for info_score_pair in self._pairs[1:]:
            best_pair = info_score_pair if (info_score_pair.score[subset, prediction_function, score_measure] < best_pair.score[subset, prediction_function, score_measure]) == lower_is_better else best_pair
        return best_pair

    def _fill_fields(self, subset = None, prediction_function = None, score_measure = None):
        first_score = self._pairs[0].score
        if subset is None:
            subset = first_score.get_only_data_subset()
        else:
            assert subset in self.get_data_subsets(), 'Subset {} is not one of {}'.format(subset, self.get_data_subsets())
        if prediction_function is None:
            prediction_function = first_score.get_only_prediction_function()
        else:
            assert prediction_function in self.get_prediction_functions(), 'Prediction function {} is not one of {}'.format(prediction_function, self.get_prediction_functions())
        if score_measure is None:
            score_measure = first_score.get_only_cost()
        else:
            assert score_measure in self.get_costs(), 'Score function {} is not one of {}'.format(score_measure, self.get_costs())
        return subset, prediction_function, score_measure

    def __str__(self):
        desc = 'InfoScorePairSequence<{} epochs over {}s. '.format(self[-1].info.epoch, self[-1].info.time)
        keys = self[0].score.keys()
        for subset, pred_fcn, cost_fcn in keys:
            values = self.get_values(subset, pred_fcn, cost_fcn)
            desc+=',({}{}{}) in [{} to {}]'.format(subset, '' if pred_fcn is None else ','+str(pred_fcn), ','+str(cost_fcn), min(values), max(values))
        desc += '>'
        return desc

    def get_table(self):
        from tabulate import tabulate
        if len(self)==0:
            return repr(self) + '\n' + '<Empty>'
        else:
            rows = self[0].get_table_headers() + [pair.get_table_row() for pair in self]
            return repr(self)+'\n  '+tabulate(rows).replace('\n', '\n  ')

    def get_summary(self, subset = None, prediction_function = None, score_measure = None, lower_is_better = False, score_format='.3g'):
        subsets = self._pairs[0].score.get_data_subsets() if subset is None else [subset]
        funcs = self._pairs[0].score.get_prediction_functions() if prediction_function is None else [prediction_function]
        costs = self._pairs[0].score.get_costs() if score_measure is None else [score_measure]
        entries = []
        for subset in subsets:
            for func in funcs:
                for cost in costs:
                    title = ','.join(([subset] if len(subsets)>1 else [])+([func] if len(funcs)>1 else [])+([cost] if len(costs)>1 else []))
                    score = self.get_best_value(subset=subset, prediction_function=func, score_measure=cost, lower_is_better=lower_is_better)
                    entries.append(('{}:{:'+score_format+'}').format(title, score))
        return '; '.join(entries)

    def get_data_subsets(self):
        return self._pairs[0].score.get_data_subsets()

    def get_prediction_functions(self):
        return self._pairs[0].score.get_prediction_functions()

    def get_costs(self):
        return self._pairs[0].score.get_costs()


def plot_info_score_pairs(ispc, prediction_function = None, score_measure=None, name='', show=True):
    """
    :param info_score_pair_sequences: An InfoScorePairSequence or a list of InfoScorePair Sequences, or a dict of them.
    :return:
    """

    from matplotlib import pyplot as plt

    colour = next(plt.gca()._get_lines.prop_cycler)['color']
    epochs = [info.epoch for info, _ in ispc]
    training_curve = ispc.get_values(subset='train', prediction_function = prediction_function, score_measure=score_measure)
    test_curve = ispc.get_values(subset='test', prediction_function=prediction_function, score_measure=score_measure)
    plt.plot(epochs, training_curve, color=colour, linestyle='--', label='{}:{}'.format(name, 'training'))
    plt.plot(epochs, test_curve, color=colour, label='{}:{}'.format(name, 'test'))
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()


def plot_info_score_pairs_collection(info_score_pair_sequences, prediction_function = None, score_measure=None, show=True):

    from matplotlib import pyplot as plt
    if isinstance(info_score_pair_sequences, InfoScorePairSequence):
        info_score_pair_sequences = [info_score_pair_sequences]

    if isinstance(info_score_pair_sequences, (list, tuple)):
        info_score_pair_sequences = OrderedDict((ix, ispc) for ix, ispc in enumerate(info_score_pair_sequences))

    colours = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    for (name, ispc), colour in zip(info_score_pair_sequences.items(), colours):
        plot_info_score_pairs(ispc, prediction_function = prediction_function, score_measure=score_measure, show=False, name=name)
    if show:
        plt.show()


def _dataset_to_test_pair(dataset, include = 'train+test'):
    """
    :param dataset:
    :param include:
    :return:
    """
    assert include in ('train', 'test', 'train+test', 'training', 'training+test')
    pairs = []
    if 'train' in include:
        pairs.append(('train', (dataset.training_set.input, dataset.training_set.target)))
    if 'test' in include:
        pairs.append(('test', (dataset.test_set.input, dataset.test_set.target)))
    return pairs


def assess_prediction_functions(test_pairs, functions, costs, print_results=False, prediction_minibatches = None):
    """

    :param test_pairs: A list<pair_name, (x, y)>, where x, y are equal-length vectors representing the samples in a dataset.
        Eg. [('training', (x_train, y_train)), ('test', (x_test, y_test))]
    :param functions: A list<function_name, function> of functions for computing the forward pass.
    :param costs: A list<(cost_name, cost_function)> or dict<cost_name: cost_function> of cost functions, where cost_function has the form:
        cost = cost_fcn(guess, y), where cost is a scalar, and guess is the output of the prediction function given one
            of the inputs (x) in test_pairs.
    :param prediction_minibatches: Size of minibatches to predict in.
    :return: A ModelTestScore object
    """
    if isinstance(test_pairs, DataSet):
        test_pairs = _dataset_to_test_pair(test_pairs)
    assert isinstance(test_pairs, list)
    assert all(len(_)==2 for _ in test_pairs)
    assert all(len(pair)==2 for name, pair in test_pairs)
    if isinstance(functions, dict):
        functions = functions.items()
    if callable(functions):
        functions = [(functions.__name__ if hasattr(functions, '__name__') else None, functions)]
    else:
        assert all(callable(f) for name, f in functions)
    if callable(costs):
        costs = [(costs.__name__, costs)]
    elif isinstance(costs, string_types):
        costs = [(costs, get_evaluation_function(costs))]
    elif isinstance(costs, dict):
        costs = costs.items()
    else:
        costs = [(cost, get_evaluation_function(cost)) if isinstance(cost, string_types) else (cost.__name__, cost) if callable(cost) else cost for cost in costs]
    assert all(callable(cost) for name, cost in costs)

    results = ModelTestScore()
    for test_pair_name, (x, y) in test_pairs:
        for function_name, function in functions:
            if prediction_minibatches is None:
                predictions = function(x)
            else:
                predictions = minibatch_process(function, minibatch_size=prediction_minibatches, mb_args=(x, ))
            for cost_name, cost_function in costs:
                results[test_pair_name, function_name, cost_name] = cost_function(predictions, y)

    if print_results:
        print(results.get_table())

    return results


def print_score_results(score, info=None):
    """
    :param results: An OrderedDict in the format returned by assess_prediction_functions_on_generator.
    :return:
    """
    if info is not None:
        print('Epoch {} (after {:.3g}s)'.format(info.epoch, info.time))
    test_pair_names, function_names, cost_names = [remove_duplicates(k) for k in zip(*score.keys())]
    rows = build_table(
        lookup_fcn=lambda test_pair_name_and_function_name_, cost_name_: score[test_pair_name_and_function_name_[0], test_pair_name_and_function_name_[1], cost_name_],
        row_categories = [[test_pair_name for test_pair_name in test_pair_names], [function_name for function_name in function_names]],
        column_categories = [cost_name for cost_name in cost_names],
        row_header_labels=['Subset', 'Function'],
        clear_repeated_headers = False,
        remove_unchanging_cols=True
        )
    import tabulate
    print(tabulate.tabulate(rows))


def train_and_test_online_predictor_checkpoints(dataset, train_fcn, predict_fcn, minibatch_size, checkpoint_generator = None, n_epochs=None, test_epochs=None,
            score_measure='percent_argmax_correct', test_callback=None, training_callback = None, score_collection = None,
            test_on = 'training+test', pass_iteration_info_to_training = False, predict_in_minibatches=False):
    """
    Train an online predictor.  Return a data structure with info about the training.
    :param dataset: A DataSet object
    :param train_fcn: A function of the form train_fcn(x, y) which updates the parameters, OR
        train_fcn(x, y, iteration_info)   if pass_iteration_info_to_training==True
    :param predict_fcn: A function of the form y=predict_fcn(x) which makes a prediction giben inputs
    :param minibatch_size: Minibatch size
    :param n_epochs: Number of epoch
    :param test_epochs: Epochs to test at
    :param score_measure: String or function of the form:
        score = score_measure(guess, ground_truth)
        To be used in testing.
    :param test_callback: Function to be called after a test.  It has the form: f(info, score)
    :param training_callback: Function to be called after every training iteration.  It has the form f(info, x, y) where
    :param score_collection: If not None, a InfoScoreCollection object into which you save scores.  This allows you to
        access the scores before this function returns.
    :param test_on: What to run tests on: {'training', 'test', 'training+test'}
    :param pass_iteration_info_to_training: Pass an IterationInfo object into the training function (x, y, iter_info).
        This object contains fields (epoch, sample, time) which can be used to do things like adjust parameters on a schedule.
    :param predict_in_minibatches: Also run the prediction function in minibatches (for example to save memory)
    :return: A list<info, scores>  where...
        IterationInfo object (see artemis.ml.tools.iteration.py) with fields:
            'iteration', 'epoch', 'sample', 'time', 'test_now', 'done'
        scores is dict<(subset, prediction_function, cost_function) -> score>  where:
            subset is a string identifying the subset (eg 'train', 'test')
            prediction_function is identifies the prediction function (usually None, but can be used if you specify multiple prediction functions)
            cost_function is identifiers the cost function.
    """

    last_time = last_epoch = 0
    info_score_pairs = InfoScorePairSequence() if score_collection is None else score_collection

    for (x_mini, y_mini), info in zip_minibatch_iterate_info(dataset.training_set.xy, minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=test_epochs):
        if info.test_now:
            rate = (info.time-last_time)/(info.epoch - last_epoch) if info.epoch>0 else float('nan')
            print('Epoch {}.  Rate: {:.3g}s/epoch'.format(info.epoch, rate))
            last_epoch = info.epoch
            last_time = info.time
            test_pairs = _dataset_to_test_pair(dataset, include = test_on)
            score = assess_prediction_functions(test_pairs, functions=predict_fcn, costs=score_measure, print_results=True, prediction_minibatches=minibatch_size if predict_in_minibatches else None)
            p = InfoScorePair(info, score)
            info_score_pairs.append(p)
            # print p.get_table(remove_headers=len(info_score_pairs)>1)
            if test_callback is not None:
                test_callback(info, score)
            yield info_score_pairs
        if not info.done:
            if pass_iteration_info_to_training:
                train_fcn(x_mini, y_mini, info)
            else:
                train_fcn(x_mini, y_mini)
            if training_callback is not None:
                training_callback(info, x_mini, y_mini)
        else:
            yield info_score_pairs


def train_and_test_online_predictor(*args, **kwargs):

    for info_score_pairs in train_and_test_online_predictor_checkpoints(*args, **kwargs):
        pass
    return info_score_pairs


def get_best_score(score_info_pairs, subset = 'test', prediction_function = None, score_measure = None, lower_is_better = False):
    """
    DEPRECATED!!!

    Given a list of (info, score) pairs which represet the progress over training, find the best score and return it.
    :param score_info_pairs: A list<(IterationInfo, dict)> of the type returned in train_and_test_online_predictor
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
    # DEPRECATED!!!
    best_info, best_score = get_best_score(score_info_pairs, **best_score_kwargs)
    print_score_results(score=best_score, info=best_info)


class ParameterSchedule(object):

    def __init__(self, schedule, print_variable_name = None):
        """
        Given a schedule for a changing parameter (e.g. learning rate) get the values for this parameter at a given time.
        e.g.:
            learning_rate_scheduler = ScheduledParameter({0: 0.1, 10: 0.01, 100: 0.001}, print_variable_name='eta')
            new_learning_rate = learning_rate_scheduler.get_new_value(epoch=14)
            assert new_learning_rate == 0.01

        :param schedule: Can be:
            - A dict<epoch: value> where the epoch is a number indicating the training progress and the value
              indicates the value that the parameter should take.
            - A function which takes the epoch and returns a parameter value.
            - A number or array, in which case the value remains constant
        """
        if isinstance(schedule, (int, float, np.ndarray)):
            schedule = {0: schedule}
        if isinstance(schedule, dict):
            assert all(isinstance(num, (int, float)) for num in schedule.keys())
            self._reverse_sorted_schedule_checkpoints = sorted(schedule.keys(), reverse=True)
        else:
            assert callable(schedule)
        self.schedule = schedule
        self.print_variable_name = print_variable_name
        self.last_value = None  # Just used for print statements.

    def get_new_value(self, epoch):
        if isinstance(self.schedule, dict):
            new_value = self.schedule[next(e for e in self._reverse_sorted_schedule_checkpoints if e <= epoch)]
        else:
            new_value = self.schedule(epoch)
        if self.last_value != new_value and self.print_variable_name is not None:
            print('Epoch {}: {} = {}'.format(epoch, self.print_variable_name, new_value))
            self.last_value = new_value
        return new_value
