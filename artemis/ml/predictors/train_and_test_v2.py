from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.nested_structures import SequentialStructBuilder
import numpy as np
import time


def train_and_test_predictor(
        f_train,
        f_predict,
        losses,
        training_data_gen,
        test_data_gen_constructors,
        n_training_iters = None,
        n_test_iters = None,
        test_checkpoints = ('lin', 1000),
        collapse_loss = 'mean',
        progress_update_period = '5s',
        in_test_callback = None,
        post_test_callback = None,
        post_train_callback = None,
        measures = None
        ):

    if measures is None:
        measures = SequentialStructBuilder()
    is_test_time = Checkpoints(test_checkpoints) if not isinstance(test_checkpoints, Checkpoints) else test_checkpoints
    pi = ProgressIndicator(n_training_iters, "Training", update_every=progress_update_period)
    for inputs, targets in training_data_gen:
        if is_test_time():
            this_test_measures = measures['test'].open_next()
            this_test_measures['iter'] = pi.get_iterations()
            this_test_measures['time'] = pi.get_elapsed()
            this_test_measures['results'] = do_test(
                test_subset_generators={subset_name: constructor() for subset_name, constructor in test_data_gen_constructors.items()},
                f_predict=f_predict,
                loss_dict=losses,
                n_test_iters=n_test_iters,
                collapse_loss = collapse_loss,
                in_test_callback = in_test_callback,
                )
            if post_test_callback is not None:
                post_test_callback(this_test_measures)
            yield measures.to_struct_arrays()

        f_train(inputs, targets)
        pi.print_update()
        if post_train_callback:
            post_train_callback(inputs=inputs, targets=targets, iter=pi.get_iterations(), training_measures=measures['training'])


def do_test(test_subset_generators, f_predict, loss_dict, n_test_iters, collapse_loss = 'mean', in_test_callback = None):

    if callable(loss_dict):
        loss_dict = dict(loss=loss_dict)

    these_test_results = SequentialStructBuilder()
    losses = {}
    pi = ProgressIndicator(n_test_iters, "Testing")
    for subset_name, subset_generator in test_subset_generators.items():
        start_time = time.time()
        test_data_generator = subset_generator
        losses[subset_name] = []
        for inputs, targets in test_data_generator:
            outputs = f_predict(inputs)
            for loss_name, f_loss in loss_dict.items():
                these_test_results[subset_name][loss_name].next = f_loss(outputs, targets)
            pi.print_update()
            if in_test_callback is not None:
                in_test_callback(inputs=inputs, targets=targets, outputs=outputs)

        n_tests = len(these_test_results[subset_name][loss_name])
        assert n_tests>0, "It appears that subset '{}' had no tests!".format(subset_name)
        these_test_results[subset_name]['n_tests'] = n_tests
        if collapse_loss is not None:
            collapse_func = {'mean': np.mean}[collapse_loss]
            for loss_name, f_loss in loss_dict.items():
                these_test_results[subset_name][loss_name] = collapse_func(these_test_results[subset_name][loss_name].to_array())
        these_test_results[subset_name]['time'] = time.time() - start_time
    return these_test_results.to_struct_arrays()


#
# def training_results_oneline_summary(results):
#
#     for subset in results['test'].keys():
#         for loss_name, loss_value in results['test'][subset].items():
#
#
#     train_overlaps = result['train']['mean_overlap']
#     test_overlaps = result['test']['mean_overlap']
#     # losses = result['test']['mean_loss']
#     # return '{} Tests.  Mean Overlap: {:.3g} -> {:.3g}, Loss {:.3g} -> {:.3g}'.format(len(overlaps), overlaps[0], overlaps[-1], losses[0], losses[-1])
#     return '{} Tests.  Overlaps: Test: {:.3g} -> {:.3g}, Train: {:.3g} -> {:.3g}'.format(len(test_overlaps), test_overlaps[0], test_overlaps[-1], train_overlaps[0], train_overlaps[-1])
#
#
# # def plot_results()