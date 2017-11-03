from artemis.general.duck import Duck
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.checkpoint_counter import Checkpoints
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
        save_train_return = False,
        measures = None,
        iterations_to_end = False
        ):
    """
    :param f_train:
    :param f_predict:
    :param losses:
    :param training_data_gen:
    :param test_data_gen_constructors:
    :param samples_count:
    :param n_training_iters:
    :param n_test_iters:
    :param test_checkpoints:
    :param collapse_loss:
    :param progress_update_period:
    :param in_test_callback:
    :param post_test_callback:
    :param post_train_callback:
    :param save_train_return:
    :param measures:
    :return: If yield_array_data is False, an ArrayStruct with fields:
        'training'                          Results recorded during training callbacks
            training_iter                   The iteration of the training callback
        'testing'                           Results recorded during tests
            test_iter                       The index of the test
                'iter'                      The number of training iterations finished at the time that the test is run
                'time'                      The time at which the test is run
                'samples'                   The number of samples seen so far
                'results'                   A structure containing results
                    subset_name             The name of the testing subset e.g. 'train', 'test' (you give subset names in test_data_gen_constructors)
                        'losses'
                            loss_name       The name of the loss function (you provide this in losses)
                            n_tests         The numner of tests that were run for this subset
                        'time'              The time, in seconds, that it took to test on this subset.


        Otherwise, if true, a structure the same object but training_iter and test_iter pushed to the leaf-position
    """

    if measures is None:
        measures = Duck()
    if 'training' not in measures:
        measures['training'] = Duck()
    if 'testing' not in measures:
        measures['testing'] = Duck()

    is_test_time = Checkpoints(test_checkpoints) if not isinstance(test_checkpoints, Checkpoints) else test_checkpoints
    pi = ProgressIndicator(n_training_iters, "Training", update_every=progress_update_period)

    for inputs, targets in training_data_gen:
        if is_test_time():

            this_test_measures = measures['testing'].open(next)
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
            if iterations_to_end:
                measures_to_yield = measures.arrayify_axis(axis=1, subkeys='training')
                measures_to_yield = measures_to_yield.arrayify_axis(axis=1, subkeys='testing', inplace=True)
                yield measures_to_yield.to_struct()
            else:
                yield measures

        train_return = f_train(inputs, targets)
        pi.print_update()
        if save_train_return:
            measures['training', 'returns'] = train_return
        if post_train_callback:
            return_val = post_train_callback(inputs=inputs, targets=targets, iter=pi.get_iterations())
            if return_val is not None:
                measures['training', 'callback', next, ...] = return_val


def do_test(test_subset_generators, f_predict, loss_dict, n_test_iters, collapse_loss = 'mean', in_test_callback = None):

    if callable(loss_dict):
        loss_dict = dict(loss=loss_dict)

    these_test_results = Duck()
    losses = {}
    pi = ProgressIndicator(n_test_iters, "Testing")
    for subset_name, subset_generator in test_subset_generators.items():
        start_time = time.time()
        losses[subset_name] = []
        n_tests = 0
        for inputs, targets in subset_generator:
            n_tests+=1
            outputs = f_predict(inputs)
            for loss_name, f_loss in loss_dict.items():
                these_test_results[subset_name, 'losses', loss_name, next] = f_loss(outputs, targets)
            pi.print_update()
            if in_test_callback is not None:
                in_test_callback(inputs=inputs, targets=targets, outputs=outputs)

        assert n_tests>0, "It appears that subset '{}' had no tests!".format(subset_name)
        these_test_results[subset_name, 'n_tests'] = n_tests
        if collapse_loss is not None:
            collapse_func = {'mean': np.mean}[collapse_loss]
            for loss_name, f_loss in loss_dict.items():
                these_test_results[subset_name, 'losses', loss_name] = collapse_func(these_test_results[subset_name, 'losses', loss_name])
        these_test_results[subset_name, 'time'] = time.time() - start_time
    return these_test_results
