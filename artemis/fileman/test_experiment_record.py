import time
from general.test_mode import set_test_mode
import os
import pickle
from fileman.experiment_record import ExperimentRecord, start_experiment, run_experiment, show_experiment, \
    get_latest_experiment_identifier, get_or_run_notebook_experiment, get_local_experiment_path, register_experiment, \
    get_experiment_info, load_experiment
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'peter'

"""
The experiment interface.  Currently, this can be used in many ways.  We'd like to
converge on just using the interface demonstrated in test_experiment_interface.  So
use that if you're looking for an example.
"""


def _run_experiment():

    print 'aaa'
    plt.figure('sensible defaults')
    dat = np.random.randn(4, 5)
    plt.subplot(211)
    plt.imshow(dat)
    plt.subplot(212)
    plt.imshow(dat, interpolation = 'nearest', cmap = 'gray')
    plt.show()
    print 'bbb'
    plt.plot(np.random.randn(10))
    plt.show()


def test_experiment_with():

    with ExperimentRecord(filename = 'test_exp', save_result=True, print_to_console=True) as exp_1:
        _run_experiment()

    assert exp_1.get_logs() == 'aaa\nbbb\n'
    figs = exp_1.show_figures()
    assert len(exp_1.get_figure_locs()) == 2

    # Now assert that you can load an experiment from file and again display the figures.
    exp_file = exp_1.get_file_path()
    with open(exp_file) as f:
        exp_1_copy = pickle.load(f)

    assert exp_1_copy.get_logs() == 'aaa\nbbb\n'
    exp_1_copy.show_figures()
    assert len(exp_1.get_figure_locs()) == 2


def test_start_experiment():
    """
    An alternative syntax to the with statement - less tidy but possibly better
    for notebooks and such because it avoids you having to indent all code in the
    experiment.
    """

    exp = start_experiment(save_result = False)
    _run_experiment()
    exp.end_and_show()
    assert len(exp.get_figure_locs()) == 2


def test_run_and_show():
    """
    This is nice because it no longer required that an experiment be run and shown in a
    single session - each experiment just has a unique identifier that can be used to show
    its results whenevs.
    """
    experiment = run_experiment('the_exp', exp_dict = {'the_exp': _run_experiment}, save_result = True)
    show_experiment(experiment.get_identifier())
    os.remove(experiment.get_file_path())


def test_get_latest():

    experiment_1 = run_experiment('test_get_latest', exp_dict = {'test_get_latest': _run_experiment}, save_result = True)
    time.sleep(0.01)
    experiment_2 = run_experiment('test_get_latest', exp_dict = {'test_get_latest': _run_experiment}, save_result = True)
    identifier = get_latest_experiment_identifier('test_get_latest')
    assert identifier == experiment_2.get_identifier()
    os.remove(experiment_1.get_file_path())
    os.remove(experiment_2.get_file_path())


def test_get_or_run_experiment():

    name = 'test_get_or_run'

    while get_latest_experiment_identifier(name) is not None:
        ident = get_latest_experiment_identifier(name)
        os.remove(get_local_experiment_path(ident))

    counter = [0]

    def add_one():
        counter[0] += 1

    experiment_1 = get_or_run_notebook_experiment('test_get_or_run', exp_dict = {'test_get_or_run': add_one}, save_result = True)
    time.sleep(0.001)
    experiment_2 = get_or_run_notebook_experiment('test_get_or_run', exp_dict = {'test_get_or_run': add_one}, save_result = True)
    assert counter[0] == 1  # 0 in case this experiment existed before
    assert experiment_1.get_file_path() == experiment_2.get_file_path()
    os.remove(experiment_1.get_file_path())


def test_experiment_interface():

    register_experiment(
        name = 'my_test_experiment',
        function=_run_experiment,
        description="See if this thing works",
        conclusion="It does."
        )

    exp_rec = run_experiment('my_test_experiment', save_result=True)
    print get_experiment_info('my_test_experiment')
    assert exp_rec.get_logs() == 'aaa\nbbb\n'

    same_exp_rec = load_experiment(get_latest_experiment_identifier(name = 'my_test_experiment'))
    assert same_exp_rec.get_logs() == 'aaa\nbbb\n'


if __name__ == '__main__':

    set_test_mode(True)

    test_experiment_interface()
    test_get_or_run_experiment()
    test_get_latest()
    test_run_and_show()
    test_experiment_with()
    test_start_experiment()
