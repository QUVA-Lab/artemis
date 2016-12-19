import atexit
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from artemis.experiments.experiment_record import run_experiment, show_experiment, \
    get_latest_experiment_identifier, get_experiment_info, load_experiment_record, ExperimentRecord, record_experiment, \
    delete_experiment_with_id, get_current_experiment_dir, experiment_function, open_in_experiment_dir
from artemis.experiments.deprecated import register_experiment, start_experiment, end_current_experiment
from artemis.general.test_mode import set_test_mode

__author__ = 'peter'

"""
The experiment interface.  Currently, this can be used in many ways.  We'd like to
converge on just using the interface demonstrated in test_get_latest_identifier.  So
use that if you're looking for an example.
"""


@experiment_function
def experiment_test_function():

    with warnings.catch_warnings():  # This is just to stop that stupid matplotlib warning from screwing up our logs.
        warnings.simplefilter('ignore')

        print 'aaa'
        plt.figure('sensible defaults')
        dat = np.random.randn(4, 5)
        plt.subplot(211)
        plt.imshow(dat)
        plt.subplot(212)
        plt.imshow(dat, interpolation = 'nearest', cmap = 'gray')
        plt.show()
        print 'bbb'
        plt.figure()
        plt.plot(np.random.randn(10))
        plt.show()


def assert_experiment_record_is_correct(exp_rec, show_figures=False):

    assert exp_rec.get_log() == 'aaa\nbbb\n'
    assert len(exp_rec.get_figure_locs()) == 2

    # Now assert that you can load an experiment from file and again display the figures.
    exp_dir = exp_rec.get_dir()
    exp_rec_copy = ExperimentRecord(exp_dir)
    assert exp_rec_copy.get_log() == 'aaa\nbbb\n'
    if show_figures:
        exp_rec_copy.show_figures()
    assert len(exp_rec_copy.get_figure_locs()) == 2


def test_experiment_with():
    """
    DEPRECATED INTERFACE

    This syntax uses the record_experiment function directly instead of hiding it.
    """

    delete_experiment_with_id('test_exp')

    with record_experiment(identifier = 'test_exp', print_to_console=True) as exp_rec:
        experiment_test_function()

    assert_experiment_record_is_correct(exp_rec, show_figures=False)


def test_start_experiment():
    """
    DEPRECATED INTERFACE

    An alternative syntax to the with statement - less tidy but possibly better
    for notebooks and such because it avoids you having to indent all code in the
    experiment.
    """

    record = start_experiment('start_stop_test')
    experiment_test_function()
    end_current_experiment()
    assert_experiment_record_is_correct(record, show_figures=False)
    record.delete()


def test_run_and_show():
    """

    This is nice because it no longer required that an experiment be run and shown in a
    single session - each experiment just has a unique identifier that can be used to show
    its results whenevs.
    """
    experiment_record = experiment_test_function.run(keep_record=True)
    assert_experiment_record_is_correct(experiment_record, show_figures=False)
    show_experiment(experiment_record.get_identifier())
    # Delay cleanup otherwise the show complains that file does not exist due to race condition.
    atexit.register(experiment_record.delete)


def test_get_latest():
    record_1 = experiment_test_function.run(keep_record=True)
    time.sleep(0.01)
    record_2 = experiment_test_function.run(keep_record=True)
    identifier = get_latest_experiment_identifier('experiment_test_function')
    assert identifier == record_2.get_identifier()
    atexit.register(record_1.delete)
    atexit.register(record_2.delete)


def test_get_latest_identifier():

    exp_rec = experiment_test_function.run(keep_record=True)
    print get_experiment_info('experiment_test_function')
    assert_experiment_record_is_correct(exp_rec, show_figures=False)
    last_experiment_identifier = get_latest_experiment_identifier(name='experiment_test_function')
    assert last_experiment_identifier is not None, 'Experiment was run, this should not be none'
    same_exp_rec = load_experiment_record(last_experiment_identifier)
    assert_experiment_record_is_correct(same_exp_rec, show_figures=False)
    same_exp_rec.delete()


def test_accessing_experiment_dir():

    @experiment_function
    def access_dir_test():
        print '123'
        print 'abc'
        dir = get_current_experiment_dir()
        with open_in_experiment_dir('my_test_file.txt', 'w') as f:
            f.write('Experiment Directory is: {}'.format(dir))

    record = access_dir_test.run(keep_record=True)

    filepaths = record.list_files()

    assert 'my_test_file.txt' in filepaths

    with record.open_file('my_test_file.txt') as f:
        assert f.read() == 'Experiment Directory is: {}'.format(record.get_dir())

    with record.open_file('output.txt') as f:
        assert f.read() == '123\nabc\n'


if __name__ == '__main__':

    set_test_mode(True)
    test_get_latest_identifier()
    test_get_latest()
    test_run_and_show()
    test_experiment_with()
    test_start_experiment()
    test_accessing_experiment_dir()
