import atexit
import time
import warnings

import itertools
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from artemis.experiments.experiment_record import run_experiment, show_experiment, \
    get_latest_experiment_identifier, get_experiment_info, load_experiment_record, ExperimentRecord, record_experiment, \
    delete_experiment_with_id, get_current_experiment_dir, experiment_function, open_in_experiment_dir, \
    get_all_experiment_ids, clear_experiments, experiment_testing_context
from artemis.experiments.deprecated import register_experiment, start_experiment, end_current_experiment
from artemis.general.test_mode import set_test_mode, UseTestContext

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

    with experiment_testing_context():
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

    with experiment_testing_context():
        record = start_experiment('start_stop_test')
        experiment_test_function()
        end_current_experiment()
        assert_experiment_record_is_correct(record, show_figures=False)


def test_run_and_show():
    """

    This is nice because it no longer required that an experiment be run and shown in a
    single session - each experiment just has a unique identifier that can be used to show
    its results whenevs.
    """
    with experiment_testing_context():
        experiment_record = experiment_test_function.run()
        assert_experiment_record_is_correct(experiment_record, show_figures=False)
        show_experiment(experiment_record.get_identifier())


def test_get_latest():
    with experiment_testing_context():
        record_1 = experiment_test_function.run()
        time.sleep(0.01)
        record_2 = experiment_test_function.run()
        identifier = get_latest_experiment_identifier('experiment_test_function')
        assert identifier == record_2.get_identifier()


def test_get_latest_identifier():

    with experiment_testing_context():
        exp_rec = experiment_test_function.run()
        print get_experiment_info('experiment_test_function')
        assert_experiment_record_is_correct(exp_rec)
        last_experiment_identifier = get_latest_experiment_identifier(name='experiment_test_function')
        assert last_experiment_identifier is not None, 'Experiment was run, this should not be none'
        same_exp_rec = load_experiment_record(last_experiment_identifier)
        assert_experiment_record_is_correct(same_exp_rec)


def test_accessing_experiment_dir():

    with experiment_testing_context():

        @experiment_function
        def access_dir_test():
            print '123'
            print 'abc'
            dir = get_current_experiment_dir()
            with open_in_experiment_dir('my_test_file.txt', 'w') as f:
                f.write('Experiment Directory is: {}'.format(dir))

        record = access_dir_test.run()

        filepaths = record.list_files()

        assert 'my_test_file.txt' in filepaths

        with record.open_file('my_test_file.txt') as f:
            assert f.read() == 'Experiment Directory is: {}'.format(record.get_dir())

        with record.open_file('output.txt') as f:
            assert f.read() == '123\nabc\n'


@experiment_function
def add_some_numbers_test_experiment(a=1, b=1):
    c = a + b
    print c
    return c


def test_saving_result():
    # Run root experiment
    with experiment_testing_context():
        rec = add_some_numbers_test_experiment.run()
        assert rec.get_result() == 2


def test_variants():

    @experiment_function
    def add_some_numbers(a=1, b=1):
        c=a+b
        print c
        return c

    with experiment_testing_context():

        # Create a named variant
        e1=add_some_numbers.add_variant('b is 3', b=3)
        assert e1.run().get_result()==4

        # Creata a sub-variant
        e11 = e1.add_variant('a is 2', a=2)
        assert e11.run().get_result() == 5

        # Create unnamed variant
        e2=add_some_numbers.add_variant(b=4)
        assert e2.run().get_result()==5
        assert e2.get_name() == 'add_some_numbers.b=4'

        # Create array of variants
        e_list = [add_some_numbers.add_variant(b=i) for i in xrange(5, 8)]
        assert [e.get_name() for e in e_list] == ['add_some_numbers.b=5', 'add_some_numbers.b=6', 'add_some_numbers.b=7']
        assert [e.run().get_result()==j for e, j in zip(e_list, range(6, 11))]

        # Create grid of variants
        e_grid = [add_some_numbers.add_variant(a=a, b=b) for a, b in itertools.product([2, 3], [4, 5, 6])]
        assert [e.get_name() for e in e_grid] == ['add_some_numbers.a=2,b=4', 'add_some_numbers.a=2,b=5', 'add_some_numbers.a=2,b=6',
                                                  'add_some_numbers.a=3,b=4', 'add_some_numbers.a=3,b=5', 'add_some_numbers.a=3,b=6']
        assert add_some_numbers.get_unnamed_variant(a=2, b=4).run().get_result()==6
        assert add_some_numbers.get_unnamed_variant(a=3, b=5).run().get_result()==8

        experiments = add_some_numbers.get_all_variants(include_roots=True)
        assert len(experiments)==13


if __name__ == '__main__':
    set_test_mode(True)
    test_get_latest_identifier()
    test_get_latest()
    test_run_and_show()
    test_experiment_with()
    test_start_experiment()
    test_accessing_experiment_dir()
    test_saving_result()
    test_variants()
