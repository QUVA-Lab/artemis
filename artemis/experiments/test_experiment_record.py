import itertools
import time
import warnings
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pytest
from six.moves import xrange

from artemis.experiments.decorators import experiment_function, experiment_root
from artemis.experiments.deprecated import start_experiment, end_current_experiment
from artemis.experiments.experiment_management import run_multiple_experiments
from artemis.experiments.experiment_record import \
    load_experiment_record, ExperimentRecord, record_experiment, \
    delete_experiment_with_id, get_current_record_dir, open_in_record_dir, \
    ExpStatusOptions, clear_experiment_records, get_current_experiment_id, get_current_experiment_record, \
    get_current_record_id, has_experiment_record, experiment_id_to_record_ids
from artemis.experiments.experiments import get_experiment_info, load_experiment, experiment_testing_context, \
    clear_all_experiments
from artemis.general.test_mode import set_test_mode

__author__ = 'peter'

"""
The experiment interface.  Currently, this can be used in many ways.  We'd like to
converge on just using the interface demonstrated in test_experiment_api.  So
use that if you're looking for an example.
"""


@experiment_function
def experiment_test_function(seed=1234):

    rng = np.random.RandomState(seed)
    with warnings.catch_warnings():  # This is just to stop that stupid matplotlib warning from screwing up our logs.
        warnings.simplefilter('ignore')

        print('aaa')
        plt.figure('sensible defaults')
        dat = rng.randn(4, 5)
        plt.subplot(211)
        plt.imshow(dat)
        plt.subplot(212)
        plt.imshow(dat, interpolation = 'nearest', cmap = 'gray')
        plt.show()
        print('bbb')
        plt.figure()
        plt.plot(rng.randn(10))
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

    delete_experiment_with_id('start_stop_test')
    with experiment_testing_context():
        record = start_experiment('start_stop_test')
        experiment_test_function()
        end_current_experiment()
        assert_experiment_record_is_correct(record, show_figures=False)


def test_get_latest():
    with experiment_testing_context():
        record_1 = experiment_test_function.run()
        time.sleep(0.01)
        record_2 = experiment_test_function.run()
        identifier = load_experiment('experiment_test_function').get_latest_record().get_id()
        assert identifier == record_2.get_id()


def test_get_latest_identifier():

    with experiment_testing_context():
        exp_rec = experiment_test_function.run()
        print(get_experiment_info('experiment_test_function'))
        assert_experiment_record_is_correct(exp_rec)
        last_experiment_identifier = load_experiment('experiment_test_function').get_latest_record().get_id()
        assert last_experiment_identifier is not None, 'Experiment was run, this should not be none'
        same_exp_rec = load_experiment_record(last_experiment_identifier)
        assert_experiment_record_is_correct(same_exp_rec)


def test_accessing_experiment_dir():

    with experiment_testing_context():

        @experiment_function
        def access_dir_test():
            print('123')
            print('abc')
            dir = get_current_record_dir()
            with open_in_record_dir('my_test_file.txt', 'w') as f:
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
    print(c)
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
        print(c)
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
        assert e2.get_id() == 'add_some_numbers.b=4'

        # Create array of variants
        e_list = [add_some_numbers.add_variant(b=i) for i in xrange(5, 8)]
        assert [e.get_id() for e in e_list] == ['add_some_numbers.b=5', 'add_some_numbers.b=6', 'add_some_numbers.b=7']
        assert [e.run().get_result()==j for e, j in zip(e_list, range(6, 11))]

        # Create grid of variants
        e_grid = [add_some_numbers.add_variant(a=a, b=b) for a, b in itertools.product([2, 3], [4, 5, 6])]
        assert [e.get_id() for e in e_grid] == ['add_some_numbers.a=2,b=4', 'add_some_numbers.a=2,b=5', 'add_some_numbers.a=2,b=6',
                                                  'add_some_numbers.a=3,b=4', 'add_some_numbers.a=3,b=5', 'add_some_numbers.a=3,b=6']
        assert add_some_numbers.get_variant(a=2, b=4).run().get_result()==6
        assert add_some_numbers.get_variant(a=3, b=5).run().get_result()==8

        experiments = add_some_numbers.get_all_variants(include_roots=True, include_self=True)
        assert len(experiments)==13


@experiment_function
def my_api_test(a=1, b=3):
    print('aaa')
    return a*b

my_api_test.add_variant('a2b2', a=2, b=2)
my_api_test.add_variant('a3b2', a=3, b=2)


def test_experiment_api(try_browse=False):

    with experiment_testing_context():
        my_api_test.get_variant('a2b2').run()
        record = my_api_test.get_variant('a2b2').get_latest_record()

        assert record.get_log() == 'aaa\n'
        assert record.get_result() == 4
        assert record.get_args() == OrderedDict([('a', 2), ('b', 2)])
        assert record.get_status() == ExpStatusOptions.FINISHED

    if try_browse:
        my_api_test.browse()


def test_figure_saving(show_them = False):

    with experiment_testing_context():
        record = experiment_test_function.run()

    plt.close('all')  # Close all figures
    figs = record.load_figures()
    assert len(figs)==2
    if show_them:
        plt.show()


def test_get_variant_records_and_delete():

    with experiment_testing_context():

        for record in my_api_test.get_variant_records(flat=True):
            record.delete()

        assert len(my_api_test.get_variant_records(flat=True))==0

        my_api_test.run()
        my_api_test.get_variant('a2b2').run()

        assert len(my_api_test.get_variant_records(flat=True))==2

        for record in my_api_test.get_variant_records(flat=True):
            record.delete()

        assert len(my_api_test.get_variant_records(flat=True))==0


def test_experiments_play_well_with_debug():

    with experiment_testing_context():

        @experiment_function
        def my_simple_test():
            plt.show._needmain=False  # pyplot does this internally whenever a breakpoint is reached.
            return 1

        my_simple_test.run()


def test_run_multiple_experiments():

    with experiment_testing_context():

        experiments = my_api_test.get_all_variants()
        assert len(experiments)==3

        records = run_multiple_experiments(experiments)
        assert [record.get_result() for record in records] == [3, 4, 6]

        records = run_multiple_experiments(experiments, parallel=True)
        assert [record.get_result() for record in records] == [3, 4, 6]


def test_parallel_run_errors():

    with experiment_testing_context():

        @experiment_function
        def my_error_causing_test(a=1):
            raise Exception('nononono')

        my_error_causing_test.add_variant(a=2)

        variants = my_error_causing_test.get_all_variants()

        assert len(variants)==2

        run_multiple_experiments(variants, parallel=True, raise_exceptions=False)

        with pytest.raises(Exception) as err:
            run_multiple_experiments(variants, parallel=True, raise_exceptions=True)
        print("^^^ Dont't worry, the above is not actually an error, we were just asserting that we caught the error.")

        assert str(err.value) == 'nononono'


def test_invalid_arg_detection():
    """
    Check that we notice when an experiment is redefined with new args.
    """

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_experiment_gfdsbhtds(a=1, b=[2, 3.], c={'a': 5, 'b': [6, 7]}):
            return a+1

        rec = my_experiment_gfdsbhtds.run()

        assert rec.args_valid()
        clear_all_experiments()

        @experiment_function
        def my_experiment_gfdsbhtds(a=1, b=[2, 3.], c={'a': 5, 'b': [6, 7]}):
            return a+1

        assert rec.args_valid()  # Assert that the args still match
        clear_all_experiments()

        @experiment_function
        def my_experiment_gfdsbhtds(a=1, b=[2, 3.], c={'a': 5, 'b': [6, 8]}):  # CHANGE IN ARGS!
            return a+1

        assert not rec.args_valid()


def test_invalid_arg_detection_2():
    """
    Check that we notice when an experiment is redefined with new args.
    """

    with experiment_testing_context(new_experiment_lib=True):

        a = {"a%s"%i for i in range(100)}

        @experiment_function
        def my_experiment_gfdsbhtds(a=a):
            return None

        rec = my_experiment_gfdsbhtds.run()

        assert rec.args_valid() is True
        clear_all_experiments()

        @experiment_function
        def my_experiment_gfdsbhtds(a=a):
            return None

        assert rec.args_valid() is True  # Assert that the args still match


def test_experiment_errors():

    with experiment_testing_context(new_experiment_lib=True):

        class MyManualException(Exception):
            pass

        @experiment_function
        def my_experiment_fdsgbdn():
            raise MyManualException()

        with pytest.raises(MyManualException):
            my_experiment_fdsgbdn.run()
        with pytest.raises(MyManualException):
            my_experiment_fdsgbdn.run()

        assert my_experiment_fdsgbdn.get_latest_record().get_status() == ExpStatusOptions.ERROR

        # Previously this caused an error because simple context managers didn't catch errors


def test_experiment_corrupt_detection():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_experiment_bfdsssdvs(a=1):
            return a+2

        r = my_experiment_bfdsssdvs.run()
        assert r.get_status() == ExpStatusOptions.FINISHED
        os.remove(os.path.join(r.get_dir(), 'info.pkl'))  # Manually remove the info file
        r = my_experiment_bfdsssdvs.get_latest_record()
        assert r.get_status() == ExpStatusOptions.CORRUPT


def test_current_experiment_access_functions():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_experiment_dfgsdgfdaf(a=1):

            assert a==4, 'Only meant to run aaa variant.'
            experiment_id = get_current_experiment_id()

            rec = get_current_experiment_record()

            record_id = get_current_record_id()

            assert record_id.endswith('-'+experiment_id)

            assert experiment_id == 'my_experiment_dfgsdgfdaf.aaa'
            loc = get_current_record_dir()

            _, record_dir = os.path.split(loc)

            assert record_dir == record_id
            assert os.path.isdir(loc)
            assert loc.endswith(record_id)

            with open_in_record_dir('somefile.pkl', 'wb') as f:
                pickle.dump([1, 2, 3], f, protocol=pickle.HIGHEST_PROTOCOL)

            assert os.path.exists(os.path.join(loc, 'somefile.pkl'))

            with open_in_record_dir('somefile.pkl', 'rb') as f:
                assert pickle.load(f) == [1, 2, 3]

            exp = rec.get_experiment()
            assert exp.get_id() == experiment_id
            assert exp.get_args() == rec.get_args() == OrderedDict([('a', 4)])
            assert rec.get_dir() == loc
            assert has_experiment_record(experiment_id)
            assert record_id in experiment_id_to_record_ids(experiment_id)
            return a+2

        v = my_experiment_dfgsdgfdaf.add_variant('aaa', a=4)

        v.run()


def test_generator_experiment():

    with experiment_testing_context(new_experiment_lib=True):
        @experiment_root
        def my_generator_exp(n_steps, poison_4 = False):
            for i in range(n_steps):
                if poison_4 and i==4:
                    raise Exception('Unlucky Number!')
                yield i

        X1 = my_generator_exp.add_variant(n_steps=5)
        X2 = my_generator_exp.add_variant(n_steps=5, poison_4 = True)

        rec1 = X1.run()
        rec2 = X2.run(raise_exceptions = False)

        assert rec1.get_result() == 4
        assert rec2.get_result() == 3


if __name__ == '__main__':

    set_test_mode(True)
    test_get_latest_identifier()
    test_get_latest()
    test_experiment_with()
    test_start_experiment()
    test_accessing_experiment_dir()
    test_saving_result()
    test_variants()
    test_experiment_api(try_browse=False)
    test_figure_saving(show_them=False)
    test_get_variant_records_and_delete()
    test_experiments_play_well_with_debug()
    test_run_multiple_experiments()
    test_parallel_run_errors()
    test_invalid_arg_detection()
    test_invalid_arg_detection_2()
    test_experiment_errors()
    test_experiment_corrupt_detection()
    test_current_experiment_access_functions()
    test_generator_experiment()
