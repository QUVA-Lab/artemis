import atexit
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from artemis.experiments.experiment_record import run_experiment, show_experiment, \
    get_latest_experiment_identifier, get_experiment_info, load_experiment, ExperimentRecord, record_experiment, \
    delete_experiment_with_id
from artemis.experiments.deprecated import register_experiment, start_experiment, end_current_experiment
from artemis.general.test_mode import set_test_mode

__author__ = 'peter'

"""
The experiment interface.  Currently, this can be used in many ways.  We'd like to
converge on just using the interface demonstrated in test_experiment_interface.  So
use that if you're looking for an example.
"""


def _run_experiment():

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


register_experiment(
    name = 'test_experiment',
    description = "Testing the experiment framework",
    function = _run_experiment,
    conclusion = "Nothing to mention"
    )


def test_experiment_with():

    delete_experiment_with_id('test_exp')

    with record_experiment(identifier = 'test_exp', print_to_console=True) as exp_rec:
        _run_experiment()

    assert exp_rec.get_log() == 'aaa\nbbb\n'
    figs = exp_rec.show_figures()
    assert len(exp_rec.get_figure_locs()) == 2

    # Now assert that you can load an experiment from file and again display the figures.
    exp_dir = exp_rec.get_dir()
    exp_rec_copy = ExperimentRecord(exp_dir)
    assert exp_rec_copy.get_log() == 'aaa\nbbb\n'
    exp_rec_copy.show_figures()
    assert len(exp_rec_copy.get_figure_locs()) == 2


def test_start_experiment():
    """
    An alternative syntax to the with statement - less tidy but possibly better
    for notebooks and such because it avoids you having to indent all code in the
    experiment.
    """

    record = start_experiment('start_stop_test')
    _run_experiment()
    end_current_experiment()
    assert len(record.get_figure_locs()) == 2
    record.delete()


def test_run_and_show():
    """
    This is nice because it no longer required that an experiment be run and shown in a
    single session - each experiment just has a unique identifier that can be used to show
    its results whenevs.
    """
    experiment_record = run_experiment('test_experiment', keep_record = True)
    show_experiment(experiment_record.get_identifier())

    # Delay cleanup otherwise the show complains that file does not exist due to race condition.
    atexit.register(lambda: shutil.rmtree(experiment_record.get_dir()))


def test_get_latest():
    experiment_1 = run_experiment('test_experiment', keep_record = True)
    time.sleep(0.01)
    experiment_2 = run_experiment('test_experiment', keep_record = True)
    identifier = get_latest_experiment_identifier('test_experiment')
    assert identifier == experiment_2.get_identifier()

    atexit.register(lambda: shutil.rmtree(experiment_1.get_dir()))
    atexit.register(lambda: shutil.rmtree(experiment_2.get_dir()))


def test_experiment_interface():

    register_experiment(
        name = 'my_test_experiment',
        function=_run_experiment,
        description="See if this thing works",
        conclusion="It does."
        )

    exp_rec = run_experiment('my_test_experiment', keep_record = True)
    print get_experiment_info('my_test_experiment')
    assert exp_rec.get_log() == 'aaa\nbbb\n'
    same_exp_rec = load_experiment(get_latest_experiment_identifier(name = 'my_test_experiment'))
    assert same_exp_rec.get_log() == 'aaa\nbbb\n'
    same_exp_rec.delete()


if __name__ == '__main__':

    set_test_mode(True)
    test_experiment_interface()
    test_get_latest()
    test_run_and_show()
    test_experiment_with()
    test_start_experiment()
