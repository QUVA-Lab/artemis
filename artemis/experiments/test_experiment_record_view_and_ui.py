from artemis.experiments.decorators import ExperimentFunction
from artemis.experiments.experiment_record_view import display_experiment_record, compare_experiment_results, \
    get_oneline_result_string, print_experiment_record_argtable, show_experiment_records
from artemis.experiments.experiments import experiment_testing_context
from artemis.general.display import CaptureStdOut


def display_it(result):
    print str(result) + 'aaa'


def one_liner(result):
    return str(result) + 'bbb'


def compare_them(results):
    print ', '.join('{}: {}'.format(k, results[k]) for k in sorted(results.keys()))


@ExperimentFunction(display_function=display_it, one_liner_function=one_liner, comparison_function=compare_them)
def my_xxxyyy_test_experiment(a=1, b=2):
    print 'xxx' if a==1 else 'yyy'
    return a+b


my_xxxyyy_test_experiment.add_variant('a2', a=2)


def test_experiments_function_additions():

    with experiment_testing_context():
        r1=my_xxxyyy_test_experiment.run()
        r2=my_xxxyyy_test_experiment.get_variant('a2').run()

        assert r1.get_log() == 'xxx\n'
        assert r2.get_log() == 'yyy\n'

        assert get_oneline_result_string(my_xxxyyy_test_experiment.get_latest_record()) == '3bbb'
        assert get_oneline_result_string(my_xxxyyy_test_experiment.get_variant('a2').get_latest_record()) == '4bbb'

        with CaptureStdOut() as cap:
            display_experiment_record(my_xxxyyy_test_experiment.get_latest_record())
        assert cap.read() == '3aaa\n'

        with CaptureStdOut() as cap:
            compare_experiment_results([my_xxxyyy_test_experiment, my_xxxyyy_test_experiment.get_variant('a2')])
        assert cap.read() == 'my_xxxyyy_test_experiment: 3, my_xxxyyy_test_experiment.a2: 4\n'

        print_experiment_record_argtable([r1, r2])

        show_experiment_records([r1, r2])


def test_experiment_function_ui():

    with experiment_testing_context():
        for existing_record in my_xxxyyy_test_experiment.get_variant_records(flat=True):
            existing_record.delete()
        assert len(my_xxxyyy_test_experiment.get_variant_records(flat=True)) == 0

        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='run all -p', close_after=True)
        assert len(my_xxxyyy_test_experiment.get_variant_records()) == 2

        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='argtable all', close_after=True)
        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='compare all', close_after=True)
        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='display all', close_after=True)
        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='show all', close_after=True)


if __name__ == '__main__':
    test_experiments_function_additions()
    test_experiment_function_ui()
