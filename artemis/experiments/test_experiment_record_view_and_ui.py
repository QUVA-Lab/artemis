import pytest

from artemis.experiments.decorators import ExperimentFunction, experiment_function
from artemis.experiments.experiment_record_view import get_oneline_result_string, print_experiment_record_argtable, \
    compare_experiment_records, get_record_invalid_arg_string
from artemis.experiments.experiments import experiment_testing_context, clear_all_experiments
from artemis.general.display import CaptureStdOut, assert_things_are_printed


def display_it(result):
    print(str(result) + 'aaa')


def one_liner(result):
    return str(result) + 'bbb'


def compare_them(results):
    print(', '.join('{}: {}'.format(k, results[k]) for k in sorted(results.keys())))


@ExperimentFunction(display_function=display_it, one_liner_function=one_liner, comparison_function=compare_them)
def my_xxxyyy_test_experiment(a=1, b=2):

    if b==17:
        raise Exception('b should never be 17')

    print('xxx' if a==1 else 'yyy')
    return a+b


my_xxxyyy_test_experiment.add_variant('a2', a=2)
my_xxxyyy_test_experiment.add_variant(b=17)


def test_experiments_function_additions():

    with experiment_testing_context():

        for rec in my_xxxyyy_test_experiment.get_variant_records(flat=True):
            rec.delete()

        r1=my_xxxyyy_test_experiment.run()
        r2=my_xxxyyy_test_experiment.get_variant('a2').run()
        with pytest.raises(Exception):
            my_xxxyyy_test_experiment.get_variant(b=17).run()
        r3 = my_xxxyyy_test_experiment.get_variant(b=17).get_latest_record()

        assert r1.get_log() == 'xxx\n'
        assert r2.get_log() == 'yyy\n'

        assert get_oneline_result_string(my_xxxyyy_test_experiment.get_latest_record()) == '3bbb'
        assert get_oneline_result_string(my_xxxyyy_test_experiment.get_variant('a2').get_latest_record()) == '4bbb'
        assert get_oneline_result_string(my_xxxyyy_test_experiment.get_variant(b=17).get_latest_record()) == '<No result has been saved>'

        with CaptureStdOut() as cap:
            my_xxxyyy_test_experiment.show(my_xxxyyy_test_experiment.get_latest_record())
        assert cap.read() == '3aaa\n'

        with CaptureStdOut() as cap:
            my_xxxyyy_test_experiment.compare([r1, r2])
        assert cap.read() == 'my_xxxyyy_test_experiment: 3, my_xxxyyy_test_experiment.a2: 4\n'

        print('='*100+'\n ARGTABLE \n'+'='*100)
        print_experiment_record_argtable([r1, r2, r3])

        print('='*100+'\n SHOW \n'+'='*100)
        compare_experiment_records([r1, r2, r3])


def test_experiment_function_ui():

    with experiment_testing_context():
        for existing_record in my_xxxyyy_test_experiment.get_variant_records(flat=True):
            existing_record.delete()

        assert len(my_xxxyyy_test_experiment.get_variant_records(flat=True)) == 0

        my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='run all', close_after=True)
        assert len(my_xxxyyy_test_experiment.get_variant_records()) == 3

        import time
        time.sleep(0.1)

        with assert_things_are_printed(min_len=1200, things=['Common Args', 'Different Args', 'Result', 'a=1, b=2', 'a=2, b=2', 'a=1, b=17']):
            my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='argtable all', close_after=True)

        with assert_things_are_printed(min_len=600, things=['my_xxxyyy_test_experiment: 3', 'my_xxxyyy_test_experiment.a2: 4']):
            my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='compare all -r', close_after=True)

        with assert_things_are_printed(min_len=1300, things=['Result', 'Logs', 'Ran Succesfully', 'Traceback']):
            my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='show all -o', close_after=True)

        with assert_things_are_printed(min_len=1250, things=['3aaa', '4aaa']):
            my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='show all -r', close_after=True)

        with assert_things_are_printed(min_len=1700, things=['Result', 'Logs', 'Ran Succesfully']):
            my_xxxyyy_test_experiment.browse(raise_display_errors=True, command='show 0 -o', close_after=True)


def test_invalid_arg_text():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_invalid_arg_test(a=1, b={'c': 3, 'd': 4}):
            return a+b['c']+b['d']

        record = my_invalid_arg_test.run()
        assert get_record_invalid_arg_string(record, recursive=True) == '<No Change>'
        clear_all_experiments()

        @experiment_function
        def my_invalid_arg_test(a=2, b={'c': 3, 'd': 4}):
            return a+b['c']+b['d']

        assert get_record_invalid_arg_string(record, recursive=True) == 'Change: {a:1}->{a:2}'
        clear_all_experiments()

        @experiment_function
        def my_invalid_arg_test(a=2, b={'c': 3, 'd': 2}):
            return a+b['c']+b['d']

        assert get_record_invalid_arg_string(record, recursive=True) == "Change: {a:1,b['d']:4}->{a:2,b['d']:2}"


class MyArgumentObject(object):

    def __init__(self, a=1):
        self.a=a


def test_invalid_arg_text_when_object_arg():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_unhashable_arg_test(a=MyArgumentObject(a=3)):
            return a.a+2

        record = my_unhashable_arg_test.run()
        assert record.get_result() == 5

        assert get_record_invalid_arg_string(record, recursive=True) == '<No Change>'

        # ---------------------
        clear_all_experiments()

        @experiment_function
        def my_unhashable_arg_test(a=MyArgumentObject(a=3)):
            return a.a+2

        assert get_record_invalid_arg_string(record, recursive=True) == '<No Change>'

        # ---------------------
        clear_all_experiments()

        @experiment_function
        def my_unhashable_arg_test(a=MyArgumentObject(a=4)):
            return a.a+2

        assert get_record_invalid_arg_string(record, recursive=True) == 'Change: {a.a:3}->{a.a:4}'


def test_simple_experiment_show():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_simdfsfdsgfs(a=1):

            print('xxxxx')
            print('yyyyy')
            return a+2

        rec = my_simdfsfdsgfs.run()

        with assert_things_are_printed(things=['my_simdfsfdsgfs', 'xxxxx\nyyyyy\n']):
            compare_experiment_records(rec)


def test_view_modes():

    with experiment_testing_context(new_experiment_lib=True):

        @experiment_function
        def my_simdfsffsdfsfs(a=1):

            print('xxxxx')
            print('yyyyy')
            return a+2

        my_simdfsffsdfsfs.add_variant(a=2)
        my_simdfsffsdfsfs.add_variant(a=3)

        my_simdfsffsdfsfs.run()

        for view_mode in ('full', 'results'):
            my_simdfsffsdfsfs.browse(view_mode=view_mode, command = 'q')


if __name__ == '__main__':
    test_experiments_function_additions()
    test_experiment_function_ui()
    test_invalid_arg_text()
    test_invalid_arg_text_when_object_arg()
    test_simple_experiment_show()
    test_view_modes()
