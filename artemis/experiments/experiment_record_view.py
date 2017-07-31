from artemis.experiments.experiment_management import load_lastest_experiment_results
from artemis.experiments.experiment_record import NoSavedResultError, ExpInfoFields, load_experiment_record
from artemis.experiments.experiments import is_experiment_loadable, load_experiment
from artemis.general.display import deepstr, truncate_string, hold_numpy_printoptions
from artemis.general.should_be_builtins import separate_common_items, all_equal, bad_value
from artemis.general.tables import build_table
from tabulate import tabulate


def get_record_result_string(record, func='deep', truncate_to = None, array_print_threshold=8, array_float_format='.3g', oneline=False):
    """
    Get a string representing the result of the experiment.
    :param record:
    :param func:
    :return:
    """
    with hold_numpy_printoptions(threshold = array_print_threshold, formatter={'float': lambda x: '{{:{}}}'.format(array_float_format).format(x)}):
        if isinstance(func, basestring):
            func = {
                'deep': deepstr,
                'str': str,
                }
        else:
            assert callable(func), 'func must be callable'
        try:
            result = record.get_result()
        except NoSavedResultError:
            return '<No result has been saved>'
        string = func(result)

    if truncate_to is not None:
        string = truncate_string(string, truncation=truncate_to, message = '...<truncated>')
    if oneline:
        string = string.replace('\n', ', ')
    return string


def get_record_full_string(record, show_info = True, show_logs = True, truncate_logs = None, show_result ='deep', truncate_result = None):
    """
    Get a human-readable string containing info about the experiment record.

    :param show_info: Show info about experiment (name, id, runtime, etc)
    :param show_logs: Show logs (True, False, or an integer character count to truncate logs at)
    :param show_result: Show the result.  Options for display are:
        'short': Print a one-liner (note: sometimes prints multiple lines)
        'long': Directly call str
        'deep': Use the deepstr function for a compact nested printout.
    :return: A string to print.
    """

    assert show_result in (False, 'full', 'deep')
    full_info_string = '{header} {rid} {header}\n'.format(header="=" * 10, rid=record.get_identifier())
    if show_info:
        full_info_string += '{}\n'.format(record.info.get_text())
    if show_logs:
        log = record.get_log()
        if truncate_logs is not None and len(log)>truncate_logs:
            log = log[:truncate_logs-100] + '\n\n ... LOG TRUNCATED TO {} CHARACTERS ... \n\n'.format(truncate_logs) + log[-100:]
        full_info_string += '{subborder} Logs {subborder}\n{log}\n'.format(subborder='-' * 20, log=log)
    if show_result:
        result_str = get_record_result_string(record, truncate_to=truncate_result, func=show_result)
        full_info_string += '{subborder} Result {subborder}\n{result}\n'.format(subborder='-' * 20, result=result_str)
    full_info_string += "=" * 50 + '\n'
    return full_info_string


def get_record_invalid_arg_string(record):
    """
    Return a string identifying ig the arguments for this experiment are still valid.
    :return:
    """
    experiment_id = record.get_experiment_id()
    if is_experiment_loadable(experiment_id):
        last_run_args = dict(record.info.get_field(ExpInfoFields.ARGS))
        current_args = dict(record.get_experiment().get_args())
        validity = record.is_valid(last_run_args=last_run_args, current_args=current_args)
        if validity is False:
            last_arg_str, this_arg_str = [
                ['{}:{}'.format(k, v) for k, v in argdict.iteritems()] if isinstance(argdict, dict) else
                ['{}:{}'.format(k, v) for k, v in argdict]
                for argdict in (last_run_args, current_args)]
            common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
            notes = "No: Args changed!: {{{}}}->{{{}}}".format(','.join(old_args), ','.join(new_args))
        elif validity is None:
            notes = "Cannot Determine."
        else:
            notes = "Yes"
    else:
        notes = "<Experiment Not Currently Imported>"
    return notes


def show_record(record, hang=False):
    if isinstance(record, basestring):
        record = load_experiment_record(record)
    print get_record_full_string(record)
    record.show_figures(hang=hang)


def get_oneline_result_string(record, truncate_to=None, array_float_format='.3g', array_print_threshold=8):

    one_liner = record.get_experiment().one_liner_function
    if one_liner is None:
        one_liner = 'str'
    return get_record_result_string(record, func=one_liner, truncate_to=truncate_to, array_print_threshold=array_print_threshold,
        array_float_format=array_float_format, oneline=True)


def display_record(record):
    result = record.get_result()
    display_func = record.get_experiment().display_function
    if display_func is None:
        print deepstr(result)
    else:
        display_func(result)


def compare_results(experiment_ids, error_if_no_result = True):
    comp_functions = [load_experiment(eid).comparison_function for eid in experiment_ids]
    assert all_equal(comp_functions), 'Experiments must have same comparison functions.'
    comp_function = comp_functions[0]
    assert comp_function is not None, 'Cannot compare results, because you have not specified any comparison function for this experiment.  Use @ExperimentFunction(comparison_function = my_func)'
    results = load_lastest_experiment_results(experiment_ids)
    comp_function(results)


def print_experiment_record_argtable(record_identifiers):

    records = [load_experiment_record(ident) for ident in record_identifiers]
    # info_results = OrderedDict([(identifier, record.get_info()) for identifier, record in zip(record_identifiers, records)]])

    funtion_names = [record.info.get_field(ExpInfoFields.FUNCTION) for record in records]
    args = [record.info.get_field(ExpInfoFields.ARGS) for record in records]
    results = [record.get_result() for record in records]

    common_args, different_args = separate_common_items(args)

    def lookup_fcn(identifier, column):
        index = record_identifiers.index(identifier)
        if column=='Function':
            return funtion_names[index]
        elif column=='Run Time':
            return records[index].info.get_field_text(ExpInfoFields.RUNTIME)
        elif column=='Common Args':
            return ', '.join('{}={}'.format(k, v) for k, v in common_args)
        elif column=='Different Args':
            return ', '.join('{}={}'.format(k, v) for k, v in different_args[index])
        elif column=='Result':
            return results[index]
        else:
            bad_value(column)

    rows = build_table(lookup_fcn,
        row_categories=record_identifiers,
        column_categories=['Function', 'Run Time', 'Common Args', 'Different Args', 'Result'],
        prettify_labels=False
        )

    print tabulate(rows)