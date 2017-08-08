import re
from collections import OrderedDict

from artemis.experiments.experiment_management import load_lastest_experiment_results
from artemis.experiments.experiment_record import NoSavedResultError, ExpInfoFields, ExperimentRecord, ExpStatusOptions
from artemis.experiments.experiments import is_experiment_loadable, GLOBAL_EXPERIMENT_LIBRARY
from artemis.general.display import deepstr, truncate_string, hold_numpy_printoptions, side_by_side, CaptureStdOut, \
    surround_with_header, section_with_header
from artemis.general.nested_structures import flatten_struct
from artemis.general.should_be_builtins import separate_common_items, all_equal, bad_value, izip_equal
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
                }[func]
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


def get_record_full_string(record, show_info = True, show_logs = True, truncate_logs = None, show_result ='deep',
        show_exceptions=True, truncate_result = None, include_bottom_border = True, header_width=64):
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
    full_info_string = surround_with_header(record.get_id(), width=header_width, char='=') + '\n'
    if show_info:
        full_info_string += '{}\n'.format(record.info.get_text())
    if show_logs:
        log = record.get_log()
        if truncate_logs is not None and len(log)>truncate_logs:
            log = log[:truncate_logs-100] + '\n\n ... LOG TRUNCATED TO {} CHARACTERS ... \n\n'.format(truncate_logs) + log[-100:]
        full_info_string += section_with_header('Logs', log, width=header_width)

    error_trace = record.get_error_trace()
    if show_exceptions and error_trace is not None:
        full_info_string += section_with_header('Error Trace', record.get_error_trace(), width=header_width)

    if show_result:
        result_str = get_record_result_string(record, truncate_to=truncate_result, func=show_result)
        full_info_string += section_with_header('Result', result_str, width=header_width, bottom_char='=' if include_bottom_border else None)

    return full_info_string


def get_record_invalid_arg_string(record, recursive=True):
    """
    Return a string identifying ig the arguments for this experiment are still valid.
    :return:
    """
    experiment_id = record.get_experiment_id()
    if is_experiment_loadable(experiment_id):
        if record.info.has_field(ExpInfoFields.ARGS):
            last_run_args = OrderedDict(record.info.get_field(ExpInfoFields.ARGS))
            current_args = OrderedDict(record.get_experiment().get_args())
            validity = record.args_valid(last_run_args=last_run_args, current_args=current_args)
            if validity is False:
                if recursive:
                    last_run_args = OrderedDict(flatten_struct(last_run_args, first_dict_is_namespace=True))
                    current_args = OrderedDict(flatten_struct(current_args, first_dict_is_namespace=True))
                last_arg_str, this_arg_str = [['{}:{}'.format(k, v) for k, v in argdict.iteritems()] for argdict in (last_run_args, current_args)]
                common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
                notes = "No: Args changed!: {{{}}}->{{{}}}".format(','.join(old_args), ','.join(new_args))
            elif validity is None:
                notes = "Cannot Determine: Unhashable Args"
            else:
                notes = "Yes"
        else:
            notes = "Cannot Determine: Inconsistent Experiment Record"
    else:
        notes = "Cannot Determine: Experiment Not Imported"
    return notes


def get_oneline_result_string(record, truncate_to=None, array_float_format='.3g', array_print_threshold=8):
    """
    Get a string that describes the result of the record in one line.  This can optionally be specified by
    experiment.one_liner_function.

    :param record: An ExperimentRecord.
    :param truncate_to:
    :param array_float_format:
    :param array_print_threshold:
    :return: A string with no newlines briefly describing the result of the record.
    """
    if not is_experiment_loadable(record.get_experiment_id()):
        one_liner_function=str
    else:
        one_liner_function = record.get_experiment().one_liner_function
        if one_liner_function is None:
            one_liner_function = str
    return get_record_result_string(record, func=one_liner_function, truncate_to=truncate_to, array_print_threshold=array_print_threshold,
        array_float_format=array_float_format, oneline=True)


def display_experiment_record(record):
    result = record.get_result(err_if_none=False)
    display_func = record.get_experiment().display_function
    if display_func is None:
        print deepstr(result)
    else:
        display_func(result)


def compare_experiment_results(experiments, error_if_no_result = False):
    comp_functions = [ex.comparison_function for ex in experiments]
    assert all_equal(comp_functions), 'Experiments must have same comparison functions.'
    comp_function = comp_functions[0]
    assert comp_function is not None, 'Cannot compare results, because you have not specified any comparison function for this experiment.  Use @ExperimentFunction(comparison_function = my_func)'
    results = load_lastest_experiment_results(experiments, error_if_no_result=error_if_no_result)
    comp_function(results)


def print_experiment_record_argtable(records):
    """
    Print a table comparing experiment arguments and their results.
    """
    funtion_names = [record.info.get_field(ExpInfoFields.FUNCTION) for record in records]
    args = [record.info.get_field(ExpInfoFields.ARGS) for record in records]
    results = [record.get_result(err_if_none=False) for record in records]

    common_args, different_args = separate_common_items(args)

    record_ids = [record.get_id() for record in records]

    def lookup_fcn(record_id, column):
        index = record_ids.index(record_id)
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
        row_categories=record_ids,
        column_categories=['Function', 'Run Time', 'Common Args', 'Different Args', 'Result'],
        prettify_labels=False
        )

    print tabulate(rows)


def show_experiment_records(records, parallel_text=None, hang_notice = None, show_logs=True, truncate_logs=None,
        truncate_result=10000, header_width=100, show_result ='deep', hang=True):
    """
    Show the console logs, figures, and results of a collection of experiments.

    :param records:
    :param parallel_text:
    :param hang_notice:
    :return:
    """
    if isinstance(records, ExperimentRecord):
        records = [records]
    if parallel_text is None:
        parallel_text = len(records)>1
    if len(records)==0:
        print '... No records to show ...'
    else:
        strings = [get_record_full_string(rec, show_logs=show_logs, show_result=show_result, truncate_logs=truncate_logs,
                    truncate_result=truncate_result, header_width=header_width, include_bottom_border=False) for rec in records]
    has_matplotlib_figures = any(loc.endswith('.pkl') for rec in records for loc in rec.get_figure_locs())
    from matplotlib import pyplot as plt
    if has_matplotlib_figures:
        from artemis.plotting.saving_plots import interactive_matplotlib_context
        for rec in records:
            rec.show_figures(hang=False)
        if hang_notice is not None:
            print hang_notice

        with interactive_matplotlib_context(not hang):
            plt.show()

    if any(rec.get_experiment().display_function is not None for rec in records):
        from artemis.plotting.saving_plots import interactive_matplotlib_context
        with interactive_matplotlib_context():
            for i, rec in enumerate(records):
                with CaptureStdOut(print_to_console=False) as cap:
                    display_experiment_record(rec)
                if cap != '':
                    # strings[i] += '{subborder} Result Display {subborder}\n{out} \n{border}'.format(subborder='-'*20, out=cap.read(), border='='*50)
                    strings[i] += section_with_header('Result Display', cap.read(), width=header_width, bottom_char='=')

        if parallel_text:
            print side_by_side(strings, max_linewidth=128)
        else:
            for string in strings:
                print string

    return has_matplotlib_figures


def find_experiment(*search_terms):
    """
    Find an experiment.  Invoke
    :param search_term: A term that will be used to search for an experiment.
    :return:
    """

    found_experiments = OrderedDict((name, ex) for name, ex in GLOBAL_EXPERIMENT_LIBRARY.iteritems() if all(re.search(term, name) for term in search_terms))
    if len(found_experiments)==0:
        raise Exception("None of the {} experiments matched the search: '{}'".format(len(GLOBAL_EXPERIMENT_LIBRARY), search_terms))
    elif len(found_experiments)>1:
        raise Exception("More than one experiment matched the search '{}', you need to be more specific.  Found: {}".format(search_terms, found_experiments.keys()))
    else:
        return found_experiments.values()[0]


def make_record_comparison_table(records, args_to_show=None, results_extractor = None, print_table = False):
    """
    Make a table comparing the arguments and results of different experiment records.  You can use the output
    of this function with the tabulate package to make a nice readable table.

    :param records: A list of records whose results to compare
    :param args_to_show: A list of arguments to show.  If none, it will just show all arguments
        that differ between experiments.
    :param results_extractor: A dict<str->callable> where the callables take the result of the
        experiment as an argument and return an entry in the table.
    :param print_table: Optionally, import tabulate and print the table here and now.
    :return: headers, rows
        headers is a list of of headers for the top of the table
        rows is a list of lists filling in the information.

    example usage:

        headers, rows = make_record_comparison_table(
            record_ids = [experiment_id_to_latest_record_id(eid) for eid in [
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.version=mlp',
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.full-gd.n_steps=1',
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.full-gd.n_steps=20',
                ]],
            results_extractor={
                'Test': lambda result: result.get_best('test').score.get_score('test'),
                'Train': lambda result: result.get_best('test').score.get_score('train'),
                }
             )
        import tabulate
        print tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt)
    """

    args = [rec.info.get_field(ExpInfoFields.ARGS) for rec in records]
    if args_to_show is None:
        common, separate = separate_common_items(args)
        args_to_show = [k for k, v in separate[0]]

    if results_extractor is None:
        results_extractor = {'Result': str}
    elif callable(results_extractor):
        results_extractor = {'Result': results_extractor}
    else:
        assert isinstance(results_extractor, dict)

    headers = args_to_show + results_extractor.keys()

    rows = []
    for record, record_args in izip_equal(records, args):
        arg_dict = dict(record_args)
        args_vals = [arg_dict[k] for k in args_to_show]
        results = record.get_result()
        rows.append(args_vals+[f(results) for f in results_extractor.values()])

    if print_table:
        import tabulate
        print tabulate.tabulate(rows, headers=headers, tablefmt='simple')
    return headers, rows

