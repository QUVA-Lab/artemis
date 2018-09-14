import re
from collections import OrderedDict

from six import string_types
from tabulate import tabulate
import numpy as np
from artemis.experiments.experiment_record import NoSavedResultError, ExpInfoFields, ExperimentRecord, \
    load_experiment_record, is_matplotlib_imported, UnPicklableArg
from artemis.general.display import deepstr, truncate_string, hold_numpy_printoptions, side_by_side, \
    surround_with_header, section_with_header, dict_to_str
from artemis.general.nested_structures import flatten_struct, PRIMATIVE_TYPES
from artemis.general.should_be_builtins import separate_common_items, bad_value, izip_equal, \
    remove_duplicates, get_unique_name, entries_to_table
from artemis.general.tables import build_table


def get_record_result_string(record, func='deep', truncate_to = None, array_print_threshold=8, array_float_format='.3g', oneline=False, default_one_liner_func=str):
    """
    Get a string representing the result of the experiment.
    :param record:
    :param func:
    :return:
    """
    with hold_numpy_printoptions(threshold = array_print_threshold, formatter={'float': lambda x: '{{:{}}}'.format(array_float_format).format(x)}):
        if isinstance(func, string_types):
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
        if not isinstance(string, string_types):
            string = default_one_liner_func(string)

    if truncate_to is not None:
        string = truncate_string(string, truncation=truncate_to, message = '...<truncated>')
    if oneline:
        string = string.replace('\n', ', ')
    return string


def _get_record_info_section(record, header_width):
    return section_with_header('Info', record.info.get_text(), width=header_width)


def _get_record_log_section(record, truncate_logs = None, header_width=64):
    log = record.get_log()
    if truncate_logs is not None and len(log)>truncate_logs:
        log = log[:truncate_logs-100] + '\n\n ... LOG TRUNCATED TO {} CHARACTERS ... \n\n'.format(truncate_logs) + log[-100:]
    return section_with_header('Logs', log, width=header_width)


def _get_record_error_trace_section(record, header_width):
    error_trace = record.get_error_trace()
    if error_trace is None:
        return ''
    else:
        return section_with_header('Error Trace', record.get_error_trace(), width=header_width)


def _get_result_section(record, truncate_result, show_result, header_width):
    assert show_result in (False, 'full', 'deep')
    result_str = get_record_result_string(record, truncate_to=truncate_result, func=show_result)
    return section_with_header('Result', result_str, width=header_width)


def get_record_full_string(record, show_info = True, show_logs = True, truncate_logs = None, show_result ='deep',
        show_exceptions=True, truncate_result = None, include_bottom_border = True, header_width=64, return_list = False):
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

    parts = [surround_with_header(record.get_id(), width=header_width, char='=')]

    if show_info:
        parts.append(section_with_header('Info', record.info.get_text(), width=header_width))

    if show_logs:
        log = record.get_log()
        if truncate_logs is not None and len(log)>truncate_logs:
            log = log[:truncate_logs-100] + '\n\n ... LOG TRUNCATED TO {} CHARACTERS ... \n\n'.format(truncate_logs) + log[-100:]
        # return section_with_header('Logs', log, width=header_width)
        parts.append(section_with_header('Logs', log, width=header_width))

    if show_exceptions:
        error_trace = record.get_error_trace()
        error_trace_text = '' if error_trace is None else section_with_header('Error Trace', record.get_error_trace(), width=header_width)
        parts.append(error_trace_text)

    if show_result:
        assert show_result in (False, 'full', 'deep')
        result_str = get_record_result_string(record, truncate_to=truncate_result, func=show_result)
        parts.append(section_with_header('Result', result_str, width=header_width))

    if return_list:
        return parts
    else:
        return '\n'.join(parts)


def get_record_invalid_arg_string(record, recursive=False, ignore_valid_keys=(), note_version = 'full'):
    """
    Return a string identifying ig the arguments for this experiment are still valid.
    :return:
    """
    from artemis.experiments.experiments import is_experiment_loadable
    assert note_version in ('full', 'short')
    experiment_id = record.get_experiment_id()
    if is_experiment_loadable(experiment_id):
        if record.info.has_field(ExpInfoFields.ARGS):
            last_run_args = OrderedDict([(k,v) for k,v in record.get_args().items() if k not in ignore_valid_keys])
            current_args = OrderedDict([(k,v) for k,v in record.get_experiment().get_args().items() if k not in ignore_valid_keys])
            if recursive:
                last_run_args = OrderedDict(flatten_struct(last_run_args, first_dict_is_namespace=True, primatives = PRIMATIVE_TYPES + (UnPicklableArg, )))
                last_run_args = OrderedDict([(k, v) for k, v in last_run_args.items() if k not in ignore_valid_keys])
                current_args = OrderedDict(flatten_struct(current_args, first_dict_is_namespace=True))
                current_args = OrderedDict([(k, v) for k, v in current_args.items() if k not in ignore_valid_keys])

            validity = record.args_valid(last_run_args=last_run_args, current_args=current_args)
            if validity is False:
                common, (old_args, new_args) = separate_common_items([list(last_run_args.items()), list(current_args.items())])
                if len(old_args)+len(new_args)==0:
                    raise Exception('Error displaying different args.  Bug Peter.')

                all_changed_arg_names = remove_duplicates(list(name for name, _ in old_args)+list(name for name, _ in new_args))
                changestr = ', '.join("{}:{}->{}".format(k, last_run_args[k] if k in last_run_args else '<N/A>', current_args[k] if k in current_args else '<N/A>') for k in all_changed_arg_names)
                notes = ("Change: " if note_version=='full' else "") + changestr
            elif validity is None:
                notes = "Cannot Determine: Unhashable Args" if note_version=='full' else '<Unhashable Args>'
            else:
                notes = "<No Change>"
        else:
            notes = "Cannot Determine: Inconsistent Experiment Record" if note_version == 'full' else '<Inconsistent Record>'
    else:
        notes = "Cannot Determine: Experiment Not Imported" if note_version=='full' else '<Not Imported>'
    return notes


def get_oneline_result_string(record, truncate_to=None, array_float_format='.3g', array_print_threshold=8, default_one_liner_func=dict_to_str):
    """
    Get a string that describes the result of the record in one line.  This can optionally be specified by
    experiment.one_liner_function.

    :param record: An ExperimentRecord.
    :param truncate_to:
    :param array_float_format:
    :param array_print_threshold:
    :return: A string with no newlines briefly describing the result of the record.
    """
    from artemis.experiments.experiments import is_experiment_loadable
    if isinstance(record, string_types):
        record = load_experiment_record(record)
    if not is_experiment_loadable(record.get_experiment_id()):
        one_liner_function=default_one_liner_func
    else:
        one_liner_function = record.get_experiment().one_liner_function
        if one_liner_function is None:
            one_liner_function = default_one_liner_func
    return get_record_result_string(record, func=one_liner_function, truncate_to=truncate_to, array_print_threshold=array_print_threshold,
        array_float_format=array_float_format, oneline=True, default_one_liner_func=default_one_liner_func)


def print_experiment_record_argtable(records):
    """
    Print a table comparing experiment arguments and their results.
    """
    funtion_names = [record.info.get_field(ExpInfoFields.FUNCTION) for record in records]
    args = [record.get_args() for record in records]
    common_args, different_args = separate_common_items(args)
    record_ids = [record.get_id() for record in records]
    def lookup_fcn(record_id, column):
        index = record_ids.index(record_id)
        if column=='Function':
            return funtion_names[index]
        elif column=='Run Time':
            return records[index].info.get_field_text(ExpInfoFields.RUNTIME)
        elif column=='Common Args':
            return ', '.join('{}={}'.format(k, v) for k, v in common_args.items())
        elif column=='Different Args':
            return ', '.join('{}={}'.format(k, v) for k, v in different_args[index].items())
        elif column=='Result':
            return get_oneline_result_string(records[index])
        else:
            bad_value(column)

    rows = build_table(lookup_fcn,
        row_categories=record_ids,
        column_categories=['Function', 'Run Time', 'Common Args', 'Different Args', 'Result'],
        prettify_labels=False
        )

    print(tabulate(rows))


def get_column_change_ordering(tabular_data):
    """
    Get the order in which to rearrange the columns so that the fastest-changing data comes last.

    :param tabular_data: A list of equal-length lists
    :return: A set of permutation indices for the columns.
    """
    n_rows, n_columns = len(tabular_data), len(tabular_data[0])
    deltas = [sum(row_prev[i]!=row[i] for row_prev, row in zip(tabular_data[:-1], tabular_data[1:])) for i in range(n_columns)]
    return np.argsort(deltas)


def get_different_args(args, no_arg_filler = 'N/A', arrange_by_deltas=False):
    """
    Get a table of different args between records.
    :param Sequence[List[Tuple[str, Any]]] args: A list of lists of argument (name, value) pairs.
    :param no_arg_filler: The filler value to use if a record does not have a particular argument (possibly due to an argument being added to the code after the record was made)
    :param arrange_by_deltas: If true, order arguments so that the fastest-changing ones are in the last column
    :return Tuple[List[str], List[List[Any]]]: (arg_names, arg_values) where:
        arg_names is a list of arguments that differ between records
        arg_values is a len(records)-list of len(arg_names) lists of values of the arguments for each record.
    """
    args = list(args)
    common_args, different_args = separate_common_items(args)
    all_different_args = list(remove_duplicates((k for dargs in different_args for k in dargs.keys())))
    values = [[record_args[argname] if argname in record_args else no_arg_filler for argname in all_different_args] for record_args in args]
    if arrange_by_deltas:
        col_shuf_ixs = get_column_change_ordering(values)
        all_different_args = [all_different_args[i] for i in col_shuf_ixs]
        values = [[row[i] for i in col_shuf_ixs] for row in values]
    return all_different_args, values


def get_exportiment_record_arg_result_table(records):
    record_ids = [record.get_id() for record in records]
    all_different_args, arg_values = get_different_args([r.get_args() for r in records], no_arg_filler='N/A')

    parsed_results = [record.get_experiment().result_parser(record.get_result()) for record in records]
    result_fields, result_data = entries_to_table(parsed_results)
    result_fields = [get_unique_name(rf, all_different_args) for rf in result_fields]  # Just avoid name collisions

    # result_column_name = get_unique_name('Results', taken_names=all_different_args)

    def lookup_fcn(record_id, arg_or_result_name):
        row_index = record_ids.index(record_id)
        if arg_or_result_name in result_fields:
            return result_data[row_index][result_fields.index(arg_or_result_name)]
        else:
            column_index = all_different_args.index(arg_or_result_name)
            return arg_values[row_index][column_index]

    rows = build_table(lookup_fcn,
        row_categories=record_ids,
        column_categories=all_different_args + result_fields,
        prettify_labels=False,
        include_row_category=False,
        )

    return rows[0], rows[1:]

    # return tabulate(rows[1:], headers=rows[0])


def show_record(record, show_logs=True, truncate_logs=None, truncate_result=10000, header_width=100, show_result ='deep', hang=True):
    """
    Show the results of an experiment record.
    :param record:
    :param show_logs:
    :param truncate_logs:
    :param truncate_result:
    :param header_width:
    :param show_result:
    :param hang:
    :return:
    """
    string = get_record_full_string(record, show_logs=show_logs, show_result=show_result, truncate_logs=truncate_logs,
        truncate_result=truncate_result, header_width=header_width, include_bottom_border=False)

    has_matplotlib_figures = any(loc.endswith('.pkl') for loc in record.get_figure_locs())
    if has_matplotlib_figures:
        record.show_figures(hang=hang)
    print(string)


def show_multiple_records(records, func = None):

    if func is None:
        func = lambda rec: rec.get_experiment().show(rec)

    if is_matplotlib_imported():
        from artemis.plotting.manage_plotting import delay_show
        with delay_show():
            for rec in records:
                func(rec)
    else:
        for rec in records:
            func(rec)


def compare_experiment_records(records, parallel_text=None, show_logs=True, truncate_logs=None,
        truncate_result=10000, header_width=100, max_linewidth=128, show_result ='deep'):
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
        print('... No records to show ...')
        return
    else:
        records_sections = [get_record_full_string(rec, show_logs=show_logs, show_result=show_result, truncate_logs=truncate_logs,
                    truncate_result=truncate_result, header_width=header_width, include_bottom_border=False, return_list=True) for rec in records]

    if parallel_text:
        full_string = '\n'.join(side_by_side(records_section, max_linewidth=max_linewidth) for records_section in zip(*records_sections))
    else:
        full_string = '\n'.join('\n'.join(record_sections) for record_sections in records_sections)

    print(full_string)

    has_matplotlib_figures = any(loc.endswith('.pkl') for rec in records for loc in rec.get_figure_locs())
    if has_matplotlib_figures:
        from artemis.plotting.manage_plotting import delay_show
        with delay_show():
            for rec in records:
                rec.show_figures()

    return has_matplotlib_figures


def make_record_comparison_table(records, args_to_show=None, results_extractor = None, print_table = False, tablefmt='simple', reorder_by_args=False):
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

    args = [rec.get_args(ExpInfoFields.ARGS) for rec in records]
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

    if reorder_by_args:
        rows = sorted(rows)

    if print_table:
        import tabulate
        print(tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt))
    return headers, rows


def separate_common_args(records, as_dicts=False, return_dict = False, only_shared_argdiffs = False):
    """

    :param records: A List of records
    :param return_dict: Return the different args as a dict<ExperimentRecord: args>
    :return: (common, different)
        Where common is an OrderedDict of common args
        different is a list (the same lengths of records) of OrderedDicts containing args that are not the same in all records.
    """
    common, argdiff = separate_common_items([list(rec.get_args().items()) for rec in records])
    if only_shared_argdiffs:
        args_that_they_all_have = set.intersection(*(set(k for k, v in different) for different in argdiff))
        argdiff = [[(k, v) for k, v in ad if k in args_that_they_all_have] for ad in argdiff]

    if as_dicts:
        common = OrderedDict(common)
        argdiff = [OrderedDict(ad) for ad in argdiff]

    if return_dict:
        argdiff = {rec.get_id(): args for rec, args in zip(records, argdiff)}
    return common, argdiff
