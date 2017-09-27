import getpass
import multiprocessing
import traceback
from collections import OrderedDict
from functools import partial
from importlib import import_module
from six import string_types
from six.moves import reduce, xrange

from artemis.experiments.experiment_record import (load_experiment_record, ExpInfoFields,
    ExpStatusOptions, ARTEMIS_LOGGER, record_id_to_experiment_id)
from artemis.experiments.experiments import load_experiment, get_global_experiment_library
from artemis.fileman.config_files import get_home_dir
from artemis.general.hashing import compute_fixed_hash
from artemis.general.should_be_builtins import izip_equal, detect_duplicates, remove_common_prefix, memoize


def pull_experiments(user, ip, experiment_names, include_variants=True):
    """
    Pull experiments from another computer matching the given experiment name.

    :param user:
    :param ip:
    :param experiment_name:
    :param include_variants:
    :return:
    """
    import pexpect
    import sys

    if isinstance(experiment_names, string_types):
        experiment_names = [experiment_names]

    inclusions = ' '.join("--include='**/*-{exp_name}{variants}/*'".format(exp_name=exp_name, variants = '*' if include_variants else '') for exp_name in experiment_names)

    home = get_home_dir()

    command = "rsync -a -m -i {inclusions} --include='*/' --exclude='*' {user}@{ip}:~/.artemis/experiments/ {home}/.artemis/experiments/".format(
        inclusions=inclusions,
        user=user,
        ip=ip,
        home=home
        )
    password = getpass.getpass("Enter password for {}@{}:".format(user, ip))
    child = pexpect.spawn(command)
    code = child.expect([pexpect.TIMEOUT, 'password:'])
    if code == 0:
        print(("Got unexpected output: %s %s" % (child.before, child.after)))
        sys.exit()
    else:
        child.sendline(password)
    output = child.read()
    return output


def load_lastest_experiment_results(experiments, error_if_no_result = True):
    """
    Given a list of experiments (or experiment ids), return an OrderedDict<record_id: result>
    :param experiments: A list of Experiment objects (or strings identifying experiment ID is ok too)
    :param error_if_no_result: Raise an error if an experiment has no completed results.
    :return: OrderedDict<record_id: result>
    """
    experiments = [load_experiment(ex) if isinstance(ex, string_types) else ex for ex in experiments]
    records = [ex.get_latest_record(err_if_none=error_if_no_result, only_completed=True) for ex in experiments]
    record_results = load_record_results([r for r in records if r is not None], err_if_no_result=error_if_no_result)
    experiment_latest_results = OrderedDict((rec.get_experiment_id(), val) for rec, val in record_results.items())
    return experiment_latest_results


def load_record_results(records, err_if_no_result =True, index_by_id = False):
    """
    Given a list of experiment records, return an OrderedDict<record: result>
    :param records: A list of ExperimentRecord objects
    :param err_if_no_result: True to raise an error if a record has no result.
    :return:  OrderedDict<ExperimentRecord: result>
    """
    results = OrderedDict()
    for record in records:
        index = record.get_id() if index_by_id else record
        if not record.has_result():
            if err_if_no_result:
                raise Exception('Record {} had no result.'.format(record.get_id()))
            else:
                ARTEMIS_LOGGER.warn('Experiment Record {} had no saved result.  Not including this in results'.format(record.get_id()))
        else:
            results[index] = record.get_result()
    return results


def select_experiments(user_range, exp_record_dict, return_dict=False):

    exp_filter = _filter_experiments(user_range, exp_record_dict)
    if return_dict:
        return OrderedDict((name, exp_record_dict[name]) for name in exp_record_dict if exp_filter[name])
    else:
        return [name for name in exp_record_dict if exp_filter[name]]


def select_last_record_of_experiments(user_range, exp_record_dict):
    experiments = select_experiments(user_range=user_range, exp_record_dict=exp_record_dict)
    records = [load_experiment(ex).get_latest_record(only_completed=True, err_if_none=False) for ex in experiments]
    if None in records:
        print('WARNING: Experiments {} have no completed records.', [e for e, r in izip_equal(experiments, records) if r is None])
    return records


def _filter_experiments(user_range, exp_record_dict):

    if user_range in exp_record_dict:
        is_in = [k==user_range for k in exp_record_dict]
    else:
        number_range = interpret_numbers(user_range)
        if number_range is not None:
            # experiment_ids = [experiment_list[i] for i in number_range]
            is_in = [i in number_range for i in xrange(len(exp_record_dict))]
        elif user_range == 'all':
            # experiment_ids = experiment_list
            is_in = [True]*len(exp_record_dict)
        elif user_range.startswith('has:'):
            phrase = user_range[len('has:'):]
            # experiment_ids = [exp_id for exp_id in experiment_list if phrase in exp_id]
            is_in = [phrase in exp_id for exp_id in exp_record_dict]
        elif user_range.startswith('1diff:'):
            # select experiments whose arguments differ by one element from the selected experiments
            base_range = user_range[len('1diff:'):]
            base_range_exps = select_experiments(base_range, exp_record_dict) # list<experiment_id>
            all_exp_args_hashes = {eid: set(compute_fixed_hash(a) for a in load_experiment(eid).get_args().items()) for eid in exp_record_dict} # dict<experiment_id : set<arg_hashes>>
            # assert all_equal_length(all_exp_args_hashes.values()), 'All variants must have the same number of arguments' # Note: we diable this because we may have lists of experiments with different root functions.
            is_in = [any(len(all_exp_args_hashes[eid].difference(all_exp_args_hashes[other_eid]))<=1 for other_eid in base_range_exps) for eid in exp_record_dict]
        elif user_range.startswith('hasnot:'):
            phrase = user_range[len('hasnot:'):]
            # experiment_ids = [exp_id for exp_id in experiment_list if phrase not in exp_id]
            is_in = [phrase not in exp_id for exp_id in exp_record_dict]
        elif user_range in ('unfinished', 'invalid'):  # Return all experiments where all records are unfinished/invalid
            record_filters = _filter_records(user_range, exp_record_dict)
            # experiment_ids = [exp_id for exp_id in experiment_list if len(record_filters[exp_id])]
            is_in = [all(record_filters[exp_id]) for exp_id in exp_record_dict]
        else:
            raise Exception("Don't know how to use input '{}' to select experiments".format(user_range))

    return OrderedDict((exp_id, exp_is_in) for exp_id, exp_is_in in izip_equal(exp_record_dict, is_in))


def select_experiment_records(user_range, exp_record_dict, flat=True, load_records = True):
    """
    :param user_range:
    :param exp_record_dict: An OrderedDict<experiment_name: list<experiment_record_name>>
    :param flat: Return a list of experiment records, instead of an OrderedDict
    :return: if not flat, an An OrderedDict<experiment_name: list<experiment_record_name>>
        otherwise a list<experiment_record_name>
    """
    filters = _filter_records(user_range, exp_record_dict)
    filtered_dict = _select_record_from_filters(filters, exp_record_dict) if load_records else _select_record_ids_from_filters(filters, exp_record_dict)
    if flat:
        return [record_id for records in filtered_dict.values() for record_id in records]
    else:
        return filtered_dict


def _select_record_from_filters(filters, exp_record_dict):
    return OrderedDict((k, [load_experiment_record(record_id) for record_id, f in izip_equal(exp_record_dict[k], filters[k]) if f]) for k in exp_record_dict.keys())


def _select_record_ids_from_filters(filters, exp_record_dict):
    return OrderedDict((k, [record_id for record_id, f in izip_equal(exp_record_dict[k], filters[k]) if f]) for k in exp_record_dict.keys())


def _bitwise_and(a, b):
    return [a_ and b_ for a_, b_ in izip_equal(a, b)]


def _bitwise_or(a, b):
    return [a_ or b_ for a_, b_ in izip_equal(a, b)]


def _bitwise_andcascade(a, b):
    """
    :param a: A list of booleans whose length matches the number of True elements in b
    :param b: A list of booleans
    :return: A list
    """
    assert sum(b) == len(a), 'The number of elements in b: {}, did not match the number of true elements in a: {}'.format(sum(b), len(a))
    a_iter = iter(a)
    return [b_ and next(a_iter) for b_ in b]


def _bitwise_not(a):
    return [not a_ for a_ in a]


def _bitwise_filter_op(op, *filter_sets):

    output_set = filter_sets[0].copy()
    if op=='not':
        assert len(filter_sets)==1
        for k in output_set.keys():
            output_set[k] = _bitwise_not(filter_sets[0][k])
    elif op in ('and', 'or'):
        for k in output_set.keys():
            output_set[k] = reduce(_bitwise_and if op=='and' else _bitwise_or, [fs[k] for fs in filter_sets])
    elif op=='andcascade':
        for k in output_set.keys():
            output_set[k] = reduce(_bitwise_andcascade, [fs[k] for fs in filter_sets[::-1]])
    else:
        raise AssertionError('op should be one of {}'.format(('and', 'or', 'andcascade', 'not')))
    return output_set


def _filter_records(user_range, exp_record_dict):
    """
    :param user_range:
    :param exp_record_dict:
    :return: An OrderedDict<experiment_id -> list<True or False>> indicating whether each record from the given experiment passed the filter
    """

    if user_range=='unfinished':
        return _filter_records('~finished', exp_record_dict)
    elif user_range=='last':
        return _filter_records('~old', exp_record_dict)
    elif '|' in user_range:
        return _bitwise_filter_op('or', *[_filter_records(subrange, exp_record_dict) for subrange in user_range.split('|')])
    elif '&' in user_range:
        return _bitwise_filter_op('and', *[_filter_records(subrange, exp_record_dict) for subrange in user_range.split('&')])
    elif '>' in user_range:
        ix = user_range.index('>')
        first_part, second_part = user_range[:ix], user_range[ix+1:]
        _first_stage_filters = _filter_records(first_part, exp_record_dict)
        _new_dict = _select_record_ids_from_filters(_first_stage_filters, exp_record_dict)
        _second_stage_filters = _filter_records(second_part, _new_dict)
        return _bitwise_filter_op('andcascade', _first_stage_filters, _second_stage_filters)

    elif user_range.startswith('~'):
        return _bitwise_filter_op('not', _filter_records(user_range[1:], exp_record_dict))

    base = OrderedDict((k, [False]*len(v)) for k, v in exp_record_dict.items())
    if user_range in exp_record_dict:  # User just lists an experiment
        base[user_range] = [True]*len(base[user_range])
        return base

    number_range = interpret_numbers(user_range)
    keys = list(exp_record_dict.keys())
    if number_range is not None:
        for i in number_range:
            if i>len(keys):
                raise RecordSelectionError('Experiment {} does not exist (they go from 0 to {})'.format(i, len(keys)-1))
            base[keys[i]] = [True]*len(base[keys[i]])
    elif '.' in user_range:
        exp_rec_pairs = interpret_record_identifier(user_range)
        for exp_number, rec_number in exp_rec_pairs:
            if rec_number>=len(base[keys[exp_number]]):
                raise RecordSelectionError('Selection {}.{} does not exist.'.format(exp_number, rec_number))
            base[keys[exp_number]][rec_number] = True
    elif user_range == 'old':
        for k, v in base.items():
            base[k] = ([True]*(len(v)-1)+[False]) if len(v)>0 else []
    elif user_range == 'finished':
        for k, v in base.iteritems():
            base[k] = [load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) == ExpStatusOptions.FINISHED for rec_id in exp_record_dict[k]]
    elif user_range == 'invalid':
        for k, v in base.items():
            base[k] = [load_experiment_record(rec_id).args_valid() is False for rec_id in exp_record_dict[k]]
    elif user_range == 'all':
        for k, v in base.items():
            base[k] = [True]*len(v)
    elif user_range == 'errors':
        for k, v in base.items():
            base[k] = [load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS)==ExpStatusOptions.ERROR for rec_id in exp_record_dict[k]]
    elif user_range == 'result':
        for k, v in base.items():
            base[k] = [load_experiment_record(rec_id).has_result() for rec_id in exp_record_dict[k]]
    else:
        raise RecordSelectionError("Don't know how to interpret subset '{}'".format(user_range))
    return base


class RecordSelectionError(Exception):

    pass


def _filter_experiment_record_list(user_range, experiment_record_ids):
    if user_range=='all':
        return [True]*len(experiment_record_ids)
    elif user_range=='new':
        return detect_duplicates(experiment_record_ids, key=record_id_to_experiment_id, keep_last=True)
        # return [n for n, is_old in izip_equal(get_record_ids(), old) if not old]
    elif user_range=='old':
        return [not x for x in _filter_records(user_range, 'new')]
    elif user_range=='orphans':
        orphans = []
        global_lib = get_global_experiment_library()
        for i, record_id in enumerate(experiment_record_ids):
            info = load_experiment_record(record_id).info
            if 'Module' in info:
                try:
                    import_module(info['Module'])
                    if not record_id_to_experiment_id(record_id) in global_lib:
                        orphans.append(True)
                    else:
                        orphans.append(False)
                except ImportError:
                    orphans.append(True)
            else:  # They must be old... lets kill them!
                orphans.append(True)
        return orphans
    else:
        which_ones = interpret_numbers(user_range)
        if which_ones is None:
            raise Exception('Could not interpret user range: "{}"'.format(user_range))
        filters = [False]*len(experiment_record_ids)
        for i in which_ones:
            filters[i] = True
        return filters


def select_experiment_records_from_list(user_range, experiment_records):
    return [rec_id for rec_id, f in izip_equal(experiment_records, _filter_experiment_record_list(user_range, experiment_records)) if f]


def interpret_record_identifier(user_range):
    """
    You can identify a single record with, eg 3.4, meaning "record 4 from experiment 3:.
    You can identify a range with, eg 3.1-3, meaning "records 3.1, 3.2, 3.3"
    :param user_range: The user input
    :return: A list of 2 tuples (exp_no, record_no).  e.g [(3, 4)], or [(3, 1), (3, 2), (3, 3)] in the above examples.
    """
    if ',' in user_range:
        parts = user_range.split(',')
        return [pair for p in parts for pair in interpret_record_identifier(p)]
    if '.' not in user_range:
        return None
    else:
        exp_number, record_numbers = user_range.split('.')
        return [(int(exp_number), rec_num) for rec_num in interpret_numbers(record_numbers)]


def interpret_numbers(user_range):
    """
    :param user_range: A string specifying a range of numbers.  Eg.
        interpret_numbers('4-6')==[4,5,6]
        interpret_numbers('4,6')==[4,6]
        interpret_numbers('4,6-9')==[4,6,7,8,9]
    :return: A list of integers, or None if the input is not numberic
    """
    if all(d in '0123456789-,' for d in user_range):
        numbers_and_ranges = user_range.split(',')
        numbers = [n for lst in [[int(s)] if '-' not in s else range(int(s[:s.index('-')]), int(s[s.index('-')+1:])+1) for s in numbers_and_ranges] for n in lst]
        return numbers
    else:
        return None


def run_experiment(name, exp_dict='global', **experiment_record_kwargs):
    """
    Run an experiment and save the results.  Return a string which uniquely identifies the experiment.
    You can run the experiment agin later by calling show_experiment(location_string):

    :param name: The name for the experiment (must reference something in exp_dict)
    :param exp_dict: A dict<str:func> where funcs is a function with no arguments that run the experiment.
    :param experiment_record_kwargs: Passed to ExperimentRecord.

    :return: A location_string, uniquely identifying the experiment.
    """
    if exp_dict == 'global':
        exp_dict = get_global_experiment_library()
    experiment = exp_dict[name]
    return experiment.run(**experiment_record_kwargs)


def run_experiment_ignoring_errors(name, **kwargs):
    try:
        return run_experiment(name, **kwargs)
    except Exception as err:
        traceback.print_exc()


def run_multiple_experiments(experiments, parallel = False, cpu_count=None, display_results=False, raise_exceptions=True, notes = (), run_args = {}):
    """
    Run multiple experiments, optionally in parallel with multiprocessing.

    :param experiments: A collection of experiments
    :param parallel: True to run in parallel, with multiprocessing
    :param cpu_count: If parallel, number of CPUs to use (defaults to all)
    :param raise_exceptions: Terminate exectution when one experiment fails.
    :param run_args: Other args to pass to Experiment.run()
    :return: A collection of experiment records.
    """

    if parallel:
        experiment_identifiers = [ex.get_id() for ex in experiments]
        if cpu_count is None:
            cpu_count = multiprocessing.cpu_count()
        func = run_experiment if raise_exceptions else run_experiment_ignoring_errors
        p = multiprocessing.Pool(processes=cpu_count)
        return p.map(partial(func, notes=notes, **run_args), experiment_identifiers)
    else:
        return [ex.run(raise_exceptions=raise_exceptions, display_results=display_results, notes=notes, **run_args) for ex in experiments]


def remove_common_results_prefix(results_dict):
    """
    Remove the common prefix for the results you are comparing.
    :param results_dict: An OrderedDict of experiment Results
    :return: An OrderedDict of results with the common beginnings of the keys truncated.
    """
    # TODO: Fix this so that it splits correctly, not just on '.', which is not necessarily a separator.
    assert isinstance(dict, OrderedDict), 'Expecting an OrderedDict of <experiment_name -> result>'

    split_keys = [k.split('.') for k in results_dict.keys()]
    trimmed_keys = remove_common_prefix(split_keys)
    return OrderedDict((k, v) for k, v in izip_equal(trimmed_keys, results_dict.values()))


def deprefix_experiment_ids(experiment_ids):
    """
    Given a list of experiment ids, removed the common root experiments from the list.
    :param experiment_ids: A list of experiment ids.
    :return: A list of experiment ids with the root prefix removed.
    """

    # First build dict mapping experiment_ids to their parents experiment_ids
    glib = get_global_experiment_library()
    exp_to_parent = {}
    for eid in glib.keys():
        ex = glib[eid]
        for var in ex.get_variants():
            exp_to_parent[var.get_id()] = ex.get_id()

    @memoize
    def get_experiment_tuple(exp_id):
        if exp_id in exp_to_parent:
            parent_id = exp_to_parent[exp_id]
            parent_tuple = get_experiment_tuple(parent_id)
            return parent_tuple + (exp_id[len(parent_id)+1:], )
        else:
            return (exp_id, )

    # Then for each experiment in the list,
    tuples = [get_experiment_tuple(eid) for eid in experiment_ids]
    de_prefixed_tuples = remove_common_prefix(tuples, keep_base=False)
    new_strings = ['.'+'.'.join(ex_tup) for ex_tup in de_prefixed_tuples]
    return new_strings

