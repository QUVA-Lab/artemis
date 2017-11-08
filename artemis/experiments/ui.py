import argparse
import logging
import os
import pickle
import shlex
import shutil
from collections import OrderedDict
from functools import partial
from multiprocessing import Process

from six.moves import input
from tabulate import tabulate

from artemis.experiments.experiment_management import deprefix_experiment_ids, \
    RecordSelectionError, run_multiple_experiments_with_slurm
from artemis.experiments.experiment_management import get_experient_to_record_dict
from artemis.experiments.experiment_management import (pull_experiments, select_experiments, select_experiment_records,
                                                       select_experiment_records_from_list, interpret_numbers,
                                                       run_multiple_experiments)
from artemis.experiments.experiment_record import ExpStatusOptions
from artemis.experiments.experiment_record import (get_all_record_ids, clear_experiment_records,
                                                   load_experiment_record, ExpInfoFields)
from artemis.experiments.experiment_record_view import (get_record_full_string, get_record_invalid_arg_string,
                                                        print_experiment_record_argtable, get_oneline_result_string,
                                                        compare_experiment_records)
from artemis.experiments.experiment_record_view import show_record, show_multiple_records
from artemis.experiments.experiments import load_experiment, get_nonroot_global_experiment_library
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.display import IndentPrint, side_by_side, truncate_string, surround_with_header, format_duration, format_time_stamp
from artemis.general.hashing import compute_fixed_hash
from artemis.general.mymath import levenshtein_distance
from artemis.general.should_be_builtins import all_equal, insert_at, izip_equal, separate_common_items, bad_value

try:
    import readline  # Makes input() behave like interactive shell.
    # http://stackoverflow.com/questions/15416054/command-line-in-python-with-history
except:
    pass  # readline not available


try:
    from enum import Enum
except ImportError:
    raise ImportError("Failed to import the enum package. This was added in python 3.4 but backported back to 2.4.  To install, run 'pip install --upgrade pip enum34'")


def _warn_with_prompt(message= None, prompt = 'Press Enter to continue or q then Enter to quit', use_prompt=True):
    if message:
        print(message)
    if use_prompt:
        resp = input('({}) >> '.format(prompt))
        out = resp.strip().lower()
        if out=='q':
            quit()
        else:
            return out


def browse_experiments(command=None, **kwargs):
    """
    Browse Experiments

    :param command: Optionally, a string command to pass directly to the UI.  (e.g. "run 1")
    :param root_experiment: The Experiment whose (self and) children to browse
    :param catch_errors: Catch errors that arise while running experiments
    :param close_after: Close after issuing one command.
    :param just_last_record: Only show the most recent record for each experiment.
    :param view_mode: How to view experiments {'full', 'results'} ('results' leads to a narrower display).
    :param raise_display_errors: Raise errors that arise when displaying the table (otherwise just indicate that display failed in table)
    :param run_args: A dict of named arguments to pass on to Experiment.run
    :param keep_record: Keep a record of the experiment after running.
    :param truncate_result_to: An integer, indicating the maximum length of the result string to display.
    :param cache_result_string: Cache the result string (useful when it takes a very long time to display the results
        when opening up the menu - often when results are long lists).
    """
    browser = ExperimentBrowser(**kwargs)
    browser.launch(command=command)


class ExperimentBrowser(object):

    QUIT = 'Quit'
    REFRESH = 'Refresh'
    HELP_TEXT = """
This program lists the experiments that you have defined (referenced by E#) alongside the records of console output,
plots, results, referenced by (E#.R# - for example 4.1) created by running these experiments.  Command examples:

> 4                   Run experiment 4
> run 4               Run experiment 4
> run 4-6             Run experiment 4, 5, and 6
> run all             Run all experiments
> run unfinished      Run all experiments which have not yet run to completion.
> run 4-6 -s          Run experiments 4, 5, and 6 in sequence, and catch all errors.
> run 4-6 -e          Run experiments 4, 5, and 6 in sequence, and stop on errors
> run 4-6 -p          Run experiments 4, 5, and 6 in parallel processes, and catch all errors.
> run 4-6 -p2         Run experiments 4, 5, and 6 in parallel processes, using up to 2 processes at a time.
> call 4              Call experiment 4 (like running, but doesn't save a record)
> filter 4-6          Just show experiments 4-6 and their records
> filter has:xyz      Just show experiments with "xyz" in the name and their records
> filter 1diff:3      Just show all experiments that have no more than 1 argument different from experiment 3.
> filter -            Clear the filter and show the full list of experiments
> filterrec last      Just show the last record of each experiment
> filterrec finished  Just show completed records
> filterrec ~finished Just show non-completed experiments
> filterrec finished>last   Just show the last finished runs of each experiment
> results 4-6         View the results experiments 4, 5, 6
> view results        View just the columns for experiment name and result
> view full           View all columns (the default view)
> show 4              Show the output from the last run of experiment 4 (if it has been run already).
> show 4-6            Show the output of experiments 4,5,6 together.
> records             Browse through all experiment records.
> compare 4.1,5.3     Print a table comparing the arguments and results of records 4.1 and 5.3.
> selectexp 4-6       Show the list of experiments (and their records) selected by the "experiment selector" "4-6" (see below for possible experiment selectors)
> selectrec 4-6       Show the list of records selected by the "record selector" "4-6" (see below for possible record selectors)
> sidebyside 4.1,5.3  Display the output of record from experiments 4.1,5.3 side by side.
> delete 4.1          Delete record 1 of experiment 4
> delete unfinished   Delete all experiment records that have not run to completion
> delete 4-6          Delete all records from experiments 4, 5, 6.  You will be asked to confirm the deletion.
> pull 1-4 machine1   Pulls records from experiments 1,2,3,4 from machine1 (requires some setup, see artemis.remote.remote_machines.py)
> q                   Quit.
> r                   Refresh list of experiments.
> clearcache          Clear the cached display of experiment records in the UI (caching is used only if cache_result_string==True)

Commands 'run', 'call', 'filter', 'pull', '1diff', 'selectexp' allow you to select experiments.  You can select
experiments in the following ways:

    Experiment
    Selector        Action
    --------------- ---------------------------------------------------------------------------------------------------
    4               Select experiment #4
    4-6             Select experiments 4, 5, 6
    all             Select all experiments
    unfinished      Select all experiment for which there are no records of it being run to completion.
    invalid         Select all experiments where all records were made before arguments to the experiment have changed
    has:xyz         Select all experiments with the string "xyz" in their names
    hasnot:xyz      Select all experiments without substring "xyz" in their names
    1diff:3         Select all experiments who have no more than 1 argument which is different from experiment 3's arguments.

Commands 'results', 'show', 'records', 'compare', 'sidebyside', 'selectrec', 'filterrec', 'delete' allow you to specify a range of
experiment records.  You can specify records in the following ways:

    Record
    Selector        Action
    --------------- ---------------------------------------------------------------------------------------------------
    4.2             Select record 2 for experiment 4
    4               Select all records for experiment 4
    4-6             Select all records for experiments 4, 5, 6
    4.2-5           Select records 2, 3, 4, 5 for experiment 4
    4.3,4.4         Select records 4.3, 4.4
    all             Select all records
    old             Select all records that are not the the most recent run for that experiment
    finished        Select all records that have not run to completion
    invalid         Select all records for which the arguments to their experiments have changed since they were run
    errors          Select all records that ended in error
    ~invalid        Select all records that are not invalid (the '~' can be used to negate any of the above)
    invalid|errors  Select all records that are invalid or ended in error (the '|' can be used to "or" any of the above)
    invalid&errors  Select all records that are invalid and ended in error (the '&' can be used to "and" any of the above)
    finished@last   Select the last finished record of each experiment (the '@' can be used to cascade any of the above)
"""

    def __init__(self, root_experiment = None, catch_errors = True, close_after = False, filterexp=None, filterrec = None,
            view_mode ='full', raise_display_errors=False, run_args=None, keep_record=True, truncate_result_to=100,
            ignore_valid_keys=(), cache_result_string = False, slurm_kwargs={}, remove_prefix = None, display_format='nested',
            show_args=False, catch_selection_errors=True, max_width=None, table_package = 'tabulate'):
        """
        :param root_experiment: The Experiment whose (self and) children to browse
        :param catch_errors: Catch errors that arise while running experiments
        :param close_after: Close after issuing one command.
        :param filterexp: Filter the experiments with this selection (see help for how to use)
        :param filterrec: Filter the experiment records with this selection (see help for how to use)
        :param view_mode: How to view experiments {'full', 'results'} ('results' leads to a narrower display).
        :param raise_display_errors: Raise errors that arise when displaying the table (otherwise just indicate that display failed in table)
        :param run_args: A dict of named arguments to pass on to Experiment.run
        :param keep_record: Keep a record of the experiment after running.
        :param truncate_result_to: An integer, indicating the maximum length of the result string to display.
        :param ignore_valid_keys: When checking whether arguments are valid, ignore arguments with names in this list
        :param cache_result_string: Cache the result string (useful when it takes a very long time to display the results
            when opening up the menu - often when results are long lists).
        :param slurm_kwargs:
        :param remove_prefix: Remove the common prefix on the experiment ids in the display.
        :param display_format: How experements and their records are displayed: 'nested' or 'flat'.  'nested' might be
            better for narrow console outputs.
        """

        if run_args is None:
            run_args = {}
        if 'keep_record' not in run_args:
            run_args['keep_record'] = keep_record
        if remove_prefix is None:
            remove_prefix = display_format=='flat'
        self.root_experiment = root_experiment
        self.close_after = close_after
        self.catch_errors = catch_errors
        self.exp_record_dict = None
        self.raise_display_errors = raise_display_errors
        self.view_mode = view_mode
        self._filter = filterexp
        self._filterrec = filterrec
        self.run_args = {} if run_args is None else run_args
        self.truncate_result_to = truncate_result_to
        self.cache_result_string = cache_result_string
        self.ignore_valid_keys = ignore_valid_keys
        self.slurm_kwargs = slurm_kwargs
        self.remove_prefix = remove_prefix
        self.display_format = display_format
        self.show_args = show_args
        self.catch_selection_errors = catch_selection_errors
        self.max_width = max_width
        self.table_package = table_package

    def _reload_record_dict(self):
        names = get_nonroot_global_experiment_library().keys()
        if self.root_experiment is not None:
            # We could just go [ex.name for ex in self.root_experiment.get_all_variants(include_self=True)]
            # but we want to preserve the order in which experiments were created
            descendents_of_root = set(ex.name for ex in self.root_experiment.get_all_variants(include_self=True))
            names = [name for name in names if name in descendents_of_root]
        all_experiments = get_experient_to_record_dict(names)
        return all_experiments

    def _filter_record_dict(self, all_experiments):
        # Apply filters and display Table:
        if self._filter is not None:
            try:
                all_experiments = OrderedDict((exp_name, all_experiments[exp_name]) for exp_name in select_experiments(self._filter, all_experiments))
            except RecordSelectionError as err:
                old_filter = self._filter
                self._filter = None
                raise RecordSelectionError("Failed to apply experiment filter: '{}' because {}.  Removing filter.".format(old_filter, err))
        if self._filterrec is not None:
            try:
                all_experiments = select_experiment_records(self._filterrec, all_experiments, load_records=False, flat=False)
            except RecordSelectionError as err:
                old_filterrec = self._filterrec
                self._filterrec = None
                raise RecordSelectionError("Failed to apply record filter: '{}' because {}.  Removing filter.".format(old_filterrec, err))
        return all_experiments

    def launch(self, command=None):

        func_dict = {
            'run': self.run,
            'test': self.test,
            'show': self.show,
            'call': self.call,
            'kill': self.kill,
            'selectexp': self.selectexp,
            'selectrec': self.selectrec,
            'view': self.view,
            'h': self.help,
            'filter': self.filter,
            'filterrec': self.filterrec,
            'displayformat': self.displayformat,
            'explist': self.explist,
            'sidebyside': self.side_by_side,
            'argtable': self.argtable,
            'compare': self.compare,
            'delete': self.delete,
            'errortrace': self.errortrace,
            'q': self.quit,
            'records': self.records,
            'pull': self.pull,
            'clearcache': clear_ui_cache,
            }

        display_again = True
        while True:

            if display_again:
                    all_experiments = self._reload_record_dict()
                    try:
                        self.exp_record_dict = self._filter_record_dict(all_experiments)
                    except RecordSelectionError as err:
                        _warn_with_prompt(str(err), use_prompt=self.catch_selection_errors)
                        if not self.catch_selection_errors:
                            raise
                        else:
                            continue
                    print(self.get_experiment_list_str(self.exp_record_dict))
                    if self._filter is not None or self._filterrec is not None:
                        print('[Showing {}/{} experiments and {}/{} records after Experiment Filter: "{}" and Record Filter: "{}"]'.format(
                            len(self.exp_record_dict), len(all_experiments),
                            sum(len(v) for v in self.exp_record_dict.values()), sum(len(v) for v in all_experiments.values()),
                            self._filter, self._filterrec
                            ))

            # Get command from user
            if command is None:
                user_input = input('Enter command or experiment # to run (h for help) >> ').strip()
            else:
                user_input = command
                command = None

            display_again = True
            out = None
            # Respond to user input
            with IndentPrint():
                try:
                    split = user_input.split(' ')
                    if len(split)==0:
                        continue
                    cmd = split[0]
                    args = split[1:]

                    if cmd in ('', 'r'): # Refresh
                        continue
                    elif interpret_numbers(cmd) is not None:  # If you've listed a number, you implicitely call run
                        args = [cmd]+args
                        cmd = 'run'

                    if cmd in func_dict:
                        out = func_dict[cmd](*args)
                        display_again = False
                    else:
                        edit_distances = [levenshtein_distance(cmd, other_cmd) for other_cmd in func_dict.keys()]
                        min_distance = min(edit_distances)
                        closest = func_dict.keys()[edit_distances.index(min_distance)]
                        suggestion = ' Did you mean "{}"?.  '.format(closest) if min_distance<=2 else ''
                        if self.close_after:
                            raise Exception('Unrecognised command: "{}"'.format(cmd))
                        else:
                            print('Unrecognised command: "{}". {}'.format(cmd, suggestion))
                        display_again = False
                    if out is self.QUIT or self.close_after:
                        break
                    elif out is self.REFRESH:
                        display_again = True

                except Exception as err:
                    print ("CAUGHT: {}".format(err))
                    if self.catch_errors or (isinstance(err, RecordSelectionError) and self.catch_selection_errors):
                        res = input('Enter "e" to view the stacktrace, or anything else to continue.')
                        if res.strip().lower() == 'e':
                            raise
                        display_again = True
                    else:
                        raise

    def get_experiment_list_str(self, exp_record_dict):
        headers = {
            'full': [ExpRecordDisplayFields.RUNS, ExpRecordDisplayFields.DURATION, ExpRecordDisplayFields.STATUS, ExpRecordDisplayFields.ARGS_CHANGED, ExpRecordDisplayFields.RESULT_STR, ExpRecordDisplayFields.NOTES],
            'results': [ExpRecordDisplayFields.RESULT_STR]
            }[self.view_mode]

        if self.remove_prefix:
            deprefixed_ids = deprefix_experiment_ids(exp_record_dict.keys())
            exp_record_dict = OrderedDict((k, v) for k, v in zip(deprefixed_ids, exp_record_dict.values()))

        row_func = _get_record_rows_cached if self.cache_result_string else _get_record_rows
        header_names = [h.value for h in headers]

        def remove_notes_if_no_notes(_record_rows, _record_headers):
            notes_column_index = _record_headers.index(ExpRecordDisplayFields.NOTES.value) if ExpRecordDisplayFields.NOTES.value in _record_headers else None
            # Remove the notes column if there are no notes!
            if notes_column_index is not None and all(row[notes_column_index]=='' or row[notes_column_index]=='-' for row in _record_rows):
                new_rows = []
                for row in _record_rows:
                    new_rows.append(row[:notes_column_index]+row[notes_column_index+1:])
                new_headers = _record_headers[:notes_column_index]+_record_headers[:notes_column_index+1:]
            else:
                new_rows = _record_rows
                new_headers = _record_headers
            return new_rows, new_headers

        if self.display_format=='nested':  # New Display mode

            if self.show_args:
                _, argdiff = separate_common_items([load_experiment(ex).get_args().items() for ex in exp_record_dict])
                argdiff = {k: args for k, args in izip_equal(exp_record_dict.keys(), argdiff)}
            # Build a list of experiments and a list of records.
            full_headers = ['#']+header_names
            record_rows = []
            experiment_rows = []
            experiment_row_ixs = []
            counter = 1  # Start at 2 because record table has the headers.
            for i, (exp_id, record_ids) in enumerate(exp_record_dict.items()):
                experiment_row_ixs.append(counter)
                exp_identifier = exp_id if not self.show_args else ','.join('{}={}'.format(k, v) for k, v in argdiff[exp_id])
                experiment_rows.append([i, exp_identifier])
                for j, record_id in enumerate(record_ids):
                    record_rows.append([j]+row_func(record_id, headers, raise_display_errors=self.raise_display_errors, truncate_to=self.truncate_result_to, ignore_valid_keys=self.ignore_valid_keys))
                    counter+=1
            record_rows, full_headers = remove_notes_if_no_notes(record_rows, full_headers)
            # Merge the experiments table and record table.

            if self.table_package=='tabulate':
                record_table_rows = tabulate(record_rows, headers=full_headers, tablefmt="pipe").split('\n')
                del record_table_rows[1]  # Get rid of that silly line.
                experiment_table_rows = tabulate(experiment_rows, numalign='left').split('\n')[1:-1]  # First and last are just borders
                longest_row = max(max(len(r) for r in record_table_rows), max(len(r) for r in experiment_table_rows)+4) if len(record_table_rows)>0 else 0
                record_table_rows = [r if len(r)==longest_row else r[:-1] + ' '*(longest_row-len(r)) + r[-1] for r in record_table_rows]
                experiment_table_rows = [('=' if i==0 else '-')*longest_row+'\n'+r + ' '*(longest_row-len(r)-1)+'|' for i, r in enumerate(experiment_table_rows)]
                all_rows = [surround_with_header('Experiments', width=longest_row, char='=')] + insert_at(record_table_rows, experiment_table_rows, indices=experiment_row_ixs) + ['='*longest_row]
                table = '\n'.join(all_rows)
            else:
                raise NotImplementedError(self.table_package)

        elif self.display_format=='flat':  # Display First record on same row
            full_headers = ['E#', 'R#', 'Experiment']+header_names
            rows = []
            for i, (exp_id, record_ids) in enumerate(exp_record_dict.items()):
                if len(record_ids)==0:
                    rows.append([str(i), '', exp_id, '<No Records>'] + ['-']*(len(headers)-1))
                else:
                    for j, record_id in enumerate(record_ids):
                        rows.append([str(i) if j==0 else '', j, exp_id if j==0 else '']+row_func(record_id, headers, raise_display_errors=self.raise_display_errors, truncate_to=self.truncate_result_to, ignore_valid_keys=self.ignore_valid_keys))
            assert len(rows[0])==len(full_headers)
            rows, full_headers = remove_notes_if_no_notes(rows, full_headers)

            if self.table_package == 'pretty_table':
                from prettytable.prettytable import PrettyTable
                table = str(PrettyTable(rows, field_names=full_headers, align='l', max_table_width=self.max_width))
            elif self.table_package == 'tabulate':
                table = tabulate(rows, headers=full_headers)
            else:
                raise NotImplementedError(self.table_package)
        else:
            raise NotImplementedError(self.display_format)

        return table

    def run(self, *args):

        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', help='A selection of experiments to run.  Examples: "3" or "3-5", or "3,4,5"')
        parser.add_argument('-p', '--parallel', default=False, nargs='*')
        parser.add_argument('-n', '--note')
        parser.add_argument('-e', '--raise_errors', default='single', nargs='*', help='By default, error are only raised if a single experiment is run.  Set "-e" to always rays errors.  "-e 0" to never raise errors.')
        parser.add_argument('-d', '--display_results', default=False, action = "store_true")
        parser.add_argument('-s', '--slurm', default=False, action = "store_true", help='Run with slurm')
        args = parser.parse_args(args)

        n_processes = \
            None if args.parallel is False else \
            'all' if len(args.parallel)==0 else \
            int(args.parallel[0]) if len(args.parallel)==1 else \
            bad_value(args.parallel, '-p can have 0 or 1 arguments.  Got: {}'.format(args.parallel))

        ids = select_experiments(args.user_range, self.exp_record_dict)

        # Raise errors if:
        # -e
        # -e 1
        # No arg, and only 1 experiment running
        raise_errors = (len(args.raise_errors)==0 or (len(args.raise_errors)==1 and args.raise_errors[0]=='1') or (args.raise_errors == 'single' and len(ids)==1))

        if args.slurm:
            run_multiple_experiments_with_slurm(
                experiments=[load_experiment(eid) for eid in ids],
                n_parallel = n_processes,
                raise_exceptions = raise_errors,
                run_args=self.run_args,
                slurm_kwargs=self.slurm_kwargs
                )
        else:
            exp_names = list(self.exp_record_dict.keys())
            run_multiple_experiments(
                experiments=[load_experiment(eid) for eid in ids],
                prefixes=[exp_names.index(eid) for eid in ids],
                parallel=n_processes,
                raise_exceptions = raise_errors,
                run_args=self.run_args,
                notes=(args.note, ) if args.note is not None else (),
                display_results=args.display_results
                )

        result = _warn_with_prompt('Finished running {} experiment{}.'.format(len(ids), '' if len(ids)==1 else 's'),
                use_prompt=not self.close_after,
                prompt='Press Enter to Continue, or "q" then Enter to Quit')
        if result=='q':
            quit()

    def test(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier).test()

    def help(self):
        _warn_with_prompt(self.HELP_TEXT, prompt = 'Press Enter to exit help.', use_prompt=not self.close_after)

    def show(self, *args):
        """
        :param user_range:  A range specifying the record
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', help='A selection of experiment records to show. ')
        parser.add_argument('-l', '--last', action='store_true', help='Just select the last record from the list of experiments.')
        parser.add_argument('-r', '--results', default=False, action = "store_true", help="Only show records with results.")
        parser.add_argument('-o', '--original', default=False, action = "store_true", help="Use the original default show function: show_record")
        parser.add_argument('-p', '--process', default=False, action = "store_true", help="Launch show in new process")
        args = parser.parse_args(args)

        user_range = args.user_range
        if args.results:
            user_range += '@result'
        if args.last:
            user_range += '@last'

        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)

        func = show_record if args.original else None
        if args.process:
            Process(target=partial(show_multiple_records, records, func)).start()
        else:
            show_multiple_records(records, func)
        _warn_with_prompt(use_prompt=False)

    def compare(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', help='A selection of experiment records to compare.  Examples: "3" or "3-5", or "3,4,5"')
        parser.add_argument('-l', '--last', default=False, action = "store_true", help="Use this flag if you want to select Experiments instead of Experiment Records, and just show the last completed.")
        parser.add_argument('-r', '--results', default=False, action = "store_true", help="Only compare records with results.")
        parser.add_argument('-o', '--original', default=False, action = "store_true", help="Use original compare funcion")

        args = parser.parse_args(args)

        user_range = args.user_range if not args.last else args.user_range + '@result@last'
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        if args.results:
            records = [rec for rec in records if rec.has_result()]
        if len(records)==0:
            raise RecordSelectionError('No records were selected with "{}"'.format(args.user_range))

        if args.original:
            func = compare_experiment_records
        else:
            compare_funcs = [rec.get_experiment().compare for rec in records]
            assert all_equal(compare_funcs), "Your records have different comparison functions - {} - so you can't compare them".format(set(compare_funcs))
            func = compare_funcs[0]

        # The following could be used to launch comparisons in a  new process.  We don't do this now because
        # comparison function often use matplotlib, and matplotlib's Tkinter backend hangs when trying to create
        # a new figure in a new thread.
        # thread = Process(target = partial(func, records))
        # thread.start()
        # thread.join()

        func(records)
        _warn_with_prompt(use_prompt=False)

    def displayformat(self, new_format):
        assert new_format in ('nested', 'flat'), "Display format must be 'nested' or 'flat', not '{}'".format(new_format)
        self.display_format = new_format
        return ExperimentBrowser.REFRESH

    def errortrace(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint("Error Traces:", show_line=True, show_end=True):
            for record in records:
                with IndentPrint(record.get_id(), show_line=True):
                    print(record.get_error_trace())
        _warn_with_prompt(use_prompt=False)

    def delete(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print('{} out of {} Records will be deleted.'.format(len(records), sum(len(recs) for recs in self.exp_record_dict.values())))
        with IndentPrint():
            print(ExperimentRecordBrowser.get_record_table(records, ))
        response = input('Type "yes" to continue. >').strip().lower()
        if response == 'yes':
            clear_experiment_records([record.get_id() for record in records])
            print('Records deleted.')
            return ExperimentBrowser.REFRESH
        else:
            _warn_with_prompt('Records were not deleted.', use_prompt=False)

    def call(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier).call()

    def selectexp(self, user_range):
        exps_to_records = select_experiments(user_range, self.exp_record_dict, return_dict=True)
        with IndentPrint():
            print(self.get_experiment_list_str(exps_to_records))
        _warn_with_prompt('Experiment Selection "{}" includes {} out of {} experiments.'.format(user_range, len(exps_to_records), len(self.exp_record_dict)), use_prompt=not self.close_after)

    def selectrec(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint():
            print(ExperimentRecordBrowser.get_record_table(records))
        _warn_with_prompt('Record Selection "{}" includes {} out of {} records.'.format(user_range, len(records), sum(len(recs) for recs in self.exp_record_dict.values())), use_prompt=not self.close_after)

    def side_by_side(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print(side_by_side([get_record_full_string(rec) for rec in records], max_linewidth=128))
        _warn_with_prompt(use_prompt=False)

    def argtable(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', nargs = '?', default='all', help='A selection of experiment records to run.  Examples: "3" or "3-5", or "3,4,5"')
        args = parser.parse_args(args)
        records = select_experiment_records(args.user_range, self.exp_record_dict, flat=True)
        print_experiment_record_argtable(records)
        _warn_with_prompt(use_prompt=False)

    def records(self, ):
        browse_experiment_records(self.exp_record_dict.keys())

    def pull(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', help='A selection of experiments whose records to pull.  Examples: "3" or "3-5", or "3,4,5"')
        parser.add_argument('machine_name', action='store', nargs = '?', default='all', help='Name of machine to pull from (must be listed in ~/.artemisrc)')
        # Following -p thing is temporary until we figure out how to deal with password only if needed
        parser.add_argument('-p', '--need_password', action='store_true', default=False, help='Put this flag if you need a password (leave it out if you have keys set up)')
        args = parser.parse_args(args)
        from artemis.remote.remote_machines import get_remote_machine_info
        info = get_remote_machine_info(args.machine_name)
        exp_names = select_experiments(args.user_range, self.exp_record_dict)
        output = pull_experiments(user=info['username'], ip=info['ip'], experiment_names=exp_names, include_variants=False, need_pass=args.need_password)
        print(output)
        return ExperimentBrowser.REFRESH

    def kill(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('user_range', action='store', help='A selection of experiments whose records to pull.  Examples: "3" or "3-5", or "3,4,5"')
        parser.add_argument('-s', '--skip', action='store_true', default=True, help='Skip the check that all selected records are currently running (just filter running ones)')
        args = parser.parse_args(args)

        records = select_experiment_records(args.user_range, self.exp_record_dict, flat=True)

        if not args.skip and (not all(record.get_status() is ExpStatusOptions.STARTED for record in records)):
            raise RecordSelectionError('Not all records you selected to kill were running: \n- {}'.format('\n- '.join('{}: {}'.format(rec.get_id(), rec.get_status()) for rec in records)))
        elif args.skip:
            records = [rec for rec in records if rec.get_status() is ExpStatusOptions.STARTED]

        if len(records)==0:
            raise RecordSelectionError('Selection "{}" selected no active processes to kill'.format(args.user_range))

        print('{} Running Experiments will be killed.'.format(len(records)))
        with IndentPrint():
            print(ExperimentRecordBrowser.get_record_table(records, ))
        response = input('Type "yes" to continue. >').strip().lower()
        if response == 'yes':
            for rec in records:
                rec.kill()
            print('Experiments killed.')
            return ExperimentBrowser.REFRESH
        else:
            _warn_with_prompt('Experiments were not killed.', use_prompt=False)

    def filter(self, user_range):
        self._filter = user_range if user_range not in ('-', '--clear') else None
        return ExperimentBrowser.REFRESH

    def filterrec(self, user_range):
        self._filterrec = user_range if user_range not in ('-', '--clear') else None
        return ExperimentBrowser.REFRESH

    def view(self, mode):
        self.view_mode = mode
        return ExperimentBrowser.REFRESH

    def explist(self, surround = ""):
        print("\n".join([surround+k+surround for k in self.exp_record_dict.keys()]))
        _warn_with_prompt(use_prompt=False)

    def quit(self):
        return ExperimentBrowser.QUIT


class ExpRecordDisplayFields(Enum):
    RUNS = 'Start Time'
    DURATION = 'Duration'
    STATUS = 'Status'
    ARGS_CHANGED = 'Args Changed?'
    RESULT_STR = 'Result'
    NOTES = 'Notes'


def _show_notes(rec):
    if rec.info.has_field(ExpInfoFields.NOTES):
        notes = rec.info.get_field(ExpInfoFields.NOTES)
        if notes is None:
            return ''
        else:
            return ';'.join(rec.info.get_field(ExpInfoFields.NOTES)).replace('\n', ';;')
    else:
        return ''


class _DisplaySettings(object):

    SETTINGS = {'ignore_valid_keys', ()}

    def __init__(self, settings_dict):
        self.settings_dict = settings_dict

    def __enter__(self):
        self.old_settings = _DisplaySettings.SETTINGS
        _DisplaySettings.SETTINGS = self.settings_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        _DisplaySettings.SETTINGS = self.old_settings

    @classmethod
    def get_setting(cls, name):
        return _DisplaySettings.SETTINGS[name]


_exp_record_field_getters = {
    ExpRecordDisplayFields.RUNS: lambda rec: format_time_stamp(rec.info.get_field(ExpInfoFields.TIMESTAMP)),
    ExpRecordDisplayFields.DURATION: lambda rec: format_duration(rec.info.get_field(ExpInfoFields.RUNTIME)) if rec.info.has_field(ExpInfoFields.RUNTIME) else '-',
    ExpRecordDisplayFields.ARGS_CHANGED: lambda rec: get_record_invalid_arg_string(rec, ignore_valid_keys=_DisplaySettings.get_setting('ignore_valid_keys')),
    ExpRecordDisplayFields.RESULT_STR: get_oneline_result_string,
    ExpRecordDisplayFields.STATUS: lambda rec: rec.info.get_field_text(ExpInfoFields.STATUS),
    ExpRecordDisplayFields.NOTES: _show_notes
}


def _get_record_rows(record_id, headers, raise_display_errors, truncate_to, ignore_valid_keys = ()):
    rec = load_experiment_record(record_id)

    with _DisplaySettings(dict(ignore_valid_keys=ignore_valid_keys)):
        if not raise_display_errors:
            values = []
            for h in headers:
                try:
                    values.append(_exp_record_field_getters[h](rec))
                except:
                    values.append('<Display Error>')
        else:
            values = [_exp_record_field_getters[h](rec) for h in headers]

    if truncate_to is not None:
        values = [truncate_string(val, truncation=truncate_to, message='...') for val in values]

    return values


def clear_ui_cache():
    shutil.rmtree(get_artemis_data_path('_ui_cache/'))


def _get_record_rows_cached(record_id, headers, raise_display_errors, truncate_to, ignore_valid_keys = ()):
    """
    We want to load the saved row only if:
    - The record is complete
    -
    :param record_id:
    :param headers:
    :return:
    """
    cache_key = compute_fixed_hash((record_id, [h.value for h in headers], truncate_to, ignore_valid_keys))
    path = get_artemis_data_path(os.path.join('_ui_cache', cache_key), make_local_dir=True)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                record_rows = pickle.load(f)
            if len(record_rows)!=len(headers):
                os.remove(path)  # This should never happen.  But in case it somehow does, we just go ahead and compute again.
            else:
                return record_rows
        except:
            logging.warn('Failed to load cached record info: {}'.format(record_id))

    info_plus_status = _get_record_rows(record_id=record_id, headers=headers+[ExpRecordDisplayFields.STATUS],
                                        raise_display_errors=raise_display_errors, truncate_to=truncate_to, ignore_valid_keys=ignore_valid_keys)
    record_rows, status = info_plus_status[:-1], info_plus_status[-1]
    if status == ExpStatusOptions.STARTED:  # In this case it's still running (maybe) and we don't want to cache because it'll change
        return record_rows
    else:
        with open(path, 'wb') as f:
            pickle.dump(record_rows, f)
        return record_rows


def browse_experiment_records(*args, **kwargs):
    """
    Browse through experiment records.

    :param names: Filter by names of experiments
    :param filter_text: Filter by regular expression
    :return:
    """

    experiment_record_browser = ExperimentRecordBrowser(*args, **kwargs)
    experiment_record_browser.launch()


class ExperimentRecordBrowser(object):

    QUIT = 'Quit'
    HELP_TEXT = """
    q:                  Quit
    r:                  Refresh
    filter <text>       filter experiments
    viewfilters         View all filters on these results
    showall:            Show all experiments ever
    allnames:           Remove any name filters
    show <number>       Show experiment with number
    side_by_side 4,6,9       Compare experiments by their numbers.
    clearall            Delete all experements from your computer
"""

    def __init__(self, experiment_names = None, filter_text = None, raise_display_errors=False):
        """
        Browse through experiment records.

        :param names: Filter by names of experiments
        :param filter_text: Filter by regular expression
        :return:
        """
        self.experiment_names = experiment_names
        self.filters = [filter_text]
        self.record_ids = self.reload_ids()
        self.raise_display_errors = raise_display_errors

    def reload_ids(self):
        return get_all_record_ids(experiment_ids= self.experiment_names, filters=self.filters)

    @staticmethod
    def get_record_table(records = None, headers = ('#', 'Identifier', 'Start Time', 'Duration', 'Status', 'Valid', 'Notes', 'Result'), raise_display_errors = False, result_truncation=100):

        d = {
            '#': lambda: i,
            'Identifier': lambda: experiment_record.get_id(),
            'Start Time': lambda: experiment_record.info.get_field_text(ExpInfoFields.TIMESTAMP, replacement_if_none='?'),
            'Duration': lambda: experiment_record.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
            'Status': lambda: experiment_record.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
            'Args': lambda: experiment_record.info.get_field_text(ExpInfoFields.ARGS, replacement_if_none='?'),
            'Valid': lambda: get_record_invalid_arg_string(experiment_record, note_version='short'),
            'Notes': lambda: experiment_record.info.get_field_text(ExpInfoFields.NOTES, replacement_if_none='?'),
            'Result': lambda: get_oneline_result_string(experiment_record, truncate_to=128)
            # experiment_record.get_experiment().get_oneline_result_string(truncate_to=result_truncation) if is_experiment_loadable(experiment_record.get_experiment_id()) else '<Experiment not loaded>'
            }

        def get_col_info(headers):
            info = []
            for h in headers:
                try:
                    info.append(d[h]())
                except:
                    info.append('<Error displaying info>')
                    if raise_display_errors:
                        raise
            return info

        rows = []
        for i, experiment_record in enumerate(records):
            rows.append(get_col_info(headers))
        assert all_equal([len(headers)] + [len(row) for row in rows]), 'Header length: {}, Row Lengths: \n {}'.format(len(headers), [len(row) for row in rows])
        return tabulate(rows, headers=headers)

    def launch(self):

        func_lookup = {
            'q': self.quit,
            'h': self.help,
            'filter': self.filter,
            'showall': self.showall,
            'args': self.args,
            'rmfilters': self.rmfilters,
            'viewfilters': self.viewfilters,
            'side_by_side': self.compare,
            'show': self.show,
            'search': self.search,
            'delete': self.delete,
        }

        while True:

            print("=============== Experiment Records ==================")
            # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
            print(self.get_record_table([load_experiment_record(rid) for rid in self.record_ids], raise_display_errors = self.raise_display_errors))
            print('-----------------------------------------------------')

            if self.experiment_names is not None or len(self.filters) != 0:
                print('Not showing all experiments.  Type "showall" to see all experiments, or "viewfilters" to view current filters.')
            user_input = input('Enter Command (show # to show and experiment, or h for help) >>').strip()
            parts = shlex.split(user_input)
            if len(parts)==0:
                print("You need to specify an command.  Press h for help.")
                continue
            cmd = parts[0]
            args = parts[1:]

            try:
                if cmd not in func_lookup:
                    raise _warn_with_prompt('Unknown Command: {}'.format(cmd))
                else:
                    return_val = func_lookup[cmd](*args)
                    if return_val==self.QUIT:
                        break
            except Exception as e:
                res = input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
                if res.strip().lower() == 'e':
                    raise

    def _select_records(self, user_range):
        return select_experiment_records_from_list(user_range, self.record_ids)

    def quit(self):
        return self.QUIT

    def help(self):
        _warn_with_prompt(self.HELP_TEXT)

    def filter(self, filter_text):
        self.filters.append(filter_text)
        self.record_ids = self.reload_ids()

    def showall(self):
        self.filters = []
        self.experiment_names = None
        self.record_ids = self.reload_ids()

    def args(self, user_range):
        print(self.get_record_table(self._select_records(user_range), headers=['Identifier', 'Args']))

    def rmfilters(self):
        self.filters = []
        self.record_ids = self.reload_ids()

    def viewfilters(self):
        _warn_with_prompt('Filtering for: \n  Names in {}\n  Expressions: {}'.format(self.experiment_names, self.filters))

    def compare(self, user_range):
        identifiers = self._select_records(user_range)
        print_experiment_record_argtable(identifiers)
        _warn_with_prompt('')

    def show(self, user_range):
        record_ids = self._select_records(user_range)
        show_multiple_records([load_experiment_record(rid) for rid in record_ids])
        _warn_with_prompt('')

    def search(self, filter_text):
        print('Found the following Records: ')
        print(self.get_record_table([rid for rid in self.record_ids if filter_text in rid]))
        _warn_with_prompt()

    def delete(self, user_range):
        ids = self._select_records(user_range)
        print('We will delete the following experiments:')
        print(self.get_record_table(ids))
        conf = input("Going to clear {} of {} experiment records shown above.  Enter 'yes' to confirm: ".format(len(ids), len(self.record_ids)))
        if conf.strip().lower() == 'yes':
            clear_experiment_records(ids=ids)
        else:
            _warn_with_prompt("Did not delete experiments")
        self.record_ids = self.reload_ids()


if __name__ == '__main__':
    browse_experiment_records()