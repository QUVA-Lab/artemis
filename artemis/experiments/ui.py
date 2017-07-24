import shlex
from collections import OrderedDict
from importlib import import_module
from artemis.experiments.experiment_record import GLOBAL_EXPERIMENT_LIBRARY, get_all_record_ids, clear_experiment_records, \
    ExperimentRecord, run_experiment_ignoring_errors, \
    experiment_id_to_record_ids, load_experiment_record, load_experiment, record_id_to_experiment_id, \
    record_id_to_timestamp, ExpInfoFields, ExpStatusOptions, has_experiment_record, NoSavedResultError
from artemis.general.display import IndentPrint, side_by_side
from artemis.general.should_be_builtins import separate_common_items, bad_value, detect_duplicates, \
    izip_equal, all_equal
from artemis.general.tables import build_table
from tabulate import tabulate
from functools import partial
import re

def _setup_input_memory():
    try:
        import readline  # Makes raw_input behave like interactive shell.
        # http://stackoverflow.com/questions/15416054/command-line-in-python-with-history
    except:
        pass  # readline not available


def _warn_with_prompt(message= None, prompt = 'Press Enter to continue', use_prompt=True):
    if message is not None:
        print message
    if use_prompt:
        raw_input('({}) >> '.format(prompt))

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


def browse_experiments(root_experiment = None, catch_errors = False, close_after = False, just_last_record=False, raise_display_errors = False, run_args = None, keep_record = True, command=None):
    """
    Browse Experiments

    :param root_experiment: Optionally, the root experiment to look at.
    :param catch_errors: True if you want to catch any errors here
    :param close_after: Close this menu after running an experiment
    :param just_last_record: Just show the last record for the experiment
    """

    if run_args is None:
        run_args = {}
    if 'keep_record' not in run_args:
        run_args['keep_record'] = keep_record

    browser = ExperimentBrowser(root_experiment=root_experiment, catch_errors=catch_errors, close_after=close_after,
        just_last_record=just_last_record, raise_display_errors=raise_display_errors, run_args=run_args)
    browser.launch(command=command)


class ExperimentBrowser(object):

    QUIT = 'Quit'
    HELP_TEXT = """
This program lists the experiments that you have defined (referenced by E#) alongside the records of console output,
plots, results, referenced by (E#.R# - for example 4.1) created by running these experiments.  Command examples:

> 4                   Run experiment 4
> run 4               Run experiment 4
> run 4-6             Run experiment 4, 5, and 6
> run all             Run all experiments
> run 4-6 -s          Run experiments 4, 5, and 6 in sequence, and catch all errors.
> run 4-6 -e          Run experiments 4, 5, and 6 in sequence, and stop on errors
> run 4-6 -p          Run experiments 4, 5, and 6 in parallel processes, and catch all errors.
> call 4              Call experiment 4 (like running, but doesn't save a record)
> filter 4-6          Just show experiments 4-6 and their records
> filter has:xyz      Just show experiments with "xyz" in the name and their records
> filter --clear      Clear all filters and show the full list of experiments
> results 4-6         View the results experiments 4, 5, 6
> view results        View just the columns for experiment name and result
> view full           View all columns (the default view)
> show 4              Show the output from the last run of experiment 4 (if it has been run already).
> records             Browse through all experiment records.
> allruns             Toggle between showing all past runs of each experiment, and just the last one.
> compare 4.1,5.3     Print a table comparing the arguments and results of records 4.1 and 5.3.
> select 4-6          Show the list of records belonging to experiments 4, 5, 6
> sidebyside 4.1,5.3  Display the output of record from experiments 4.1,5.3 side by side.
> delete 4-6          Delete all records from experiments 4, 5, 6.  You will be asked to confirm the deletion.
> q                   Quit.
> r                   Refresh list of experiments.

Commands 'run', 'call', 'filter' allow you to select experiments.  You can select experiments in the following ways:
    4               Select experiment #4
    4-6             Select experiments 4, 5, 6
    all             Select all experiments
    unfinished      Select all experiment for which there are no records of it being run to completion.
    invalid         Select all experiments where all records were made before arguments to the experiment have changed
    has:xyz         Select all experiments with the string "xyz" in their names
    hasnot:xyz      Select all experiments without substring "xyz" in their names

Commands 'results', 'show', 'records', 'compare', 'sidebyside', 'select', 'delete' allow you to specify a range of experiment
records.  You can specify records in the following ways:
    4.2             Select record 2 for experiment 4
    4               Select all records for experiment 4
    4-6             Select all records for experiments 4, 5, 6
    4.2-5           Select records 2, 3, 4, 5 for experiment 4
    4.3,4.4         Select records 4.3, 4.4
    all             Select all records
    old             Select all records that are not the the most recent run for that experiment
    unfinished      Select all records that have not run to completion
    invalid         Select all records for which the arguments to their experiments have changed since they were run
    errors          Select all records that ended in error
    invalid|errors  Select all records that are invalid or ended in error (the '|' can be used to "or" any of the above)
    invalid&errors  Select all records that are invalid and ended in error (the '&' can be used to "and" any of the above)
"""

    def __init__(self, root_experiment = None, catch_errors = False, close_after = True, just_last_record = False, view_mode ='full', raise_display_errors=False, run_args=None):

        self.root_experiment = root_experiment
        self.close_after = close_after
        self.just_last_record = just_last_record
        self.catch_errors = catch_errors
        self.exp_record_dict = self.reload_record_dict()
        self.raise_display_errors = raise_display_errors
        self.view_mode = view_mode
        self._filter = None
        self.run_args = {} if run_args is None else run_args

    def reload_record_dict(self):
        names = GLOBAL_EXPERIMENT_LIBRARY.keys() if self.root_experiment is None else [ex.name for ex in self.root_experiment.get_all_variants(include_self=True)]

        d= OrderedDict((name, experiment_id_to_record_ids(name)) for name in names)
        if self.just_last_record:
            for k in d.keys():
                d[k] = [d[k][-1]] if len(d[k])>0 else []
        return d

    def launch(self, command=None):

        func_dict = {
            'run': self.run,
            'test': self.test,
            'show': self.show,
            'call': self.call,
            'select': self.select,
            'allruns': self.allruns,
            'view': self.view,
            'h': self.help,
            'results': self.results,
            'filter': self.filter,
            'explist': self.explist,
            'sidebyside': self.side_by_side,
            'compare': self.compare,
            'compare_results': self.compare_results,
            'delete': self.delete,
            'errortrace': self.errortrace,
            'q': self.quit,
            'records': self.records
            }

        while True:
            all_experiments = self.reload_record_dict()

            print "==================== Experiments ===================="
            self.exp_record_dict = all_experiments if self._filter is None else \
                OrderedDict((exp_name, all_experiments[exp_name]) for exp_name in select_experiments(self._filter, all_experiments))
            print self.get_experiment_list_str(self.exp_record_dict, just_last_record=self.just_last_record, view_mode=self.view_mode, raise_display_errors=self.raise_display_errors)
            if self._filter is not None:
                print '[Filtered with "{}" to show {}/{} experiments]'.format(self._filter, len(self.exp_record_dict), len(all_experiments))
            print '-----------------------------------------------------'
            if command is None:
                user_input = raw_input('Enter command or experiment # to run (h for help) >> ').lstrip(' ').rstrip(' ')
            else:
                user_input=command
                command = None
            with IndentPrint():
                try:
                    split = user_input.split(' ')
                    if len(split)==0:
                        continue
                    cmd = split[0]
                    args = split[1:]

                    if cmd == '':
                        continue
                    elif cmd in func_dict:
                        out = func_dict[cmd](*args)
                    elif interpret_numbers(cmd) is not None:
                        if not any(x in args for x in ('-s', '-e', '-p')):
                            args = args + ['-e']
                        out = self.run(cmd, *args)
                    elif cmd == 'r':  # Refresh
                        continue
                    else:
                        response = raw_input('Unrecognised command: "{}".  Type "h" for help or Enter to continue. >'.format(cmd))
                        if response.lower()=='h':
                            self.help()
                        out = None
                    if out is self.QUIT or self.close_after:
                        break
                except Exception as name:
                    if self.catch_errors:
                        res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (name.__class__.__name__, name.message))
                        if res == 'e':
                            raise
                    else:
                        raise

    @staticmethod
    def get_experiment_list_str(exp_record_dict, just_last_record, view_mode='full', raise_display_errors=False):

        headers = {
            'full': ['E#', 'R#', 'Name', 'Last Run' if just_last_record else 'All Runs', 'Duration', 'Status', 'Valid', 'Result'],
            'results': ['E#', 'R#', 'Name', 'Result']
            }[view_mode]

        rows = []

        def get_field(header):
            try:
                return \
                    index if header=='#' else \
                    (str(i) if j==0 else '') if header == 'E#' else \
                    j if header == 'R#' else \
                    (name if j==0 else '') if header=='Name' else \
                    experiment_record.info.get_field_text(ExpInfoFields.TIMESTAMP) if header in ('Last Run', 'All Runs') else \
                    experiment_record.info.get_field_text(ExpInfoFields.RUNTIME) if header=='Duration' else \
                    experiment_record.info.get_field_text(ExpInfoFields.STATUS) if header=='Status' else \
                    experiment_record.get_invalid_arg_note() if header=='Valid' else \
                    experiment_record.get_one_liner() if header=='Result' else \
                    '???'
            except:
                if raise_display_errors:
                    raise
                return '<Display Error>'

        for i, (exp_id, record_ids) in enumerate(exp_record_dict.iteritems()):
            if len(record_ids)==0:
                if exp_id in exp_record_dict:
                    rows.append([str(i), '', exp_id, '<No Records>', '-', '-', '-', '-'])
            else:
                for j, record_id in enumerate(record_ids):
                    index, name = ['{}.{}'.format(i, j), exp_id] if j==0 else ['{}.{}'.format('`'*len(str(i)), j), exp_id]
                    try:
                        experiment_record = load_experiment_record(record_id)
                    except:
                        experiment_record = None
                    rows.append([get_field(h) for h in headers])
        assert all_equal([len(headers)]+[len(row) for row in rows]), 'Header length: {}, Row Lengths: \n  {}'.format(len(headers), '\n'.join([len(row) for row in rows]))
        table = tabulate(rows, headers=headers)
        return table

    def run(self, user_range, mode='-s'):
        assert mode in ('-s', '-p', '-e')
        ids = select_experiments(user_range, self.exp_record_dict)
        if len(ids)>1 and mode == '-p':
            import multiprocessing
            p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            p.map(partial(run_experiment_ignoring_errors, **self.run_args), ids)
        else:
            for experiment_identifier in ids:
                load_experiment(experiment_identifier).run(raise_exceptions=mode=='-e', display_results = False, **self.run_args )
        _warn_with_prompt('Finished running {} experiment{}.'.format(len(ids), '' if len(ids)==1 else 's'), use_prompt=not self.close_after)

    def test(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier).test()

    def help(self):
        _warn_with_prompt(self.HELP_TEXT, prompt = 'Press Enter to exit help.', use_prompt=not self.close_after)

    def show(self, user_range):

        records = [load_experiment_record(rid) for rid in select_experiment_records(user_range, self.exp_record_dict, flat=True)]
        print side_by_side([rec.get_full_info_string() for rec in records], max_linewidth=128)
        has_matplotlib_figures = any(loc.endswith('.pkl') for rec in records for loc in rec.get_figure_locs())
        if has_matplotlib_figures:
            from matplotlib import pyplot as plt
            for rec in records:
                rec.show_figures(hang=False)
            print '\n\n... Close all figures to return to experiment browser ...'
            plt.show()
        else:
            _warn_with_prompt(use_prompt=not self.close_after)

    def results(self, user_range = 'all'):
        record_ids = select_experiment_records(user_range, self.exp_record_dict)
        with IndentPrint("Results:", show_line=True, show_end=True):
            for erid in record_ids:
                record = load_experiment_record(erid)
                with IndentPrint(erid, show_line=True, show_end=True):
                    try:
                        result = record.get_result()
                        record.get_experiment().display_last(result, err_if_none=False)
                    except NoSavedResultError:
                        print '<No result was saved>'
                    except Exception as err:
                        print err

        _warn_with_prompt(use_prompt=not self.close_after)

    def errortrace(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint("Error Traces:", show_line=True, show_end=True):
            for erid in record_ids:
                with IndentPrint(erid, show_line=True):
                    record = load_experiment_record(erid)
                    print record.get_error_trace()
        _warn_with_prompt(use_prompt=not self.close_after)

    def delete(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print '{} out of {} Records will be deleted.'.format(len(record_ids), sum(len(recs) for recs in self.exp_record_dict.values()))
        with IndentPrint():
            print ExperimentRecordBrowser.get_record_table(record_ids, )
        response = raw_input('Type "yes" to continue. >')
        if response.lower() == 'yes':
            clear_experiment_records(record_ids)
            print 'Records deleted.'
        else:
            _warn_with_prompt('Records were not deleted.', use_prompt=not self.close_after)

    def call(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier)()

    def select(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint():
            print ExperimentRecordBrowser.get_record_table(record_ids)
        _warn_with_prompt('Selection "{}" includes {} out of {} records.'.format(user_range, len(record_ids), sum(len(recs) for recs in self.exp_record_dict.values())), use_prompt=not self.close_after)

    def side_by_side(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        records = [ExperimentRecord.from_identifier(rid) for rid in record_ids]
        print side_by_side([rec.get_full_info_string() for rec in records], max_linewidth=128)
        _warn_with_prompt(use_prompt=not self.close_after)

    def compare(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        compare_experiment_records(record_ids)
        _warn_with_prompt(use_prompt=not self.close_after)

    def records(self, ):
        browse_experiment_records(self.exp_record_dict.keys())

    def allruns(self, ):
        self.just_last_record = not self.just_last_record

    def filter(self, user_range):
        self._filter = user_range if user_range!='--clear' else None

    def view(self, mode):
        self.view_mode = mode

    def compare_results(self, user_range):
        experiment_ids = select_experiments(user_range, self.exp_record_dict)
        load_experiment(experiment_ids[0]).compare_results(experiment_ids)
        _warn_with_prompt(use_prompt=not self.close_after)

    def explist(self, surround = ""):
        print "\n".join([surround+k+surround for k in self.exp_record_dict.keys()])
        _warn_with_prompt(use_prompt=not self.close_after)

    def quit(self):
        return self.QUIT


def select_experiments(user_range, exp_record_dict):

    experiment_list = exp_record_dict.keys()

    number_range = interpret_numbers(user_range)
    if number_range is not None:
        return [experiment_list[i] for i in number_range]
    elif user_range == 'all':
        return experiment_list
    elif user_range.startswith('has:'):
        phrase = user_range[4:]
        return [exp_id for exp_id in experiment_list if phrase in exp_id]
    elif user_range.startswith('hasnot:'):
        phrase = user_range[len('hasnot:'):]
        return [exp_id for exp_id in experiment_list if phrase not in exp_id]
    elif user_range in ('unfinished', 'invalid'):  # Return all experiments where all records are unfinished/invalid
        record_filters = select_experiment_records(user_range, exp_record_dict)
        return [exp_id for exp_id in experiment_list if all(record_filters[exp_id])]
    else:
        raise Exception("Don't know how to use input '{}' to select experiments".format(user_range))


def select_experiment_records(user_range, exp_record_dict, flat=True):
    """
    :param user_range:
    :param exp_record_dict: An OrderedDict<experiment_name: list<experiment_record_name>>
    :param flat: Return a list of experiment records, instead of an OrderedDict
    :return: if not flat, an An OrderedDict<experiment_name: list<experiment_record_name>>
        otherwise a list<experiment_record_name>
    """
    filters = _filter_records(user_range, exp_record_dict)
    filtered_dict = OrderedDict((k, [v for v, f in izip_equal(exp_record_dict[k], filters[k]) if f]) for k in exp_record_dict.keys())
    if flat:
        return [record_id for records in filtered_dict.values() for record_id in records]
    else:
        return filtered_dict


def _filter_records(user_range, exp_record_dict):
    """
    :param user_range:
    :param exp_record_dict:
    :return:
    """

    def _bitwise(op, filter_set_1, filter_set_2):
        assert op in ('and', 'or')
        filter_set_3 = filter_set_1.copy()
        for k in filter_set_1.keys():
            filter_set_3[k] = [(a or b) if op=='or' else (a and b) for a, b in izip_equal(filter_set_1[k], filter_set_2[k])]
        return filter_set_3

    if '|' in user_range:
        return reduce(lambda a, b: _bitwise('or', a, b), [_filter_records(subrange, exp_record_dict) for subrange in user_range.split('|')])
    if '&' in user_range:
        return reduce(lambda a, b: _bitwise('and', a, b), [_filter_records(subrange, exp_record_dict) for subrange in user_range.split('&')])
    base = OrderedDict((k, [False]*len(v)) for k, v in exp_record_dict.iteritems())
    number_range = interpret_numbers(user_range)
    keys = exp_record_dict.keys()
    if number_range is not None:
        for i in number_range:
            base[keys[i]] = [True]*len(base[keys[i]])
    elif '.' in user_range:
        exp_rec_pairs = interpret_record_identifier(user_range)
        for exp_number, rec_number in exp_rec_pairs:
            base[keys[exp_number]][rec_number] = True
    elif user_range == 'old':
        for k, v in base.iteritems():
            base[k] = ([True]*(len(v)-1)+[False]) if len(v)>0 else []
    elif user_range == 'unfinished':
        for k, v in base.iteritems():
            base[k] = [load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) != ExpStatusOptions.FINISHED for rec_id in exp_record_dict[k]]
        # filtered_dict = OrderedDict((exp_id, [rec_id for rec_id in records if load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) != ExpStatusOptions.FINISHED]) for exp_id, records in exp_record_dict.iteritems())
    elif user_range == 'invalid':
        for k, v in base.iteritems():
            base[k] = [load_experiment_record(rec_id).is_valid() is False for rec_id in exp_record_dict[k]]
    elif user_range == 'all':
        for k, v in base.iteritems():
            base[k] = [True]*len(v)
    elif user_range == 'errors':
        for k, v in base.iteritems():
            base[k] = [load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS)==ExpStatusOptions.ERROR for rec_id in exp_record_dict[k]]
    else:
        raise Exception("Don't know how to interpret subset '{}'".format(user_range))
    return base


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


def compare_experiment_records(record_identifiers):

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


def browse_experiment_records(names = None, filter_text = None):
    """
    Browse through experiment records.

    :param names: Filter by names of experiments
    :param filter_text: Filter by regular expression
    :return:
    """

    experiment_record_browser = ExperimentRecordBrowser(experiment_names=names, filter_text=filter_text)
    experiment_record_browser.launch()


def select_experiment_records_from_list(user_range, experiment_records):
    return [rec_id for rec_id, f in izip_equal(experiment_records, _filter_experiment_record_list(user_range, experiment_records)) if f]


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
        for i, record_id in enumerate(experiment_record_ids):
            info = load_experiment_record(record_id).info
            if 'Module' in info:
                try:
                    import_module(info['Module'])
                    if not record_id_to_experiment_id(record_id) in GLOBAL_EXPERIMENT_LIBRARY:
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
    def get_record_table(record_ids = None, headers = ('#', 'Identifier', 'Start Time', 'Duration', 'Status', 'Valid', 'Notes', 'Result'), raise_display_errors = False):

        d = {
            '#': lambda: i,
            'Identifier': lambda: record_id,
            'Start Time': lambda: experiment_record.info.get_field_text(ExpInfoFields.TIMESTAMP, replacement_if_none='?'),
            'Duration': lambda: experiment_record.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
            'Status': lambda: experiment_record.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
            'Args': lambda: experiment_record.info.get_field_text(ExpInfoFields.ARGS, replacement_if_none='?'),
            'Valid': lambda: experiment_record.get_invalid_arg_note(),
            'Notes': lambda: experiment_record.info.get_field_text(ExpInfoFields.NOTES, replacement_if_none='?'),
            'Result': lambda: experiment_record.get_one_liner(),
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
        for i, record_id in enumerate(record_ids):
            experiment_record = load_experiment_record(record_id)
            rows.append(get_col_info(headers))
        assert all_equal([len(headers)]+[len(row) for row in rows]), 'Header length: {}, Row Lengths: \n {}'.format(len(headers), [len(row) for row in rows])
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

            print "=============== Experiment Records =================="
            # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
            print self.get_record_table(self.record_ids, raise_display_errors = self.raise_display_errors)
            print '-----------------------------------------------------'

            if self.experiment_names is not None or len(self.filters) != 0:
                print 'Not showing all experiments.  Type "showall" to see all experiments, or "viewfilters" to view current filters.'
            user_input = raw_input('Enter Command (show # to show and experiment, or h for help) >>')
            parts = shlex.split(user_input)
            if len(parts)==0:
                print "You need to specify an command.  Press h for help."
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
                res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
                if res == 'e':
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
        print self.get_record_table(self._select_records(user_range), headers=['Identifier', 'Args'])

    def rmfilters(self):
        self.filters = []
        self.record_ids = self.reload_ids()

    def viewfilters(self):
        _warn_with_prompt('Filtering for: \n  Names in {}\n  Expressions: {}'.format(self.experiment_names, self.filters))

    def compare(self, user_range):
        identifiers = self._select_records(user_range)
        compare_experiment_records(identifiers)
        _warn_with_prompt('')

    def show(self, user_range):
        for rid in self._select_records(user_range):
            load_experiment_record(rid).show()
        _warn_with_prompt('')

    def search(self, filter_text):
        print 'Found the following Records: '
        print self.get_record_table([rid for rid in self.record_ids if filter_text in rid])
        _warn_with_prompt()

    def delete(self, user_range):
        ids = self._select_records(user_range)
        print 'We will delete the following experiments:'
        print self.get_record_table(ids)
        conf = raw_input("Going to clear {} of {} experiment records shown above.  Enter 'yes' to confirm: ".format(len(ids), len(self.record_ids)))
        if conf=='yes':
            clear_experiment_records(ids=ids)
            assert not any(has_experiment_record(rid) for rid in ids), "Failed to delete them?"
        else:
            _warn_with_prompt("Did not delete experiments")
        self.record_ids = self.reload_ids()


if __name__ == '__main__':
    browse_experiment_records()
