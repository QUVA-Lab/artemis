import shlex
from collections import OrderedDict
from functools import partial
from artemis.experiments.experiment_management import pull_experiments, select_experiments, select_experiment_records, \
    select_experiment_records_from_list, interpret_numbers, run_experiment_ignoring_errors, run_multiple_experiments
from artemis.experiments.experiment_record import get_all_record_ids, clear_experiment_records, \
    experiment_id_to_record_ids, load_experiment_record, ExpInfoFields, NoSavedResultError
from artemis.experiments.experiment_record_view import get_record_full_string, get_record_invalid_arg_string, \
    print_experiment_record_argtable, compare_experiment_results, show_experiment_records, get_oneline_result_string, \
    display_experiment_record
from artemis.experiments.experiments import GLOBAL_EXPERIMENT_LIBRARY, load_experiment, is_experiment_loadable
from artemis.general.display import IndentPrint, side_by_side
from artemis.general.should_be_builtins import all_equal
from tabulate import tabulate

try:
    import readline  # Makes raw_input behave like interactive shell.
    # http://stackoverflow.com/questions/15416054/command-line-in-python-with-history
except:
    pass  # readline not available


def _warn_with_prompt(message= None, prompt = 'Press Enter to continue', use_prompt=True):
    if message is not None:
        print message
    if use_prompt:
        return raw_input('({}) >> '.format(prompt))


def browse_experiments(command=None, **kwargs):
    """
    Browse Experiments

    :param root_experiment: Optionally, the root experiment to look at.
    :param catch_errors: True if you want to catch any errors here
    :param close_after: Close this menu after running an experiment
    :param just_last_record: Just show the last record for the experiment
    """
    browser = ExperimentBrowser(**kwargs)
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
> results 4-6         View the results experiments 4, 5, 6
> view results        View just the columns for experiment name and result
> view full           View all columns (the default view)
> show 4              Show the output from the last run of experiment 4 (if it has been run already).
> show 4-6            Show the output of experiments 4,5,6 together.
> records             Browse through all experiment records.
> allruns             Toggle between showing all past runs of each experiment, and just the last one.
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

Commands 'results', 'show', 'records', 'compare', 'sidebyside', 'selectrec', 'delete' allow you to specify a range of
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
    unfinished      Select all records that have not run to completion
    invalid         Select all records for which the arguments to their experiments have changed since they were run
    errors          Select all records that ended in error
    invalid|errors  Select all records that are invalid or ended in error (the '|' can be used to "or" any of the above)
    invalid&errors  Select all records that are invalid and ended in error (the '&' can be used to "and" any of the above)
"""

    def __init__(self, root_experiment = None, catch_errors = False, close_after = False, just_last_record = False,
            view_mode ='full', raise_display_errors=False, run_args=None, keep_record=True, truncate_result_to=100):

        if run_args is None:
            run_args = {}
        if 'keep_record' not in run_args:
            run_args['keep_record'] = keep_record
        self.root_experiment = root_experiment
        self.close_after = close_after
        self.just_last_record = just_last_record
        self.catch_errors = catch_errors
        self.exp_record_dict = self.reload_record_dict()
        self.raise_display_errors = raise_display_errors
        self.view_mode = view_mode
        self._filter = None
        self.run_args = {} if run_args is None else run_args
        self.truncate_result_to = truncate_result_to

    def reload_record_dict(self):

        names = GLOBAL_EXPERIMENT_LIBRARY.keys()

        if self.root_experiment is not None:
            # We could just go [ex.name for ex in self.root_experiment.get_all_variants(include_self=True)]
            # but we want to preserve the order in which experiments were created
            descendents_of_root = set(ex.name for ex in self.root_experiment.get_all_variants(include_self=True))
            names = [name for name in names if name in descendents_of_root]

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
            'selectexp': self.selectexp,
            'selectrec': self.selectrec,
            'allruns': self.allruns,
            'view': self.view,
            'h': self.help,
            'filter': self.filter,
            'explist': self.explist,
            'sidebyside': self.side_by_side,
            'argtable': self.argtable,
            'compare': self.compare,
            'display': self.display,
            'delete': self.delete,
            'errortrace': self.errortrace,
            'q': self.quit,
            'records': self.records,
            'pull': self.pull
            }

        while True:
            all_experiments = self.reload_record_dict()

            print "==================== Experiments ===================="
            self.exp_record_dict = all_experiments if self._filter is None else \
                OrderedDict((exp_name, all_experiments[exp_name]) for exp_name in select_experiments(self._filter, all_experiments))
            print self.get_experiment_list_str(self.exp_record_dict, just_last_record=self.just_last_record, view_mode=self.view_mode, raise_display_errors=self.raise_display_errors, truncate_result_to=self.truncate_result_to)
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
    def get_experiment_list_str(exp_record_dict, just_last_record, view_mode='full', raise_display_errors=False, truncate_result_to=100):

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
                    get_record_invalid_arg_string(experiment_record) if header=='Valid' else \
                    get_oneline_result_string(experiment_record, truncate_to=truncate_result_to) if header=='Result' else \
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
        assert all_equal([len(headers)] + [len(row) for row in rows]), 'Header length: {}, Row Lengths: \n  {}'.format(len(headers), '\n'.join([len(row) for row in rows]))
        table = tabulate(rows, headers=headers)
        return table

    def run(self, user_range, mode='-s', raise_exceptions = ''):
        assert mode in ('-s', '-e') or mode.startswith('-p')
        ids = select_experiments(user_range, self.exp_record_dict)
        run_multiple_experiments(
            experiments=[load_experiment(eid) for eid in ids],
            parallel=len(ids)>1 and mode.startswith('-p'),
            raise_exceptions = raise_exceptions=='-e',
            run_args=self.run_args
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

    def show(self, user_range, parallel_arg = None):
        """
        :param user_range:  A range specifying the record
        :param parallel_arg: -p to print logs side-by-side, and -s to print them in sequence.
        """
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        parallel_text = {'-p': True, '-s': False, None: None}[parallel_arg]
        has_matplotlib_figures=show_experiment_records(records, parallel_text = parallel_text, truncate_logs=None)
        if has_matplotlib_figures:
            from matplotlib import pyplot as plt
            print '\n\n... Close all figures to return to experiment browser ...'
            plt.ioff()
            plt.show()
        else:
            _warn_with_prompt(use_prompt=not self.close_after)

    def display(self, user_range = 'all'):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint("Results:", show_line=True, show_end=True):
            for record in records:
                with IndentPrint(record.get_id(), show_line=True, show_end=True):
                    display_experiment_record(record)

        _warn_with_prompt(use_prompt=not self.close_after)

    def errortrace(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint("Error Traces:", show_line=True, show_end=True):
            for record in records:
                with IndentPrint(record.get_id(), show_line=True):
                    print record.get_error_trace()
        _warn_with_prompt(use_prompt=not self.close_after)

    def delete(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print '{} out of {} Records will be deleted.'.format(len(records), sum(len(recs) for recs in self.exp_record_dict.values()))
        with IndentPrint():
            print ExperimentRecordBrowser.get_record_table(records, )
        response = raw_input('Type "yes" to continue. >')
        if response.lower() == 'yes':
            clear_experiment_records([record.get_id() for record in records])
            print 'Records deleted.'
        else:
            _warn_with_prompt('Records were not deleted.', use_prompt=not self.close_after)

    def call(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier)()

    def selectexp(self, user_range):
        exps_to_records = select_experiments(user_range, self.exp_record_dict, return_dict=True)
        with IndentPrint():
            print self.get_experiment_list_str(exps_to_records, just_last_record=self.just_last_record, view_mode=self.view_mode, raise_display_errors=self.raise_display_errors)
            # print ExperimentRecordBrowser.get_record_table(record_ids)
        _warn_with_prompt('Experiment Selection "{}" includes {} out of {} experiments.'.format(user_range, len(exps_to_records), len(self.exp_record_dict)), use_prompt=not self.close_after)

    def selectrec(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint():
            print ExperimentRecordBrowser.get_record_table(records)
        _warn_with_prompt('Record Selection "{}" includes {} out of {} records.'.format(user_range, len(records), sum(len(recs) for recs in self.exp_record_dict.values())), use_prompt=not self.close_after)

    def side_by_side(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print side_by_side([get_record_full_string(rec) for rec in records], max_linewidth=128)
        _warn_with_prompt(use_prompt=not self.close_after)

    def argtable(self, user_range):
        records = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        print_experiment_record_argtable(records)
        _warn_with_prompt(use_prompt=not self.close_after)

    def records(self, ):
        browse_experiment_records(self.exp_record_dict.keys())

    def pull(self, user_range, machine_name):
        from artemis.remote.remote_machines import get_remote_machine_info
        info = get_remote_machine_info(machine_name)
        exp_names = select_experiments(user_range, self.exp_record_dict)
        output = pull_experiments(user=info['username'], ip=info['ip'], experiment_names=exp_names, include_variants=False)
        print output

    def allruns(self, ):
        self.just_last_record = not self.just_last_record

    def filter(self, user_range):
        self._filter = user_range if user_range not in ('-', '--clear') else None

    def view(self, mode):
        self.view_mode = mode

    def compare(self, user_range):
        experiment_ids = select_experiments(user_range, self.exp_record_dict)
        experiments = [load_experiment(eid) for eid in experiment_ids]
        compare_experiment_results(experiments, error_if_no_result=False)
        _warn_with_prompt(use_prompt=not self.close_after)

    def explist(self, surround = ""):
        print "\n".join([surround+k+surround for k in self.exp_record_dict.keys()])
        _warn_with_prompt(use_prompt=not self.close_after)

    def quit(self):
        return self.QUIT


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
            'Valid': lambda: get_record_invalid_arg_string(experiment_record),
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

            print "=============== Experiment Records =================="
            # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
            print self.get_record_table([load_experiment_record(rid) for rid in self.record_ids], raise_display_errors = self.raise_display_errors)
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
        print_experiment_record_argtable(identifiers)
        _warn_with_prompt('')

    def show(self, user_range):
        record_ids = self._select_records(user_range)
        show_experiment_records([load_experiment_record(rid) for rid in record_ids])
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
        else:
            _warn_with_prompt("Did not delete experiments")
        self.record_ids = self.reload_ids()


if __name__ == '__main__':
    browse_experiment_records()
