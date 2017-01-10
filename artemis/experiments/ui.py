import shlex
from collections import OrderedDict
from importlib import import_module

from artemis.experiments.experiment_record import GLOBAL_EXPERIMENT_LIBRARY, experiment_id_to_latest_record_id, \
    show_experiment, get_all_record_ids, clear_experiment_records, get_experiment_record, filter_experiment_ids, \
    ExperimentRecord, get_latest_experiment_record, run_experiment_ignoring_errors, \
    experiment_id_to_record_ids, load_experiment_record, load_experiment, record_id_to_experiment_id, \
    is_experiment_loadable, record_id_to_timestamp
from artemis.general.display import IndentPrint, side_by_side
from artemis.general.hashing import compute_fixed_hash
from artemis.general.should_be_builtins import separate_common_items, bad_value, remove_duplicates, detect_duplicates, \
    izip_equal
from artemis.general.tables import build_table
from tabulate import tabulate


def _get_experiment_listing():
    experiment_listing = OrderedDict()
    for i, (name, exp) in enumerate(GLOBAL_EXPERIMENT_LIBRARY.iteritems()):
        experiment_listing['%s' % (i, )] = name
    return experiment_listing


def select_experiment():
    listing = _get_experiment_listing()
    print '\n'.join(['%s : %s' % (identifier, name) for identifier, name in listing.iteritems()])
    which_one = raw_input('Select Experiment >> ')
    if which_one.lstrip(' ').rstrip(' ') in listing:
        name = listing[which_one]
        return GLOBAL_EXPERIMENT_LIBRARY[name]
    else:
        raise Exception('No experiment with id: "%s"' % (which_one, ))


def _warn_with_prompt(message= None, prompt = 'Press Enter to continue'):
    if message is not None:
        print message
    raw_input('({}) >> '.format(prompt))


def find_experiment(*search_terms):
    """
    Find an experiment.  Invoke
    :param search_term: A term that will be used to search for an experiment.
    :return:
    """
    found_experiments = OrderedDict((name, ex) for name, ex in GLOBAL_EXPERIMENT_LIBRARY.iteritems() if all(term in name for term in search_terms))
    if len(found_experiments)==0:
        raise Exception("None of the {} experiments matched the search: '{}'".format(len(GLOBAL_EXPERIMENT_LIBRARY), search_terms))
    elif len(found_experiments)>1:
        raise Exception("More than one experiment matched the search '{}', you need to be more specific.  Found: {}".format(search_terms, found_experiments.keys()))
    else:
        return found_experiments.values()[0]


class _ExperimentInfo(object):
    """This object just helps with the display of experiments."""

    def __init__(self, name):
        self.name=name

    def get_experiment(self):
        return GLOBAL_EXPERIMENT_LIBRARY[self.name]

    def _get_arg_matching_record_note(self, record):
        info = record.get_info()
        last_run_args = dict(info['Args']) if 'Args' in info else '?'
        current_args = dict(self.get_experiment().get_args())
        if compute_fixed_hash(last_run_args)!=compute_fixed_hash(current_args):
            last_arg_str, this_arg_str = [['{}:{}'.format(k, v) for k, v in argdict.iteritems()] if isinstance(argdict, dict) else argdict for argdict in (last_run_args, current_args)]
            common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
            notes = "Warning: args have changed: {} -> {}".format(','.join(old_args), ','.join(new_args))
        else:
            notes = ""
        return notes

    @staticmethod
    def get_display_string(experiment_infos, just_last_record = True):
        headers = ['#', 'Name', 'Last Run' if just_last_record else 'All Runs', 'Duration', 'Status', 'Notes']
        rows = []
        for i, e in enumerate(experiment_infos):
            record_ids = [experiment_id_to_record_ids(e.name)[-1]] if just_last_record else experiment_id_to_record_ids(e.name)
            record_rows = [_ExperimentRecordInfo(erid).get_display_info(['Start Time', 'Duration', 'Status', 'Notes']) for erid in record_ids]
            if len(record_rows)==0:
                rows.append([i, e.name, '<No Records>', '-', '-', '-'])
            else:
                for j, recrow in enumerate(record_rows):
                    rows.append(([i, e.name] if j==0 else ['', '']) + recrow)
        return tabulate(rows, headers=headers)


class _ExperimentRecordInfo(object):

    def __init__(self, identifier):
        self.record_identifier = identifier
        self._info = None

    @classmethod
    def get_headers(cls):
        return ['Identifier', 'Run Time', 'Duration', 'Status']

    @property
    def info(self):
        if self._info is None:
            self._info = get_experiment_record(self.record_identifier).get_info()
        return self._info

    def get_display_info(self, fields):

        info_dict = {
            'Identifier': lambda: self.record_identifier,
            'Start Time': lambda: record_id_to_timestamp(self.record_identifier).replace('T', ' '),
            'Duration': lambda: self.info['Run Time'] if 'Run Time' in self.info else '?',
            'Status': lambda: (self.info['Status'][:self.info['Status'].index('\n')] if '\n' in self.info['Status'] else self.info['Status']) if 'Status' in self.info else '?',
            'Notes': self._get_valid_arg_note,
            'Args': lambda: ','.join('{}={}'.format(k, v) for k, v in self.info['Args']) if 'Args' in self.info else None
            }

        return [info_dict[field]() for field in fields]

    def _get_valid_arg_note(self):
        experiment_id = record_id_to_experiment_id(self.record_identifier)
        last_run_args = dict(self.info['Args']) if 'Args' in self.info else '?'
        if is_experiment_loadable(experiment_id):
            current_args = dict(load_experiment(record_id_to_experiment_id(self.record_identifier)).get_args())
            if not self.is_valid(last_run_args=last_run_args, current_args=current_args):
                last_arg_str, this_arg_str = [['{}:{}'.format(k, v) for k, v in argdict.iteritems()] if isinstance(argdict, dict) else argdict for argdict in (last_run_args, current_args)]
                common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
                notes = "Warning: args have changed: {} -> {}".format(','.join(old_args), ','.join(new_args))
            else:
                notes = ""
        else:
            notes = "<Experiment Not Currently Imported>"
        return notes

    def is_valid(self, last_run_args = None, current_args = None):
        """
        :return: True if the experiment arguments have changed, otherwise false.
        """
        if last_run_args is None:
            info = get_experiment_record(self.record_identifier).get_info()
            last_run_args = dict(info['Args']) if 'Args' in info else '?'
        if current_args is None:
            current_args = dict(load_experiment(record_id_to_experiment_id(self.record_identifier)).get_args())
        return compute_fixed_hash(last_run_args)==compute_fixed_hash(current_args)

    @staticmethod
    def get_display_string(experiment_records, fields = ('Identifier', 'Start Time', 'Duration', 'Status', 'Notes', ), number = True):
        if number:
            numbers = number if isinstance(number, (list, tuple)) else range(len(experiment_records)) if number is True else None
            assert len(numbers)==len(experiment_records)
            headers = ['#']+list(fields)
            rows = [[n]+rec.get_display_info(fields) for n, rec in zip(numbers, experiment_records)]
        else:
            headers = list(fields)
            rows = [rec.get_display_info(fields) for rec in experiment_records]
        return tabulate(rows, headers=headers)


def browse_experiments(catch_errors = False, close_after_run = False, just_last_record=False):
    """
    Browse Experiments

    :param catch_errors: True if you want to catch any errors here
    :param close_after_run: Close this menu after running an experiment
    :param just_last_record: Just show the last record for the experiment
    """
    help_text = """
        Enter '4', or 'run 4' to run experiment 4
              'run 4-6' to run experiment 4, 5, and 6 (in separate processes)
        Enter 'call 4' to call experiment 4 (like running, but doesn't save a record)
        Enter 'results' to view the results for all experiments or 'results 4' to just view results for experiment 4
        Enter 'show 4' to show the output from the last run of experiment 4 (if it has been run already).
        Enter 'records' to browse through all experiment records.
        Enter 'allruns' to toggle between showing all past runs of each experiment, and just the last one.
        Enter 'delete 4' to delete all records for experiment 4.
              'delete 4-6' to delete all records from experiments 4, 5, 6
              'delete old' to delete all but the most recent record for each experiment
              'delete unfinished' to delete all experiment records that have not run to completion
              'delete invalid' to delete records for which the experimental parameters have since changed
              (In all cases you will be asked to confirm the deletion.)
        Enter 'display 4' to replot the result of experiment 4 (if it has been run already, (only works if you've defined
            a display function for the experiment.)
        Enter 'q' to quit.
    """

    while True:
        experiment_infos = [_ExperimentInfo(name) for name in GLOBAL_EXPERIMENT_LIBRARY.keys()]
        print "==================== Experiments ===================="
        print _ExperimentInfo.get_display_string(experiment_infos, just_last_record=just_last_record)
        print '-----------------------------------------------------'
        user_input = raw_input('Enter command or experiment # to run (h for help) >> ').lstrip(' ').rstrip(' ')

        def get_experiment_name(_number):
            if isinstance(_number, basestring):
                _number = int(_number)
            assert _number < len(experiment_infos), 'No experiment with number "{}"'.format(_number, )
            return experiment_infos[_number].name

        def get_experiment(_number):
            return GLOBAL_EXPERIMENT_LIBRARY[get_experiment_name(_number)]

        def get_experiment_ids_for(user_range):
            which_ones = interpret_numbers(user_range)
            return [get_experiment_name(n) for n in which_ones]

        def get_record_ids_for(user_range, flat=False):
            record_ids = [experiment_id_to_record_ids(eid) for eid in get_experiment_ids_for(user_range)]
            if flat:
                return [rec_id for records in record_ids for rec_id in records]
            else:
                return record_ids

        try:
            split = user_input.split(' ')
            if len(split)==0:
                continue
            cmd = split[0]
            args = split[1:]
            if cmd == 'run':
                user_range, = args
                numbers = interpret_numbers(user_range) if user_range!='all' else range(len(experiment_infos))
                assert numbers is not None, "Could not interpret '{}' as a list of experiment numbers".format(user_range)
                if len(numbers)>1:
                    import multiprocessing
                    experiment_names = [experiment_infos[i].name for i in numbers]
                    p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    p.map(run_experiment_ignoring_errors, experiment_names)
                else:
                    number, = numbers
                    get_experiment(number).run()
                if close_after_run:
                    break
            elif cmd == 'show':
                number, = args
                name = get_experiment_name(number)
                last_identifier = experiment_id_to_latest_record_id(name)
                if last_identifier is None:
                    _warn_with_prompt("No record for experiment '%s' exists yet.  Run it to create one." % (experiment_infos, ))
                else:
                    show_experiment(experiment_id_to_latest_record_id(name))
                    _warn_with_prompt()
            elif cmd == 'call':
                number, = args
                get_experiment(number)()
                if close_after_run:
                    break
            elif cmd == 'compare':
                user_range, = args
                record_ids = get_record_ids_for(user_range, flat=True)
                records = [ExperimentRecord.from_identifier(rid) for rid in record_ids]
                texts = ['{title}\n{sep}\n{info}\n{sep}\n{output}\n{sep}'.format(title=rid, sep='='*len(rid), info=record.get_info_text(), output=record.get_log())
                         for rid, record in zip(record_ids, records)]
                print side_by_side(texts, max_linewidth=128)
                _warn_with_prompt()

            elif cmd == 'allruns':
                just_last_record = not just_last_record
            elif cmd == 'display':
                number, = args
                get_experiment(number).display_last()
            elif cmd == 'h':
                _warn_with_prompt(help_text, prompt = 'Press Enter to exit help.')
            elif cmd == 'results':  # Show all results
                if len(args) == 0:
                    experiment_names = [name.name for name in experiment_infos]
                else:
                    numbers_str, = args
                    numbers = interpret_numbers(numbers_str)
                    experiment_names = [get_experiment_name(n) for n in numbers]
                display_results(experiment_identifiers=experiment_names)
                _warn_with_prompt()
            elif cmd == 'delete':
                which_ones, = args
                number_range = interpret_numbers(which_ones)
                all_experiment_names = [e.name for e in experiment_infos]
                all_records = [experiment_id_to_record_ids(name) for name in all_experiment_names]
                if number_range is not None:
                    record_ids = [rec_id for i in number_range for rec_id in all_records[i]]
                elif which_ones == 'old':
                    record_ids = [rec_id for records_for_experiment in all_records for rec_id in records_for_experiment[:-1]]
                elif which_ones == 'unfinished':
                    record_ids = [rec_id for records in all_records for rec_id in records if load_experiment_record(rec_id).get_info('Status') != 'Ran Successfully']
                elif which_ones == 'invalid':
                    record_ids = [record_id for records in all_records for record_id in records if not _ExperimentRecordInfo(record_id).is_valid()]
                else:
                    raise Exception("Don't know how to interpret subset '{}'".format(which_ones))
                print '{} out of {} Records will be deleted.'.format(len(record_ids), sum(len(recs) for recs in all_records))
                with IndentPrint():
                    print _ExperimentRecordInfo.get_display_string([_ExperimentRecordInfo(rec_id) for rec_id in record_ids])
                response = raw_input('Type "yes" to continue. >')
                if response.lower() == 'yes':
                    clear_experiment_records(record_ids)
                    _warn_with_prompt('Records deleted.')
                else:
                    _warn_with_prompt('Records were not deleted.')
            elif cmd == 'q':
                break
            elif cmd == 'records':
                experiment_names = [name.name for name in experiment_infos]
                browse_experiment_records(experiment_names)
            elif cmd.isdigit():
                get_experiment(cmd).run()
            else:
                response = raw_input('Unrecognised command: "{}".  Type "h" for help or Enter to continue. >'.format(cmd))
                if response.lower()=='h':
                    _warn_with_prompt(help_text, prompt = 'Press Enter to exit help.')
        except Exception as name:
            if catch_errors:
                res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (name.__class__.__name__, name.message))
                if res == 'e':
                    raise
            else:
                raise


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


def display_results(experiment_identifiers = None):
    """
    :param experiment_identifiers: A list of experiment identifiers. (or none to take all imported experiments)
    :return:
    """
    if experiment_identifiers is None:
        experiment_identifiers = GLOBAL_EXPERIMENT_LIBRARY.keys()

    print "Results"

    with IndentPrint(show_line=True, show_end=True):
        for eid in experiment_identifiers:
            experiment = GLOBAL_EXPERIMENT_LIBRARY[eid]
            with IndentPrint(eid, show_line=True, show_end=True):
                records = experiment_id_to_record_ids(eid)
                if len(records)==0:
                    print 'No records for this experiment'
                else:
                    for erid in experiment_id_to_record_ids(eid):
                        with IndentPrint(record_id_to_timestamp(erid), show_line=True, show_end=True):
                            record = load_experiment_record(erid)
                            result = record.get_result()
                            experiment.display_last(result, err_if_none=False)


def compare_experiment_records(record_identifiers):

    records = [get_experiment_record(ident) for ident in record_identifiers]
    # info_results = OrderedDict([(identifier, record.get_info()) for identifier, record in zip(record_identifiers, records)]])

    funtion_names = [record.get_info()['Function'] for record in records]
    args = [record.get_info()['Args'] for record in records]
    results = [record.get_result() for record in records]

    common_args, different_args = separate_common_items(args)

    def lookup_fcn(identifier, column):
        index = record_identifiers.index(identifier)
        if column=='Function':
            return funtion_names[index]
        elif column=='Run Time':
            return records[index].get_info('Run Time')
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

    help = """
        q:                  Quit
        r:                  Refresh
        filter <text>       filter experiments
        viewfilters         View all filters on these results
        showall:            Show all experiments ever
        allnames:           Remove any name filters
        show <number>       Show experiment with number
        compare #1 #2 #3    Compare experiments by their numbers.
        clearall            Delete all experements from your computer
    """
    filters = []
    refresh = True
    while True:

        if refresh:
            ids = get_all_record_ids(experiment_ids= names, filters=filters)
            refresh=False

        record_infos = [_ExperimentRecordInfo(identifier) for identifier in ids]

        print "=============== Experiment Records =================="
        # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
        print _ExperimentRecordInfo.get_display_string(record_infos)
        print '-----------------------------------------------------'

        if names is not None or filter_text is not None:
            print 'Not showing all experiments.  Type "rmfilters" to clear filters, or "viewfilters" to view current filters.'
        user_input = raw_input('Enter Command (show # to show and experiment, or h for help) >>')
        parts = shlex.split(user_input)

        if len(parts)==0:
            _warn_with_prompt("You need to specify an experiment record number!")
            continue

        cmd = parts[0]
        args = parts[1:]

        def get_record_ids(user_range = None):
            if user_range is None:
                return [info.record_identifier for info in record_infos]
            else:
                numbers = get_record_numbers(user_range)
                ids = [record_infos[n].record_identifier for n in numbers]
                return ids

        def get_record_numbers(user_range):
            if user_range=='all':
                return range(len(record_infos))
            elif user_range=='new':
                old = detect_duplicates(get_record_ids(), key=record_id_to_experiment_id, keep_last=True)
                return [n for n, is_old in izip_equal(get_record_ids(), old) if not old]
            elif user_range=='old':
                old = detect_duplicates(get_record_ids(), key=record_id_to_experiment_id, keep_last=True)
                return [n for n, is_old in izip_equal(get_record_ids(), old) if old]
            elif user_range=='orphans':
                orphans = []
                for i, record_id in enumerate(get_record_ids()):
                    info = ExperimentRecord.from_identifier(record_id).get_info()
                    if 'Module' in info:
                        try:
                            import_module(info['Module'])
                            if not record_id_to_experiment_id(record_id) in GLOBAL_EXPERIMENT_LIBRARY:
                                orphans.append(i)
                        except ImportError:
                            orphans.append(i)
                    else:  # They must be old... lets kill them!
                        orphans.append(i)

                return orphans
            else:
                which_ones = interpret_numbers(user_range)
                if which_ones is None:
                    raise Exception('Could not interpret user range: "{}"'.format(user_range))
        try:
            if cmd == 'q':
                break
            elif cmd == 'h':
                _warn_with_prompt(help)
            elif cmd == 'filter':
                filter_text, = args
                filters.append(filter_text)
                refresh = True
            elif cmd == 'showall':
                names = None
                filters = []
                refresh = True
            elif cmd == 'args':
                which_ones = interpret_numbers(args[0]) if len(args)>0 else range(len(record_infos))
                print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in which_ones], fields = ['Identifier', 'Args'], number=which_ones)
                _warn_with_prompt()

            elif cmd == 'rmfilters':
                filters = []
                refresh = True
            elif cmd == 'r':
                refresh = True
            elif cmd == 'viewfilters':
                _warn_with_prompt('Filtering for: \n  Names in {}\n  Expressions: {}'.format(names, filters))
            elif cmd == 'compare':
                user_range, = args
                which_ones = interpret_numbers(user_range)
                identifiers = [ids[ix] for ix in which_ones]
                compare_experiment_records(identifiers)
                _warn_with_prompt('')
            elif cmd == 'show':
                index, = args
                exp_id = ids[int(index)]
                show_experiment(exp_id)
                _warn_with_prompt('')
            elif cmd == 'search':
                filter_text, = args
                which_ones = [i for i, eri in enumerate(record_infos) if filter_text in eri.record_identifier]
                print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in which_ones], fields = ['Identifier', 'Args'], number=which_ones)
                _warn_with_prompt()
            elif cmd == 'delete':
                user_range, = args
                numbers = get_record_numbers(user_range)
                print 'We will delete the following experiments:'
                with IndentPrint():
                    print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in numbers], number=numbers)
                conf = raw_input("Going to clear {} of {} experiment records shown above.  Enter 'yes' to confirm: ".format(len(numbers), len(record_infos)))
                if conf=='yes':
                    clear_experiment_records(ids=ids)
                    ids = get_all_record_ids(experiment_ids=names, filters=filters)
                    assert len(ids)==0, "Failed to delete them?"
                    _warn_with_prompt("Deleted {} of {} experiment records.".format(len(numbers), len(record_infos)))
                else:
                    _warn_with_prompt("Did not delete experiments")
            else:
                _warn_with_prompt('Bad Command: %s.' % cmd)
        except Exception as e:
            res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
            if res == 'e':
                raise


if __name__ == '__main__':
    browse_experiment_records()
