import shlex
from collections import OrderedDict
from importlib import import_module


from artemis.experiments.experiment_record import GLOBAL_EXPERIMENT_LIBRARY, experiment_id_to_latest_record_id, \
    show_experiment, get_all_record_ids, clear_experiment_records, filter_experiment_ids, \
    ExperimentRecord, get_latest_experiment_record, run_experiment_ignoring_errors, \
    experiment_id_to_record_ids, load_experiment_record, load_experiment, record_id_to_experiment_id, \
    is_experiment_loadable, record_id_to_timestamp, ExpInfoFields, ExpStatusOptions, has_experiment_record
from artemis.general.display import IndentPrint, side_by_side
from artemis.general.hashing import compute_fixed_hash
from artemis.general.should_be_builtins import separate_common_items, bad_value, remove_duplicates, detect_duplicates, \
    izip_equal, all_equal
from artemis.general.tables import build_table
from tabulate import tabulate


import readline

def _setup_input_memory():
    try:
        import readline  # Makes raw_input behave like interactive shell.
        # http://stackoverflow.com/questions/15416054/command-line-in-python-with-history
    except:
        pass  # readline not available


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


# class _ExperimentInfo(object):
#     """This object just helps with the display of experiments."""
#
#     def __init__(self, name):
#         self.name=name
#
#     def get_experiment(self):
#         return GLOBAL_EXPERIMENT_LIBRARY[self.name]
#
#     def _get_arg_matching_record_note(self, record):
#         info = record.get_info()
#         last_run_args = dict(info['Args']) if 'Args' in info else '?'
#         current_args = dict(self.get_experiment().get_args())
#         if compute_fixed_hash(last_run_args)!=compute_fixed_hash(current_args):
#             last_arg_str, this_arg_str = [['{}:{}'.format(k, v) for k, v in argdict.iteritems()] if isinstance(argdict, dict) else argdict for argdict in (last_run_args, current_args)]
#             common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
#             notes = "Warning: args have changed: {} -> {}".format(','.join(old_args), ','.join(new_args))
#         else:
#             notes = ""
#         return notes
#
#     @staticmethod
#     def get_display_string(experiment_infos, just_last_record = True):
#         headers = ['#', 'Name', 'Last Run' if just_last_record else 'All Runs', 'Duration', 'Status', 'Notes', 'Result']
#         rows = []
#         for i, (exp_id, record_ids) in enumerate(experiment_infos.iteritems()):
#             # record_ids = [experiment_id_to_record_ids(e.name)[-1]] if just_last_record else experiment_id_to_record_ids(e.name)
#             record_rows = [_ExperimentRecordInfo(erid).get_display_info(['Start Time', 'Duration', 'Status', 'Notes', 'Result']) for erid in record_ids]
#
#             if len(record_rows)==0:
#                 rows.append([str(i), exp_id, '<No Records>', '-', '-', '-', '-'])
#             else:
#                 for j, recrow in enumerate(record_rows):
#                     rows.append((['{}.{}'.format(i, j), exp_id] if j==0 else ['{}.{}'.format('`'*len(str(i)), j), exp_id]) + recrow)
#
#         assert all_equal([len(headers)]+[len(row) for row in record_rows]), 'Header length: {}, Row Lengths: \n  {}'.format(len(headers), '\n'.join([len(row) for row in record_rows]))
#         rows.append([' ']*len(rows[-1]))
#         return tabulate(rows, headers=headers, floatfmt=None)


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
            self._info = load_experiment_record(self.record_identifier).info
        return self._info

    def get_display_info(self, fields):

        # try:
        info_dict = {
            'Identifier': lambda: self.info.get_field_text(ExpInfoFields.ID, replacement_if_none='?'),
            'Start Time': lambda: self.info.get_field_text(ExpInfoFields.TIMESTAMP, replacement_if_none='?'),
            'Duration': lambda: self.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
            'Status': lambda: self.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
            # 'Valid': self.,
            'Args': lambda: self.info.get_field_text(ExpInfoFields.ARGS, replacement_if_none='?'),
            'Result': lambda: load_experiment_record(self.record_identifier).get_one_liner()
            }

        return [info_dict[field]() for field in fields]
        # except:
        #     return ['<Could not Read>']*len(fields)


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

    browser = ExperimentBrowser(catch_errors=catch_errors, close_after_run=close_after_run, just_last_record=just_last_record)
    browser.launch()


    # help_text = """
    #     Enter '4', or 'run 4' to run experiment 4
    #           'run 4-6' to run experiment 4, 5, and 6 (in separate processes)
    #     Enter 'call 4' to call experiment 4 (like running, but doesn't save a record)
    #     Enter 'results' to view the results for all experiments or 'results 4' to just view results for experiment 4
    #     Enter 'show 4' to show the output from the last run of experiment 4 (if it has been run already).
    #     Enter 'records' to browse through all experiment records.
    #     Enter 'allruns' to toggle between showing all past runs of each experiment, and just the last one.
    #     Enter 'delete 4' to delete all records for experiment 4.
    #           'delete 4-6' to delete all records from experiments 4, 5, 6
    #           'delete old' to delete all but the most recent record for each experiment
    #           'delete unfinished' to delete all experiment records that have not run to completion
    #           'delete invalid' to delete records for which the experimental parameters have since changed
    #           (In all cases you will be asked to confirm the deletion.)
    #     Enter 'q' to quit.
    # """
    #
    # _setup_input_memory()
    #
    # # def get_experiment_name(_number):
    # #     if isinstance(_number, basestring):
    # #         _number = int(_number)
    # #     assert _number < len(experiment_infos), 'No experiment with number "{}"'.format(_number, )
    # #     return experiment_infos[_number].name
    #
    # # def get_experiment_ids_for(user_range):
    # #     which_ones = interpret_numbers(user_range)
    # #     return [get_experiment_name(n) for n in which_ones]
    # #
    # # def get_record_ids_for(user_range, flat=False):
    # #     record_ids = [experiment_id_to_record_ids(eid) for eid in get_experiment_ids_for(user_range)]
    # #     if flat:
    # #         return [rec_id for records in record_ids for rec_id in records]
    # #     else:
    # #         return record_ids
    #
    # while True:
    #     experiment_infos = [_ExperimentInfo(name) for name in GLOBAL_EXPERIMENT_LIBRARY.keys()]
    #     print "==================== Experiments ===================="
    #     print _ExperimentInfo.get_display_string(experiment_infos, just_last_record=just_last_record)
    #     print '-----------------------------------------------------'
    #     user_input = raw_input('Enter command or experiment # to run (h for help) >> ').lstrip(' ').rstrip(' ')
    #     exp_record_dict = OrderedDict((e.name, experiment_id_to_record_ids(e.name)) for e in experiment_infos)
    #
    #     with IndentPrint():
    #         try:
    #             split = user_input.split(' ')
    #             if len(split)==0:
    #                 continue
    #             cmd = split[0]
    #             args = split[1:]
    #             if cmd == 'run':
    #                 if len(args)==1:
    #                     user_range, = args
    #                     mode = '-p'
    #                 else:
    #                     user_range, mode = args
    #                 ids = select_experiments(user_range, exp_record_dict)
    #                 if len(ids)>1 and mode == '-p':
    #                     import multiprocessing
    #                     # experiment_names = [experiment_infos[i].name for i in numbers]
    #                     p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #                     p.map(run_experiment_ignoring_errors, ids)
    #                 elif mode == '-s':
    #                     for experiment_identifier in ids:
    #                         load_experiment(experiment_identifier).run()
    #                 if close_after_run:
    #                     break
    #             elif cmd == 'test':
    #                 user_range, = args
    #                 ids = select_experiments(user_range, exp_record_dict)
    #                 for experiment_identifier in ids:
    #                     load_experiment(experiment_identifier).test()
    #             elif cmd == 'show':
    #                 number, = args
    #                 record_ids = interpret_record_identifier(number)
    #                 for rid in record_ids:
    #                     show_experiment(rid)
    #                     _warn_with_prompt()
    #             elif cmd == 'call':
    #                 user_range, = args
    #                 ids = select_experiments(user_range, exp_record_dict)
    #                 for experiment_identifier in ids:
    #                     load_experiment(experiment_identifier)()
    #                 if close_after_run:
    #                     break
    #             elif cmd == 'select':
    #                 user_range, = args
    #                 record_ids = select_experiment_records(user_range, exp_record_dict, flat=True)
    #                 with IndentPrint():
    #                     print _ExperimentRecordInfo.get_display_string([_ExperimentRecordInfo(rec_id) for rec_id in record_ids])
    #                 _warn_with_prompt('Selection "{}" includes {} out of {} records.'.format(user_range, len(record_ids), sum(len(recs) for recs in exp_record_dict.values())))
    #             elif cmd == 'compare':
    #                 user_range, = args
    #                 record_ids = select_experiment_records(user_range, exp_record_dict, flat=True)
    #                 records = [ExperimentRecord.from_identifier(rid) for rid in record_ids]
    #                 texts = ['{title}\n{sep}\n{info}\n{sep}\n{output}\n{sep}'.format(title=rid, sep='='*len(rid), info=record.info.get_text(), output=record.get_log())
    #                          for rid, record in zip(record_ids, records)]
    #                 print side_by_side(texts, max_linewidth=128)
    #                 _warn_with_prompt()
    #             elif cmd == 'allruns':
    #                 just_last_record = not just_last_record
    #             # elif cmd == 'display':
    #             #     user_range, = args
    #             #     record_ids = select_experiment_records(user_range)
    #             #     for rid in record_ids:
    #             #         load_experiment_record(rid).display()
    #             elif cmd == 'h':
    #                 _warn_with_prompt(help_text, prompt = 'Press Enter to exit help.')
    #             elif cmd == 'results':  # Show all results
    #                 if len(args) == 0:
    #                     exp_ids = exp_record_dict.keys()
    #                 else:
    #                     numbers_str, = args
    #                     exp_ids = select_experiment_records(numbers_str)
    #                 display_results(experiment_identifiers=exp_ids)
    #                 _warn_with_prompt()
    #             elif cmd == 'delete':
    #                 user_range, = args
    #                 record_ids = select_experiment_records(user_range, exp_record_dict, flat=True)
    #                 print '{} out of {} Records will be deleted.'.format(len(record_ids), sum(len(recs) for recs in exp_record_dict.values()))
    #                 with IndentPrint():
    #                     print _ExperimentRecordInfo.get_display_string([_ExperimentRecordInfo(rec_id) for rec_id in record_ids])
    #                 response = raw_input('Type "yes" to continue. >')
    #                 if response.lower() == 'yes':
    #                     clear_experiment_records(record_ids)
    #                     print 'Records deleted.'
    #                 else:
    #                     _warn_with_prompt('Records were not deleted.')
    #             elif cmd == 'q':
    #                 break
    #             elif cmd == 'records':
    #                 experiment_names = [name.name for name in experiment_infos]
    #                 browse_experiment_records(experiment_names)
    #             elif cmd.isdigit():
    #                 user_range, = args
    #                 exp_ids = select_experiments(user_range, exp_record_dict)
    #                 for eid in exp_ids:
    #                     load_experiment(eid).run()
    #             else:
    #                 response = raw_input('Unrecognised command: "{}".  Type "h" for help or Enter to continue. >'.format(cmd))
    #                 if response.lower()=='h':
    #                     _warn_with_prompt(help_text, prompt = 'Press Enter to exit help.')
    #         except Exception as name:
    #             if catch_errors:
    #                 res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (name.__class__.__name__, name.message))
    #                 if res == 'e':
    #                     raise
    #             else:
    #                 raise


class ExperimentBrowser(object):

    QUIT = 'Quit'
    HELP_TEXT = """
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
Enter 'q' to quit.
"""

    def __init__(self, catch_errors = False, close_after_run = True, just_last_record = False):

        self.close_after_run = close_after_run
        self.just_last_record = just_last_record
        self.catch_errors = catch_errors
        self.exp_record_dict = self.reload_record_dict()

    def reload_record_dict(self):
        d= OrderedDict((name, experiment_id_to_record_ids(name)) for name in GLOBAL_EXPERIMENT_LIBRARY.keys())
        if self.just_last_record:
            for k in d.keys():
                d[k] = [d[k][-1]]
        return d

    def launch(self):

        func_dict = {
            'run': self.run,
            'test': self.test,
            'show': self.show,
            'call': self.call,
            'select': self.select,
            'allruns': self.allruns,
            'h': self.help,
            'results': self.results,
            'delete': self.delete,
            'q': self.quit,
            'records': self.records
            }

        while True:
            self.exp_record_dict = self.reload_record_dict()

            print "==================== Experiments ===================="
            print self.get_experiment_list_str(self.exp_record_dict, just_last_record=self.just_last_record)
            print '-----------------------------------------------------'
            user_input = raw_input('Enter command or experiment # to run (h for help) >> ').lstrip(' ').rstrip(' ')
            with IndentPrint():
                try:
                    split = user_input.split(' ')
                    if len(split)==0:
                        continue
                    cmd = split[0]
                    args = split[1:]

                    if cmd in func_dict:
                        out = func_dict[cmd](*args)
                    elif interpret_numbers(cmd) is not None:
                        out = self.run(cmd, *args)
                    else:
                        response = raw_input('Unrecognised command: "{}".  Type "h" for help or Enter to continue. >'.format(cmd))
                        if response.lower()=='h':
                            self.help()
                        out = None
                    if out is self.QUIT:
                        break
                except Exception as name:
                    if self.catch_errors:
                        res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (name.__class__.__name__, name.message))
                        if res == 'e':
                            raise
                    else:
                        raise

    @staticmethod
    def get_experiment_list_str(exp_record_dict, just_last_record):
        headers = ['#', 'Name', 'Last Run' if just_last_record else 'All Runs', 'Duration', 'Status', 'Valid', 'Result']
        rows = []
        for i, (exp_id, record_ids) in enumerate(exp_record_dict.iteritems()):
            if len(record_ids)==0:
                rows.append([str(i), exp_id, '<No Records>', '-', '-', '-', '-'])
            else:
                for j, record_id in enumerate(record_ids):
                    index, name = ['{}.{}'.format(i, j), exp_id] if j==0 else ['{}.{}'.format('`'*len(str(i)), j), exp_id]
                    experiment_record = load_experiment_record(record_id)
                    rows.append([
                        index,
                        name,
                        experiment_record.info.get_field_text(ExpInfoFields.ID, replacement_if_none='?'),
                        experiment_record.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
                        experiment_record.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
                        experiment_record.get_invalid_arg_note(),
                        experiment_record.get_one_liner()
                        ])
                        # info_dict = {
                        #     'Identifier': lambda: self.info.get_field_text(ExpInfoFields.ID, replacement_if_none='?'),
                        #     'Start Time': lambda: self.info.get_field_text(ExpInfoFields.TIMESTAMP, replacement_if_none='?'),
                        #     'Duration': lambda: self.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
                        #     'Status': lambda: self.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
                        #     'Valid': self._get_valid_arg_note,
                        #     'Args': lambda: self.info.get_field_text(ExpInfoFields.ARGS, replacement_if_none='?'),
                        #     'Result': lambda: load_experiment_record(self.record_identifier).get_one_liner()
                        #     }

            # record_rows = [_ExperimentRecordInfo(erid).get_display_info(['Start Time', 'Duration', 'Status', 'Notes', 'Result']) for erid in record_ids]





            # if len(record_rows)==0:
            #     rows.append([str(i), exp_id, '<No Records>', '-', '-', '-', '-'])
            # else:
            #     for j, recrow in enumerate(record_rows):
            #         rows.append((['{}.{}'.format(i, j), exp_id] if j==0 else ['{}.{}'.format('`'*len(str(i)), j), exp_id]) + recrow)
        assert all_equal([len(headers)]+[len(row) for row in rows]), 'Header length: {}, Row Lengths: \n  {}'.format(len(headers), '\n'.join([len(row) for row in record_rows]))
        rows.append([' ']*len(rows[-1]))

        return tabulate(rows, headers=headers, floatfmt=None)

    def run(self, user_range, mode='-p'):
        ids = select_experiments(user_range, self.exp_record_dict)
        if len(ids)>1 and mode == '-p':
            import multiprocessing
            p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            p.map(run_experiment_ignoring_errors, ids)
        else:
            for experiment_identifier in ids:
                load_experiment(experiment_identifier).run()
        if self.close_after_run:
            return self.QUIT

    def test(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier).test()

    def help(self):
        _warn_with_prompt(self.HELP_TEXT, prompt = 'Press Enter to exit help.')

    def show(self, user_range):
        record_ids = interpret_record_identifier(user_range)
        for rid in record_ids:
            load_experiment_record(rid).show()
            _warn_with_prompt()

    def results(self, user_range = None):
        if user_range is None:
            user_range = self.exp_record_dict.keys()
        exp_ids = select_experiment_records(user_range)
        display_results(experiment_identifiers=exp_ids)
        _warn_with_prompt()

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
            _warn_with_prompt('Records were not deleted.')

    def call(self, user_range):
        ids = select_experiments(user_range, self.exp_record_dict)
        for experiment_identifier in ids:
            load_experiment(experiment_identifier)()
        if self.close_after_run:
            return self.QUIT

    def select(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        with IndentPrint():
            print _ExperimentRecordInfo.get_display_string([_ExperimentRecordInfo(rec_id) for rec_id in record_ids])
        _warn_with_prompt('Selection "{}" includes {} out of {} records.'.format(user_range, len(record_ids), sum(len(recs) for recs in self.exp_record_dict.values())))

    def compare(self, user_range):
        record_ids = select_experiment_records(user_range, self.exp_record_dict, flat=True)
        records = [ExperimentRecord.from_identifier(rid) for rid in record_ids]
        texts = ['{title}\n{sep}\n{info}\n{sep}\n{output}\n{sep}'.format(title=rid, sep='='*len(rid), info=record.info.get_text(), output=record.get_log())
                 for rid, record in zip(record_ids, records)]
        print side_by_side(texts, max_linewidth=128)
        _warn_with_prompt()

    def records(self, ):
        browse_experiment_records(self.exp_record_dict.keys())

    def allruns(self, ):
        self.just_last_record = not self.just_last_record

    def quit(self):
        return self.QUIT


def select_experiments(user_range, exp_record_dict):

    experiment_list = exp_record_dict.keys()

    number_range = interpret_numbers(user_range)
    if number_range is not None:
        return [experiment_list[i] for i in number_range]
    elif user_range == 'all':
        return experiment_list
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
    # number_range = interpret_numbers(user_range)
    # keys = exp_record_dict.keys()
    # if number_range is not None:
    #     filtered_dict = OrderedDict((keys[i], exp_record_dict[keys[i]]) for i in number_range)
    # elif '.' in user_range:
    #     exp_rec_pairs = interpret_record_identifier(user_range)
    #     filtered_dict = OrderedDict([(keys[exp_number], [exp_record_dict[keys[exp_number]][i] for i in record_numbers])])
    # elif user_range == 'old':
    #     filtered_dict = OrderedDict((exp_id, records[:-1]) for exp_id, records in exp_record_dict.iteritems())
    # elif user_range == 'unfinished':
    #     filtered_dict = OrderedDict((exp_id, [rec_id for rec_id in records if load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) != ExpStatusOptions.FINISHED]) for exp_id, records in exp_record_dict.iteritems())
    # elif user_range == 'invalid':
    #     filtered_dict = OrderedDict((exp_id, [rec_id for rec_id in records if not _ExperimentRecordInfo(rec_id).is_valid()]) for exp_id, records in exp_record_dict.iteritems())
    # elif user_range == 'all':
    #     filtered_dict = exp_record_dict
    # else:
    #     raise Exception("Don't know how to interpret subset '{}'".format(user_range))

    filters = _filter_records(user_range, exp_record_dict)
    filtered_dict = OrderedDict((k, [v for v, f in izip_equal(exp_record_dict[k], filters[k]) if f]) for k in exp_record_dict.keys())
    if flat:
        return [record_id for records in filtered_dict.values() for record_id in records]
    else:
        return filtered_dict


def _filter_records(user_range, exp_record_dict):

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
            base[k] = [True]*(len(v)-1)+[False]
    elif user_range == 'unfinished':
        for k, v in base.iteritems():
            base[k] = [load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) != ExpStatusOptions.FINISHED for rec_id in exp_record_dict[k]]
        # filtered_dict = OrderedDict((exp_id, [rec_id for rec_id in records if load_experiment_record(rec_id).info.get_field(ExpInfoFields.STATUS) != ExpStatusOptions.FINISHED]) for exp_id, records in exp_record_dict.iteritems())
    elif user_range == 'invalid':
        for k, v in base.iteritems():
            base[k] = [_ExperimentRecordInfo(rec_id).is_valid() for rec_id in exp_record_dict[k]]
    elif user_range == 'all':
        for k, v in base.iteritems():
            base[k] = [True]*len(v)
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

    experiment_record_browser = ExperimentRecordBrowser(experiment_names=names, filter_text=filter_text)
    experiment_record_browser.launch()

    # help = """
    #     q:                  Quit
    #     r:                  Refresh
    #     filter <text>       filter experiments
    #     viewfilters         View all filters on these results
    #     showall:            Show all experiments ever
    #     allnames:           Remove any name filters
    #     show <number>       Show experiment with number
    #     compare #1 #2 #3    Compare experiments by their numbers.
    #     clearall            Delete all experements from your computer
    # """
    # filters = []
    # refresh = True
    #
    # _setup_input_memory()
    #
    # while True:
    #
    #     if refresh:
    #         ids = get_all_record_ids(experiment_ids= names, filters=filters)
    #         refresh=False
    #
    #     record_infos = [_ExperimentRecordInfo(identifier) for identifier in ids]
    #
    #     print "=============== Experiment Records =================="
    #     # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
    #     print _ExperimentRecordInfo.get_display_string(record_infos)
    #     print self.
    #     print '-----------------------------------------------------'
    #
    #     if names is not None or filter_text is not None:
    #         print 'Not showing all experiments.  Type "rmfilters" to clear filters, or "viewfilters" to view current filters.'
    #     user_input = raw_input('Enter Command (show # to show and experiment, or h for help) >>')
    #     parts = shlex.split(user_input)
    #
    #     if len(parts)==0:
    #         _warn_with_prompt("You need to specify an experiment record number!")
    #         continue
    #
    #     cmd = parts[0]
    #     args = parts[1:]
    #
    #     def get_record_ids(user_range = None):
    #         if user_range is None:
    #             return [info.record_identifier for info in record_infos]
    #         else:
    #             numbers = get_record_numbers(user_range)
    #             ids = [record_infos[n].record_identifier for n in numbers]
    #             return ids
    #
    #     def get_record_numbers(user_range):
    #         if user_range=='all':
    #             return range(len(record_infos))
    #         elif user_range=='new':
    #             old = detect_duplicates(get_record_ids(), key=record_id_to_experiment_id, keep_last=True)
    #             return [n for n, is_old in izip_equal(get_record_ids(), old) if not old]
    #         elif user_range=='old':
    #             old = detect_duplicates(get_record_ids(), key=record_id_to_experiment_id, keep_last=True)
    #             return [n for n, is_old in izip_equal(get_record_ids(), old) if old]
    #         elif user_range=='orphans':
    #             orphans = []
    #             for i, record_id in enumerate(get_record_ids()):
    #                 info = ExperimentRecord.from_identifier(record_id).get_info()
    #                 if 'Module' in info:
    #                     try:
    #                         import_module(info['Module'])
    #                         if not record_id_to_experiment_id(record_id) in GLOBAL_EXPERIMENT_LIBRARY:
    #                             orphans.append(i)
    #                     except ImportError:
    #                         orphans.append(i)
    #                 else:  # They must be old... lets kill them!
    #                     orphans.append(i)
    #
    #             return orphans
    #         else:
    #             which_ones = interpret_numbers(user_range)
    #             if which_ones is None:
    #                 raise Exception('Could not interpret user range: "{}"'.format(user_range))
    #     try:
    #         if cmd == 'q':
    #             break
    #         elif cmd == 'h':
    #             _warn_with_prompt(help)
    #         elif cmd == 'filter':
    #             filter_text, = args
    #             filters.append(filter_text)
    #             refresh = True
    #         elif cmd == 'showall':
    #             names = None
    #             filters = []
    #             refresh = True
    #         elif cmd == 'args':
    #             which_ones = interpret_numbers(args[0]) if len(args)>0 else range(len(record_infos))
    #             print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in which_ones], fields = ['Identifier', 'Args'], number=which_ones)
    #             _warn_with_prompt()
    #
    #         elif cmd == 'rmfilters':
    #             filters = []
    #             refresh = True
    #         elif cmd == 'r':
    #             refresh = True
    #         elif cmd == 'viewfilters':
    #             _warn_with_prompt('Filtering for: \n  Names in {}\n  Expressions: {}'.format(names, filters))
    #         elif cmd == 'compare':
    #             user_range, = args
    #             which_ones = interpret_numbers(user_range)
    #             identifiers = [ids[ix] for ix in which_ones]
    #             compare_experiment_records(identifiers)
    #             _warn_with_prompt('')
    #         elif cmd == 'show':
    #             index, = args
    #             exp_id = ids[int(index)]
    #             show_experiment(exp_id)
    #             _warn_with_prompt('')
    #         elif cmd == 'search':
    #             filter_text, = args
    #             which_ones = [i for i, eri in enumerate(record_infos) if filter_text in eri.record_identifier]
    #             print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in which_ones], fields = ['Identifier', 'Args'], number=which_ones)
    #             _warn_with_prompt()
    #         elif cmd == 'delete':
    #             user_range, = args
    #             numbers = get_record_numbers(user_range)
    #             print 'We will delete the following experiments:'
    #             with IndentPrint():
    #                 print _ExperimentRecordInfo.get_display_string([record_infos[n] for n in numbers], number=numbers)
    #             conf = raw_input("Going to clear {} of {} experiment records shown above.  Enter 'yes' to confirm: ".format(len(numbers), len(record_infos)))
    #             if conf=='yes':
    #                 clear_experiment_records(ids=ids)
    #                 ids = get_all_record_ids(experiment_ids=names, filters=filters)
    #                 assert len(ids)==0, "Failed to delete them?"
    #                 _warn_with_prompt("Deleted {} of {} experiment records.".format(len(numbers), len(record_infos)))
    #             else:
    #                 _warn_with_prompt("Did not delete experiments")
    #         else:
    #             _warn_with_prompt('Bad Command: %s.' % cmd)
    #     except Exception as e:
    #         res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
    #         if res == 'e':
    #             raise


def select_experiment_records_from_list(user_range, experiment_records):
    return [rec_id for rec_id, f in izip_equal(experiment_records, _filter_records(user_range, experiment_records)) if f]


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
    compare #1 #2 #3    Compare experiments by their numbers.
    clearall            Delete all experements from your computer
"""

    def __init__(self, experiment_names = None, filter_text = None):
        """
        Browse through experiment records.

        :param names: Filter by names of experiments
        :param filter_text: Filter by regular expression
        :return:
        """
        self.experiment_names = experiment_names
        self.filters = [filter_text]
        self.record_ids = self.reload_ids()

    def reload_ids(self):
        return get_all_record_ids(experiment_ids= self.experiment_names, filters=self.filters)

    @staticmethod
    def get_record_table(record_ids = None, headers = ('#', 'Identifier', 'Start Time', 'Duration', 'Status', 'Notes', 'Result')):

        def get_col_info(headers):
            d = {
                '#': i,
                'Identifier': record_id,
                'Start Time': experiment_record.info.get_field_text(ExpInfoFields.TIMESTAMP, replacement_if_none='?'),
                'Duration': experiment_record.info.get_field_text(ExpInfoFields.RUNTIME, replacement_if_none='?'),
                'Status': experiment_record.info.get_field_text(ExpInfoFields.STATUS, replacement_if_none='?'),
                'Args': experiment_record.info.get_field_text(ExpInfoFields.ARGS, replacement_if_none='?'),
                'Notes': experiment_record.info.get_field_text(ExpInfoFields.NOTES, replacement_if_none='?'),
                'Result': experiment_record.get_one_liner(),
                '<Error>': '<Error>'
                }
            return [d[h] for h in headers]
        rows = []
        for i, record_id in enumerate(record_ids):
            try:
                experiment_record = load_experiment_record(record_id)
                rows.append(get_col_info(headers))
            except:
                rows.append(get_col_info([h if h in ('#', 'Identifier') else '<Error>' for h in headers]))
        assert all_equal([len(headers)]+[len(row) for row in rows]), 'Header length: {}, Row Lengths: \n  {}'.format(len(headers), '\n'.join([len(row) for row in record_rows]))
        return tabulate(rows, headers=headers, floatfmt=None)

    def launch(self):

        func_lookup = {
            'q': self.quit,
            'h': self.help,
            'filter': self.filter,
            'showall': self.showall,
            'args': self.args,
            'rmfilters': self.rmfilters,
            'viewfilters': self.viewfilters,
            'compare': self.compare,
            'show': self.show,
            'search': self.search,
            'delete': self.delete,
        }

        while True:
            print "=============== Experiment Records =================="
            # print tabulate([[i]+e.get_row() for i, e in enumerate(record_infos)], headers=['#']+_ExperimentInfo.get_headers())
            print self.get_record_table(self.record_ids)
            print '-----------------------------------------------------'

            if self.experiment_names is not None or len(self.filters) != 0:
                print 'Not showing all experiments.  Type "rmfilters" to clear filters, or "viewfilters" to view current filters.'
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

    def showall(self):
        self.filters = []
        self.names = None

    def args(self, user_range):
        print self.get_record_table(self._select_records(user_range), headers=['Identifier', 'Args'])

    def rmfilters(self):
        self.filters = []

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


if __name__ == '__main__':
    browse_experiment_records()
