import shlex
from collections import OrderedDict

from artemis.experiments.experiment_record import GLOBAL_EXPERIMENT_LIBRARY, get_latest_experiment_identifier, \
    show_experiment, get_all_experiment_ids, clear_experiments, get_experiment_record
from artemis.general.should_be_builtins import separate_common_items, bad_value
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


def _warn_with_prompt(message, prompt = 'Press Enter to continue'):
    raw_input('%s\n  (%s) >> ' % (message, prompt))


def find_experiment(search_term):
    """
    Find an experiment.  Invoke
    :param search_term: A term that will be used to search for an experiment.
    :return:
    """
    found_experiments = OrderedDict((name, ex) for name, ex in GLOBAL_EXPERIMENT_LIBRARY.iteritems() if search_term in name)
    if len(found_experiments)==0:
        raise Exception("None of the {} experiments matched the search: '{}'".format(len(GLOBAL_EXPERIMENT_LIBRARY), search_term))
    elif len(found_experiments)>1:
        raise Exception("More than one experiment matched the search '{}', you need to be more specific.  Found: {}".format(search_term, found_experiments.keys()))
    else:
        return found_experiments.values()[0]

def browse_experiments(catch_errors = False, close_after_run = False, run_args = {}):
    while True:
        listing = _get_experiment_listing()
        print "==================== Experiments ===================="
        print '\n'.join(['%s : %s' % (identifier, name) for identifier, name in listing.iteritems()])
        print '-----------------------------------------------------'
        cmd = raw_input('Enter command or experiment # to run (h for help) >> ').lstrip(' ').rstrip(' ')

        try:
            split = cmd.split(' ')
            if len(split)==2:
                cmd, number = split
                if number not in listing:
                    _warn_with_prompt('No experiment with number "%s"' % (number, ))
                else:
                    name = listing[number]
                if cmd == 'run':
                    exp = GLOBAL_EXPERIMENT_LIBRARY[name].run(**run_args)
                    if close_after_run:
                        break
                elif cmd == 'show':
                    last_identifier = get_latest_experiment_identifier(name)
                    if last_identifier is None:
                        _warn_with_prompt("No record for experiment '%s' exists yet.  Run it to create one." % (name, ))
                    else:
                        show_experiment(get_latest_experiment_identifier(name))
                        _warn_with_prompt('')
                elif cmd == 'display':
                    try:
                        GLOBAL_EXPERIMENT_LIBRARY[name].display_last()
                    except Exception as err:
                        _warn_with_prompt("Error: %s: %s" % (err.__class__.__name__, err.message))
                else:
                    _warn_with_prompt("Unrecognised command '%s'" % (cmd, ))
            elif len(split)==1:
                if cmd == 'h':
                    _warn_with_prompt("  Enter '4' to run experiment 4\n  Enter 'show 4' to show the output from the last run of experiment 4 (if it has been run already).\n  "
                        "Enter 'records' to browse through all experiment records.\n  Enter 'display 4' to replot the result of experiment 4 (if it has been run already, "
                        "only works if you've defined a display function for the experiment.)\n  Enter 'q' to quit.",
                        prompt = 'Press Enter to exit help.')
                elif cmd == 'q':
                    break
                elif cmd == 'records':
                    browse_experiment_records()
                else:
                    if cmd not in listing:
                        _warn_with_prompt('No experiment with number "%s"' % (cmd, ))
                    else:
                        name = listing[cmd]
                        GLOBAL_EXPERIMENT_LIBRARY[name].run(**run_args)
            else:
                _warn_with_prompt('You must either enter a number to run the experiment, or "cmd #" to do some operation.  See help.')
        except Exception as e:
            if catch_errors:
                res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
                if res == 'e':
                    raise
            else:
                raise


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


def browse_experiment_records():

    ids = get_all_experiment_ids()
    while True:

        print "=============== Experiment Records =================="
        print '\n'.join(['%s: %s' % (i, exp_id) for i, exp_id in enumerate(ids)])
        print '-----------------------------------------------------'

        user_input = raw_input('Enter Command (show # to show and experiment, or h for help) >>')
        parts = shlex.split(user_input)

        cmd = parts[0]
        args = parts[1:]

        try:
            if cmd == 'q':
                break
            elif cmd == 'h':
                _warn_with_prompt('q: Quit\nfilter <text>: filter experiments\nrmfilters: Remove all filters\nshow <number> show experiment with number\ncompare #1 #2 #3: Compare experiments by their numbers.')
            elif cmd == 'filter':
                filter_text, = args
                ids = get_all_experiment_ids(filter_text)
            elif cmd == 'rmfilters':
                ids = get_all_experiment_ids()
            elif cmd == 'compare':
                indices = [int(arg) for arg in args]
                identifiers = [ids[ix] for ix in indices]
                compare_experiment_records(identifiers)
                _warn_with_prompt('')
            elif cmd == 'show':
                index, = args
                exp_id = ids[int(index)]
                show_experiment(exp_id)
                _warn_with_prompt('')
            elif cmd == 'clearall':
                conf = raw_input("Going to clear all experiment records.  Enter 'y' to confirm: ")
                if conf=='y':
                    clear_experiments()
                    ids = get_all_experiment_ids()
                    assert len(ids)==0, "Failed to delete them?"
                    print "Deleted all experiments"
                else:
                    print "Did not delete experiments"
            else:
                _warn_with_prompt('Bad Command: %s.' % cmd)
        except Exception as e:
            res = raw_input('%s: %s\nEnter "e" to view the stacktrace, or anything else to continue.' % (e.__class__.__name__, e.message))
            if res == 'e':
                raise


if __name__ == '__main__':
    browse_experiment_records()
