"""
Using this module, you can turn your main function into an "Experiment", which, when run, stores all console output, plots,
and computed results to disk (in ~/.artemis/experiments)

For a simple demo, see artemis/fileman/demo_experiments.py

Any function that can be called alone with no arguments can be turned into an experiment using the @experiment_function
decorator:

    @experiment_function
    def my_experiment(a=1, b=2, c=3):
        ...

This turns the function into an Experiment object, which has the following methods:

    result = my_experiment.run()
        Run the function, and save all text outputs and plots to disk.
        Console output, plots, and result are stored in: ~/.artemis/experiments

    ex=my_experiment.add_variant(name, **kwargs)
        Add a variant to the experiment, with new arguments that override any existing ones.
        The variant is itself an experiment, and can have variants of its own.

    ex=my_experiment.get_variant(name, *sub_names)
        Get a variant of an experiment (or one of its sub-variants).

To open up a menu where you can see and run all experiments (and their variants) that have been created run:

    browse_experiments()

To browse through records of experiments that have been run, either run this file, or call the function:

    browse_experiment_records()

You may not want an experiment to be runnable (and show up in browse_experiments) - you may just want it
to serve as a basis on which to make variants.  In this case, you can decorate with

    @experiment_root_function
    def my_experiment_root(a=1, b=2, c=3):
        ...

If you want even the variants of this experiment to be roots for other experiments, you can use method

    ex=my_experiment.add_root_variant(name, **kwargs)

To run your experiment (if not a root) and all non-root variants (and sub variants, and so on) of an experiment, run:

    my_experiment.run_all()

If your experiment takes a long time to run, you may not want to run it every time you want to see the plots of
results (this is especially so if you are refining your plotting function, and don't want to run from scratch just to
check your visualization methods).  in this case, you can do the following:

    def my_display_function(data_to_display):
        ...  # Plot data

    @ExperimentFunction(display_function = my_display_function)
    def my_experiment(a=1, b=2, c=3):
        ...
        return data_to_display

Now, you can use the method:

    my_experiment.display_last()
        Load the last results for this experiment (if any) and plot them again.

"""

import atexit
from collections import OrderedDict
from datetime import datetime
from functools import partial
import inspect
import shlex
import pickle
import shutil
import tempfile
from contextlib import contextmanager
import os
import re
from artemis.general.test_mode import is_test_mode, set_test_mode
from artemis.fileman.local_dir import format_filename, make_file_dir, get_local_path, make_dir
from artemis.fileman.persistent_print import PrintAndStoreLogger
import logging
import time
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')
ARTEMIS_LOGGER.setLevel(logging.INFO)


__author__ = 'peter'

def experiment_function(f):
    """
    Use this decorator (@experiment_function) on a function that you want to run.  e.g.

        @experiment_function
        def demo_my_experiment(a=1, b=2, c=3):
            ...

    To run the experiment in a mode where it records all print statements and matplotlib figures, go:
    demo_my_experiment.run()

    To create a variant on the experiment, with different arguments, go:

        v1 = demo_my_experiment.add_variant('try_a=2', a=2)

    The variant v1 is then also an experiment, so you can go

        v1.run()

    """
    return ExperimentFunction()(f)


def experiment_root(f):
    """
    Use this decorator on a function that you want to build variants off of:

        @experiment_root
        def demo_my_experiment(a, b=2, c=3):
            ...

        demo_my_experiment.add_variant('a1', a=1)
        demo_my_experiment.add_variant('a2', a=2)

    The root experiment is not runnable by itself, and will not appear in the list when you
    browse experiments.
    """
    return ExperimentFunction(is_root=True)(f)


class ExperimentFunction(object):
    """ Decorator for an experiment
    """
    def __init__(self, display_function = None, info=None, is_root = False):
        self.display_function = display_function
        self.info = info
        self.is_root = is_root

    def __call__(self, f):
        f.is_base_experiment = True
        ex = Experiment(
            name = f.__name__,
            function = f,
            display_function=self.display_function,
            info='Root Experiment: {name}\nDefined in: {file}\n'.format(name=f.__name__, file=inspect.getmodule(f).__file__ ),
            is_root = self.is_root
            )
        return ex


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


def browse_experiments(catch_errors = True, run_args = {}):
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
                _warn_with_prompt('q: Quit\nfilter <text>: filter experiments\nrmfilters: Remove all filters\nshow <number> show experiment with number')
            elif cmd == 'filter':
                filter_text, = args
                ids = get_all_experiment_ids(filter_text)
            elif cmd == 'rmfilters':
                ids = get_all_experiment_ids()
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



GLOBAL_EXPERIMENT_LIBRARY = OrderedDict()


class ExperimentRecord(object):

    def __init__(self, experiment_directory):
        self._experiment_directory = experiment_directory

    def show_figures(self):
        from artemis.plotting.saving_plots import show_saved_figure
        for loc in self.get_figure_locs():
            show_saved_figure(loc)

    def get_log(self):
        log_file_path = os.path.join(self._experiment_directory, 'output.txt')
        assert os.path.exists(log_file_path), 'No output file found.  Maybe "%s" is not an experiment directory?' % (self._experiment_directory, )
        with open(log_file_path) as f:
            text = f.read()
        return text

    def get_figure_locs(self, include_directory = True):
        locs = [f for f in os.listdir(self._experiment_directory) if f.startswith('fig-')]
        if include_directory:
            return [os.path.join(self._experiment_directory, f) for f in locs]
        else:
            return locs

    def show(self):
        print '{header} Showing Experiment {header}\n{info}{subborder} Logs {subborder}\n{log}\n{border}'.format(header="="*20, border="="*50, info=self.get_info(), subborder='-'*20, log=self.get_log())
        self.show_figures()

    def get_info(self):
        with open(os.path.join(self._experiment_directory, 'info.txt')) as f:
            data = f.read()
        return data

    def add_info(self, more_info):
        with open(os.path.join(self._experiment_directory, 'info.txt'), 'a') as f:
            f.write('%s\n' % (more_info, ))

    def get_result(self):
        result_loc = os.path.join(self._experiment_directory, 'result.pkl')
        if os.path.exists(result_loc):
            with open(result_loc) as f:
                result = pickle.load(f)
            return result
        else:
            return None

    def set_result(self, result):
        file_path = get_local_experiment_path(os.path.join(self._experiment_directory, 'result.pkl'))
        make_file_dir(file_path)
        with open(file_path, 'w') as f:
            pickle.dump(result, f, protocol=2)
            print 'Saving Result for Experiment "%s"' % (self.get_identifier(), )

    def get_identifier(self):
        root, identifier = os.path.split(self._experiment_directory)
        return identifier

    def get_dir(self):
        return self._experiment_directory

    def delete(self):
        shutil.rmtree(self._experiment_directory)


_CURRENT_EXPERIMENT_RECORD = None

@contextmanager
def record_experiment(identifier='%T-%N', name = 'unnamed', info = '', print_to_console = True, show_figs = None,
            save_figs = True, saved_figure_ext = '.pdf', use_temp_dir = False):
    """
    :param identifier: The string that uniquely identifies this experiment record.  Convention is that it should be in
        the format
    :param name: Base-name of the experiment
    :param print_to_console: If True, print statements still go to console - if False, they're just rerouted to file.
    :param show_figs: Show figures when the experiment produces them.  Can be:
        'hang': Show and hang
        'draw': Show but keep on going
        False: Don't show figures
    """

    identifier = format_filename(file_string = identifier, base_name=name, current_time = datetime.now())

    if show_figs is None:
        show_figs = 'draw' if is_test_mode() else 'hang'

    assert show_figs in ('hang', 'draw', False)

    if use_temp_dir:
        experiment_directory = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(experiment_directory))
    else:
        experiment_directory = get_local_path('experiments/{identifier}'.format(identifier=identifier))

    make_dir(experiment_directory)
    make_file_dir(experiment_directory)
    log_file_name = os.path.join(experiment_directory, 'output.txt')
    log_capture_context = PrintAndStoreLogger(log_file_path = log_file_name, print_to_console = print_to_console)
    log_capture_context.__enter__()
    from artemis.plotting.manage_plotting import WhatToDoOnShow
    blocking_show_context = WhatToDoOnShow(show_figs)
    blocking_show_context.__enter__()
    if save_figs:
        from artemis.plotting.saving_plots import SaveFiguresOnShow
        figure_save_context = SaveFiguresOnShow(path = os.path.join(experiment_directory, 'fig-%T-%L'+saved_figure_ext))
        figure_save_context.__enter__()

    _register_current_experiment(name, identifier)

    global _CURRENT_EXPERIMENT_RECORD
    _CURRENT_EXPERIMENT_RECORD = ExperimentRecord(experiment_directory)
    _CURRENT_EXPERIMENT_RECORD.add_info('Name: %s' % (name, ))
    _CURRENT_EXPERIMENT_RECORD.add_info('Identifier: %s' % (identifier, ))
    _CURRENT_EXPERIMENT_RECORD.add_info('Directory: %s' % (_CURRENT_EXPERIMENT_RECORD.get_dir(), ))
    yield _CURRENT_EXPERIMENT_RECORD
    _CURRENT_EXPERIMENT_RECORD = None

    blocking_show_context.__exit__(None, None, None)
    log_capture_context.__exit__(None, None, None)
    if save_figs:
        figure_save_context.__exit__(None, None, None)

    _deregister_current_experiment()


_CURRENT_EXPERIMENT_ID = None
_CURRENT_EXPERIMENT_NAME = None


def _register_current_experiment(name, identifier):
    """
    For keeping track of the current running experiment, assuring that no two experiments are running at the same time.
    :param name:
    :param identifier:
    :return:
    """
    global _CURRENT_EXPERIMENT_ID
    global _CURRENT_EXPERIMENT_NAME
    assert _CURRENT_EXPERIMENT_ID is None, "You cannot start experiment '%s' until experiment '%s' has been stopped." % (_CURRENT_EXPERIMENT_ID, identifier)
    _CURRENT_EXPERIMENT_NAME = name
    _CURRENT_EXPERIMENT_ID = identifier


def _deregister_current_experiment():
    global _CURRENT_EXPERIMENT_ID
    global _CURRENT_EXPERIMENT_NAME
    _CURRENT_EXPERIMENT_ID = None
    _CURRENT_EXPERIMENT_NAME = None


def get_current_experiment_id():
    """
    :return: A string identifying the current experiment
    """
    if _CURRENT_EXPERIMENT_ID is None:
        raise Exception("No experiment is currently running!")
    return _CURRENT_EXPERIMENT_ID


def get_current_experiment_name():
    if _CURRENT_EXPERIMENT_NAME is None:
        raise Exception("No experiment is currently running!")
    return _CURRENT_EXPERIMENT_NAME


def run_experiment(name, exp_dict = GLOBAL_EXPERIMENT_LIBRARY, **experiment_record_kwargs):
    """
    Run an experiment and save the results.  Return a string which uniquely identifies the experiment.
    You can run the experiment agin later by calling show_experiment(location_string):

    :param name: The name for the experiment (must reference something in exp_dict)
    :param exp_dict: A dict<str:func> where funcs is a function with no arguments that run the experiment.
    :param experiment_record_kwargs: Passed to ExperimentRecord.

    :return: A location_string, uniquely identifying the experiment.
    """
    experiment = exp_dict[name]
    return experiment.run(**experiment_record_kwargs)


def _get_matching_template_from_experiment_name(experiment_name, version = None, template = '%T-%N'):
    if version is not None:
        experiment_name = experiment_name + '-' + version
    named_template = template.replace('%N', re.escape(experiment_name))
    expr = named_template.replace('%T', '\d\d\d\d\.\d\d\.\d\d\T\d\d\.\d\d\.\d\d\.\d\d\d\d\d\d')
    expr = '^' + expr + '$'
    return expr


def clear_experiment_records_with_name(experiment_name=None):
    """
    Clear all experiment results.
    :param matching_expression:
    :return:
    """
    ids = get_all_experiment_ids(_get_matching_template_from_experiment_name(experiment_name))
    paths = [os.path.join(get_local_path('experiments'), identifier) for identifier in ids]
    for p in paths:
        shutil.rmtree(p)


def delete_experiment_with_id(experiment_identifier):
    if experiment_exists(experiment_identifier):
        get_experiment_record(experiment_identifier).delete()


def get_local_experiment_path(identifier):
    return os.path.join(get_local_path('experiments'), identifier)


def get_experiment_record(identifier):
    local_path = get_local_experiment_path(identifier)
    assert os.path.exists(local_path), "Couldn't find experiment '%s' at '%s'" % (identifier, local_path)
    return ExperimentRecord(local_path)


def experiment_exists(identifier):
    local_path = get_local_experiment_path(identifier)
    return os.path.exists(local_path)


def show_experiment(identifier):
    """
    Show the results of an experiment (plots and logs)
    :param identifier: A string uniquely identifying the experiment
    """
    exp_rec = get_experiment_record(identifier)
    exp_rec.show()


def merge_experiment_dicts(*dicts):
    """
    Merge dictionaries of experiments, checking that names are unique.
    """
    merge_dict = OrderedDict()
    for d in dicts:
        assert not any(k in merge_dict for k in d), "Experiments %s has been defined twice." % ([k for k in d.keys() if k in merge_dict],)
        merge_dict.update(d)
    return merge_dict


def get_latest_experiment_identifier(name, version=None, template = '%T-%N'):
    """
    Show results of the latest experiment matching the given template.
    :param name: The experiment name
    :param template: The template which turns a name into an experiment identifier
    :return: A string identifying the latest matching experiment, or None, if not found.
    """
    expr = _get_matching_template_from_experiment_name(name, version = version, template=template)
    matching_experiments = get_all_experiment_ids(expr)
    if len(matching_experiments) == 0:
        return None
    else:
        latest_experiment_id = sorted(matching_experiments)[-1]
        return latest_experiment_id


def get_lastest_result(experiment_name, version = None):
    return get_latest_experiment_record(experiment_name, version).get_result()


def get_latest_experiment_record(experiment_name, version=None):
    experiment_record_identifier = get_latest_experiment_identifier(experiment_name, version=version)
    if experiment_record_identifier is None:
        raise Exception("No saved records for experiment '{name}', version '{version}'".format(name=experiment_name, version=version))
    exp_rec = get_experiment_record(experiment_record_identifier)
    return exp_rec


def load_experiment(experiment_identifier):
    """
    Load an ExperimentRecord based on the identifier
    :param experiment_identifier: A string identifying the experiment
    :return: An ExperimentRecord object
    """
    full_path = get_local_experiment_path(identifier=experiment_identifier)
    return ExperimentRecord(full_path)


def get_all_experiment_ids(expr = None):
    """
    :param expr: A regexp for matching experiments
        None if you just want all of them
    :return: A list of experiment identifiers.
    """

    expdir = get_local_path('experiments')
    experiments = [e for e in os.listdir(expdir) if os.path.isdir(os.path.join(expdir, e))]
    if expr is not None:
        experiments = [e for e in experiments if re.match(expr, e)]
    return experiments


def _register_experiment(experiment):
    GLOBAL_EXPERIMENT_LIBRARY[experiment.name] = experiment




def clear_experiments():
    # Credit: http://stackoverflow.com/questions/185936/delete-folder-contents-in-python
    folder = get_local_path('experiments')
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def get_experiment_info(name):
    experiment = GLOBAL_EXPERIMENT_LIBRARY[name]
    return str(experiment)


def get_experiment(name):
    return GLOBAL_EXPERIMENT_LIBRARY[name]


class Experiment(object):

    def __init__(self, function=None, display_function = None, info='', conclusion = '', name = None, is_root = False):
        """
        :param function: The function defining the experiment
        :param display_function: A function that can be called to display the results returned by function.
            This can be useful if you just want to re-run the display for already-computed and saved results.
            To do this, go experiment.save_last()
        :parsam info: Info to be added to the experiment
        :param conclusion: <Deprecated> will be removed in future
        :param name: Nmae of this experiment.
        """
        self.name = name
        self.function = function
        self.info = info
        self.display_function = display_function
        self.variants = OrderedDict()
        self._notes = []
        self.is_root = is_root
        if not is_root:
            _register_experiment(self)

    def __call__(self, *args, **kwargs):
        """ Run the function as normal, without recording or anything. """
        # if (_CURRENT_EXPERIMENT_RECORD is not None) and not isinstance(self.function, partial):

        if (_CURRENT_EXPERIMENT_RECORD is not None) and hasattr(self.function, 'is_base_experiment'):
            # If we are in the wrapped function, and running it as an experiment, we log some metadata around the experiment run.
            # for the info file.
            r = _CURRENT_EXPERIMENT_RECORD
            start_time = time.time()
            try:
                arg_spec = inspect.getargspec(self.function)
                all_arg_names, _, _, defaults = arg_spec
                default_args = {k: v for k, v in zip(all_arg_names[len(all_arg_names)-(len(defaults) if defaults is not None else 0):], defaults if defaults is not None else [])}
                r.add_info('Args: %s' % (default_args, ))
            except Exception as err:
                ARTEMIS_LOGGER.error('Could not record arguments because %s: %s' % (err.__class__.__name__, err.message))
            r.add_info('Function: %s' % (self.function.__name__, ))
            r.add_info('Module: %s' % (inspect.getmodule(self.function).__file__, ))
            try:
                out = self.function(*args, **kwargs)
                r.add_info('Status: Ran Successfully')
            except Exception as err:
                r.add_info('Status: Had an Error: %s: %s' % (err.__class__.__name__, err.message))
                raise
            finally:
                fig_locs = _CURRENT_EXPERIMENT_RECORD.get_figure_locs(include_directory=False)
                r.add_info('Figures Generated: %s %s' % (len(fig_locs), fig_locs))
                r.add_info('Run Time (including plot hangs): %ss' % (time.time() - start_time))
                for n in self._notes:
                    r.add_info('Note: %s' % (n, ))
            return out
        else:
            return self.function(*args, **kwargs)

    def __str__(self):
        return 'Experiment: %s\n  Description: %s' % \
            (self.name, self.info)

    def run(self, print_to_console = True, show_figs = None, version = None, test_mode=None, keep_record=None, **experiment_record_kwargs):
        """
        Run the experiment, and return the ExperimentRecord that is generated.

        :param print_to_console: Print to console (as well as logging to file)
        :param show_figs: Show figures (as well as saving to file)
        :param version: String identifying which version of the experiment to run (refer to "versions" argument of __init__)
        :param test_mode: Run in "test_mode".  This sets the global "test_mode" flag when running the experiment.  This
            flag can be used to, for example, shorten a training session to verify that the code runs.  Can be:
                True: Run in test mode
                False: Don't run in test mode:
                None: Keep the current state of the global "is_test_mode()" flag.
        :param keep_record: Keep the folder that results are saved into.
                True: Results are saved into a folder
                False: Results folder is deleted at the end.
                None: If "test_mode" is true, then delete results at end, otherwise save them.
        :param experiment_record_kwargs: Passed to the "record_experiment" context.
        :return: The ExperimentRecord object, if keep_record is true, otherwise None
        """
        if test_mode is None:
            test_mode = is_test_mode()
        if keep_record is None:
            keep_record = not test_mode

        old_test_mode = is_test_mode()
        set_test_mode(test_mode)
        ARTEMIS_LOGGER.info('{border} {mode} Experiment: {name}{version} {border}'.format(border = '='*10, mode = "Testing" if test_mode else "Running", name=self.name, version=(' - '+version) if version is not None else ''))
        with record_experiment(name = self.name, info=self.info, print_to_console=print_to_console, show_figs=show_figs, use_temp_dir=not keep_record, **experiment_record_kwargs) as exp_rec:
            results = self()
            if self.display_function is not None:
                self.display_function(results)
        exp_rec.set_result(results)
        ARTEMIS_LOGGER.info('{border} Done {mode} Experiment: {name}{version} {border}'.format(border = '='*10, mode = "Testing" if test_mode else "Running", name=self.name, version=(' - '+version) if version is not None else ''))
        set_test_mode(old_test_mode)
        return exp_rec

    def _create_experiment_variant(self, name, (args, kwargs), is_root):
        ex = Experiment(
            name='.'.join((self.name, name)),
            function=partial(self, *args, **kwargs),
            display_function=self.display_function,
            info=self.info+'Variant: {variant}\n'.format(variant=name),
            is_root = is_root
            )
        self.variants[name] = ex
        return ex

    def add_variant(self, name, *args, **kwargs):
        """
        Add a variant to this experiment, and register it on the list of experiments.
        :param name: The name of the root variant.
        :param args, kwargs: Ordered/named arguments for this experiment variant
        :return: The experiment.
        """
        return self._create_experiment_variant(name, (args, kwargs), is_root = False)

    def add_root_variant(self, name, *args, **kwargs):
        """
        Add a variant to this experiment, but do NOT register it on the list of experiments.
        (A root variant is indended to have variants added on top of it).
        :param name: The name of the variant
        :param args, kwargs: Ordered/named arguments for this experiment variant
        :return: The experiment.
        """
        return self._create_experiment_variant(name, (args, kwargs), is_root=True)

    def add_note(self, note):
        """
        :param note:
        :return:
        """
        self._notes.append(str(note))
        return self

    def get_variant(self, name, *path):
        """
        Get a variant on this experiment.
        :param name: A the name of the variant
        :param path: Optionally, a list of names of subvariants (to call up a nested experiment)
        :return:
        """
        return self.variants[name].get_variant(*path) if len(path)>0 else self.variants[name]

    def display_last(self):
        assert self.display_function is not None, "You have not specified a display function for experiment: %s" % (self.name, )
        result = get_lastest_result(self.name)
        assert result is not None, "No result was computed for the last run of '%s'" % (self.name, )
        self.display_function(result)

    def run_all(self, **kwargs):
        """
        Run this experiment (if not a root-experiment) and all variants (if not roots).
        """
        if self.is_root:
            self.run()
        for v in self.variants:
            v.run_all(**kwargs)

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def test_all(self, **kwargs):
        self.run_all(test_mode=True, **kwargs)


# ALTERNATE INTERFACES.
# We keep these for backwards compatibility and to show how else we could organize the experiment API.
# These are out of use, as we at Artemis prefer the @experiment_function decorator.


# Register interface:

def register_experiment(name, function, description = None, conclusion = None, versions = None, current_version = None, **kwargs):
    """
    This is the old interface to experiments.  We keep it, for now, for the sake of
    backwards-compatibility.

    In the future, use the @experiment_function decorator instead.
    """
    info = ''
    if description is not None:
        info += 'Description: %s\n' % (description, )
    if conclusion is not None:
        info += 'Conclusion: %s\n' % (description, )

    if versions is not None:
        if current_version is None:
            assert len(versions)==1
            current_version = versions.keys()[0]
        assert current_version is not None
        function = partial(function, **versions[current_version])

    assert name not in GLOBAL_EXPERIMENT_LIBRARY, 'An experiment with name "%s" has already been registered!' % (name, )
    experiment = Experiment(name = name, function=function, info=info, **kwargs)
    GLOBAL_EXPERIMENT_LIBRARY[name] = experiment
    return experiment


_CURRENT_EXPERIMENT_CONTEXT = None


# Start/Stop Interface:

def start_experiment(*args, **kwargs):
    global _CURRENT_EXPERIMENT_CONTEXT
    _CURRENT_EXPERIMENT_CONTEXT = record_experiment(*args, **kwargs)
    return _CURRENT_EXPERIMENT_CONTEXT.__enter__()


def end_current_experiment():
    global _CURRENT_EXPERIMENT_CONTEXT
    _CURRENT_EXPERIMENT_CONTEXT.__exit__(None, None, None)
    _CURRENT_EXPERIMENT_CONTEXT = None


# Experiment Library interface:

class _ExpLibClass(object):

    def __setattr__(self, experiment_name, experiment):
        assert isinstance(experiment, Experiment), "Your experiment must be an experiment!"
        if experiment.name is None:
            experiment.name = experiment_name
        assert experiment_name not in GLOBAL_EXPERIMENT_LIBRARY, "Experiment %s is already in the library" % (experiment_name, )
        self.__dict__[experiment_name] = experiment
        GLOBAL_EXPERIMENT_LIBRARY[experiment_name] = experiment

    def get_experiments(self):
        return GLOBAL_EXPERIMENT_LIBRARY

    def __getattr__(self, name):
        if name in GLOBAL_EXPERIMENT_LIBRARY:
            return GLOBAL_EXPERIMENT_LIBRARY[name]
        else:
            return _ExperimentConstructor(name)


class _ExperimentConstructor(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        if self.name in GLOBAL_EXPERIMENT_LIBRARY:
            raise Exception("You tried to run create experiment '%s', but it already exists in the library.  Give it another name!" % (self.name, ))
        return register_experiment(name = self.name, **kwargs)

    def run(self):
        raise Exception("You tried to run experiment '%s', but it hasn't been made yet!" % (self.name, ))


ExperimentLibrary = _ExpLibClass()


if __name__ == '__main__':
    browse_experiment_records()
