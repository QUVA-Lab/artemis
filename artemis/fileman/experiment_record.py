from collections import OrderedDict
from datetime import datetime
import inspect
import shlex
from IPython.core.magics import logging
from general.test_mode import is_test_mode, TestMode
import os
import pickle
from IPython.core.display import display, HTML
from fileman.local_dir import format_filename, make_file_dir, get_local_path, get_relative_path
from fileman.notebook_plots import show_embedded_figure
from fileman.notebook_utils import get_local_server_dir
from fileman.notebook_utils import get_relative_link_from_relative_path
from fileman.persistent_print import capture_print
from fileman.saving_plots import clear_saved_figure_locs, get_saved_figure_locs, \
    set_show_callback, always_save_figures, show_saved_figure
import matplotlib.pyplot as plt
import re

__author__ = 'peter'


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


ExperimentLibrary = _ExpLibClass()

GLOBAL_EXPERIMENT_LIBRARY = {}


def _am_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


class ExperimentRecord(object):
    """
    Captures all logs and figures generated, and saves the result.  Usage:

    with Experiment() as exp_1:
        do_stuff()
        plot_figures()

    exp_1.show_all_figures()
    """
    VERSION = 0  # We keep this in case we want to change this class, and need record the fact that it is an old version
    # when unpickling.

    def __init__(self, name = 'unnamed', filename = '%T-%N', print_to_console = False, save_result = None, show_figs = None):
        """
        :param name: Base-name of the experiment
        :param filename: Format of the filename (placeholders: %T is replaced by time, %N by name)
        :param experiment_dir: Relative directory (relative to data dir) to save this experiment when it closes
        :param print_to_console: If True, print statements still go to console - if False, they're just rerouted to file.
        :param show_figs: Show figures when the experiment produces them.  Can be:
            'hang': Show and hang
            'draw': Show but keep on going
            False: Don't show figures
            None: 'draw' if in test mode, else 'hang'
        """
        now = datetime.now()
        if save_result is None:
            save_result = not is_test_mode()

        if show_figs is None:
            show_figs = 'draw' if is_test_mode() else 'hang'

        assert show_figs in ('hang', 'draw', False)

        self._experiment_identifier = format_filename(file_string = filename, base_name=name, current_time = now)
        self._log_file_name = format_filename('%T-%N', base_name = name, current_time = now)
        self._has_run = False
        self._print_to_console = print_to_console
        self._save_result = save_result
        self._show_figs = show_figs

    def __enter__(self):
        clear_saved_figure_locs()
        if self._show_figs == 'draw':
            plt.ion()
        else:
            plt.ioff()
        self._log_file_path = capture_print(True, to_file = True, log_file_path = self._log_file_name, print_to_console = self._print_to_console)
        always_save_figures(show = self._show_figs, print_loc = False, name = self._experiment_identifier+'-%N')
        global _CURRENT_EXPERIMENT
        _CURRENT_EXPERIMENT = self._experiment_identifier
        return self

    def __exit__(self, *args):
        # On exit, we read the log file.  After this, the log file is no longer associated with the experiment.
        capture_print(False)

        with open(get_local_path(self._log_file_path)) as f:
            self._captured_logs = f.read()

        set_show_callback(None)
        self._captured_figure_locs = get_saved_figure_locs()

        self._has_run = True

        global _CURRENT_EXPERIMENT
        _CURRENT_EXPERIMENT = None

        if self._save_result:
            file_path = get_local_experiment_path(self._experiment_identifier)
            make_file_dir(file_path)
            with open(file_path, 'w') as f:
                pickle.dump(self, f)
                print 'Saving Experiment "%s"' % (self._experiment_identifier, )

    def get_identifier(self):
        return self._experiment_identifier

    def get_logs(self):
        return self._captured_logs

    def get_figure_locs(self):
        return self._captured_figure_locs

    def show_figures(self):
        for loc in self._captured_figure_locs:
            if _am_in_ipython():
                rel_loc = get_relative_link_from_relative_path(loc)
                show_embedded_figure(rel_loc)
            else:
                show_saved_figure(loc)

    def show(self):
        if _am_in_ipython():
            display(HTML("<a href = '%s' target='_blank'>View Log File for this experiment</a>"
                         % get_relative_link_from_relative_path(self._log_file_path)))
        else:
            self.print_logs()
        self.show_figures()

    def print_logs(self):
        print self._captured_logs

    def get_file_path(self):
        return get_local_experiment_path(self._experiment_identifier)

    def end_and_show(self):
        if not self._has_run:
            self.__exit__()
        self.show()

    def __str__(self):
        return '<ExperimentRecord object %s at %s>' % (self._experiment_identifier, hex(id(self)))


_CURRENT_EXPERIMENT = None

def get_current_experiment_id():
    """
    :return: A string identifying the current experiment
    """
    if _CURRENT_EXPERIMENT is None:
        raise Exception("No experiment is currently running!")
    return _CURRENT_EXPERIMENT


def start_experiment(*args, **kwargs):
    exp = ExperimentRecord(*args, **kwargs)
    exp.__enter__()
    return exp


def run_experiment(name, exp_dict = GLOBAL_EXPERIMENT_LIBRARY, print_to_console = True, show_figs = None, **experiment_record_kwargs):
    """
    Run an experiment and save the results.  Return a string which uniquely identifies the experiment.
    You can run the experiment agin later by calling show_experiment(location_string):

    :param name: The name for the experiment (must reference something in exp_dict)
    :param exp_dict: A dict<str:func> where funcs is a function with no arguments that run the experiment.
    :param experiment_record_kwargs: Passed to ExperimentRecord.

    :return: A location_string, uniquely identifying the experiment.
    """

    if isinstance(exp_dict, dict):
        assert name in exp_dict, 'Could not find experiment "%s" in the experiment dictionary with keys %s' % (name, exp_dict.keys())
        func = exp_dict[name]
    else:
        assert hasattr(exp_dict, '__call__')
        func = exp_dict

    experiment = exp_dict[name]

    if isinstance(experiment, Experiment):
        return experiment.run(print_to_console=print_to_console, show_figs=show_figs, **experiment_record_kwargs)
    else:
        logging.warn('DEPRECATED: Switch to register_experiment.')
        with ExperimentRecord(name = name, print_to_console=print_to_console, show_figs=show_figs, **experiment_record_kwargs) as exp_rec:
            print '%s Running Experiment: %s %s' % ('='*10, name, '='*10)
            func()
            print '%s Done Experiment: %s %s' % ('-'*11, name, '-'*12)
        return exp_rec


def run_notebook_experiment(name, exp_dict, print_to_console=False, show_figs=False, **experiment_record_kwargs):
    """
    Run an experiment with settings more suited to an IPython notebook.  Here, we want to redirect all
    output to a log file, and not show the figures immediately.
    """
    return run_experiment(name, exp_dict, print_to_console = print_to_console, show_figs = show_figs, **experiment_record_kwargs)


def get_local_experiment_path(identifier):
    return format_filename(identifier, directory = get_local_path('experiments'), ext = 'exp.pkl')


def get_experiment_record(identifier):
    local_path = get_local_experiment_path(identifier)
    assert os.path.exists(local_path), "Couldn't find experiment '%s' at '%s'" % (identifier, local_path)
    with open(local_path) as f:
        exp_rec = pickle.load(f)
    return exp_rec


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


def get_or_run_notebook_experiment(name, exp_dict, display_module = True, force_compute = False, **notebook_experiment_record_kwargs):
    """
    Get the latest experiment with the given name,
    :param name: Name of the experiment
    :param exp_dict: Dictionary of experiments to chose from
    :param force_compute: Recompute the experiment no matter what
    :param notebook_experiment_record_kwargs:
    :return:
    """
    exp_id = get_latest_experiment_identifier(name=name)

    recompute = exp_id is None or force_compute

    if display_module:
        func = exp_dict[name]
        if hasattr(inspect.getmodule(func), '__file__'):
            module_rel_path = inspect.getmodule(func).__file__
            if module_rel_path.endswith('.pyc'):
                module_rel_path = module_rel_path[:-1]
            module_name = inspect.getmodule(func).__name__
            server_path = get_local_server_dir()
            rel_path = get_relative_path(module_rel_path, server_path)
            if recompute:
                display(HTML("Running Experiment %s from module <a href = '/edit/%s' target='_blank'>%s</a>" % (name, rel_path, module_name)))
            else:
                display(HTML("Showing Completed Experiment %s from module <a href = '/edit/%s' target='_blank'>%s</a>" % (exp_id, rel_path, module_name)))

    if recompute:
        exp = run_notebook_experiment(name, exp_dict, **notebook_experiment_record_kwargs)
    else:
        exp = load_experiment(exp_id)
    return exp


def get_latest_experiment_identifier(name, template = '%T-%N'):
    """
    Show results of the latest experiment matching the given template.
    :param name: The experiment name
    :param template: The template which turns a name into an experiment identifier
    :return: A string identifying the latest matching experiment, or None, if not found.
    """
    named_template = template.replace('%N', re.escape(name))
    expr = named_template.replace('%T', '\d\d\d\d\.\d\d\.\d\d\T\d\d\.\d\d\.\d\d\.\d\d\d\d\d\d')
    expr = '^' + expr + '$'
    matching_experiments = get_all_experiment_ids(expr)
    if len(matching_experiments) == 0:
        return None
    else:
        latest_experiment_id = sorted(matching_experiments)[-1]
        return latest_experiment_id


def show_latest_results(experiment_name, template = '%T-%N'):
    print GLOBAL_EXPERIMENT_LIBRARY[experiment_name]
    experiment_record_identifier = get_latest_experiment_identifier(experiment_name, template)
    if experiment_record_identifier is None:
        raise Exception('No records for experiment "%s" exist.' % (experiment_name, ))
    show_experiment(experiment_record_identifier)


def load_experiment(experiment_identifier):
    """
    Load an ExperimentRecord based on the identifier
    :param experiment_identifier: A string identifying the experiment
    :return: An ExperimentRecord object
    """
    full_path = get_local_experiment_path(identifier=experiment_identifier)
    with open(full_path) as f:
        exp = pickle.load(f)
    return exp


def get_all_experiment_ids(expr = None):
    """
    :param expr: A regexp for matching experiments
        None if you just want all of them
    :return: A list of experiment identifiers.
    """

    expdir = get_local_path('experiments')
    experiments = [e[:-len('.exp.pkl')] for e in os.listdir(expdir) if e.endswith('.exp.pkl')]
    if expr is not None:
        experiments = [e for e in experiments if re.match(expr, e)]
    return experiments


def register_experiment(name, **kwargs):
    """ See Experiment """
    assert name not in GLOBAL_EXPERIMENT_LIBRARY, 'An experiment with name "%s" has already been registered!' % (name, )
    experiment = Experiment(name = name, **kwargs)
    GLOBAL_EXPERIMENT_LIBRARY[name] = experiment
    return experiment


def browse_experiment_records():

    ids = get_all_experiment_ids()
    while True:
        print '\n'.join(['%s: %s' % (i, exp_id) for i, exp_id in enumerate(ids)])

        user_input = raw_input('Enter Command (show # to show and experiment, or h for help) >>')
        parts = shlex.split(user_input)

        cmd = parts[0]
        args = parts[1:]

        try:
            if cmd == 'q':
                break
            elif cmd == 'h':
                print 'q: Quit\nfilter <text>: filter experiments\brmfilters: Remove all filters\nshow <number> show experiment with number'
                wait_for_continue()
            elif cmd == 'filter':
                filter_text, = args
                ids = get_all_experiment_ids(filter_text)
            elif cmd == 'rmfilters':
                ids = get_all_experiment_ids()
            elif cmd == 'show':
                index, = args
                exp_id = ids[int(index)]
                show_experiment(exp_id)
                wait_for_continue()
            else:
                print 'Bad Command: %s.' % cmd
                wait_for_continue()
        except Exception as e:
            res = raw_input('%s: %s\nEnter "e" to view the message, or anything else to continue.' % (e.__class__.__name__, e.message))
            if res == 'e':
                raise


def wait_for_continue():
    raw_input('<Press Enter to Continue>')


def get_experiment_info(name):
    experiment = GLOBAL_EXPERIMENT_LIBRARY[name]
    return str(experiment)


class Experiment(object):

    def __init__(self, function, description='', conclusion = '', name = None, versions = None, current_version = None):
        if versions is not None:
            assert isinstance(versions, (list, dict))
            assert current_version is not None, 'If you specify multiple versions, you have to pick a current version'
        if isinstance(versions, list):
            assert isinstance(current_version, int)
        self.name = name
        self.function = function
        self.description = description
        self.conclusion = conclusion
        self.versions = versions
        self.current_version = current_version

    def __str__(self):
        return 'Experiment: %s\n  Defined in: %s\n  Description: %s\n  Conclusion: %s' % \
            (self.name, inspect.getmodule(self.function).__name__, self.description, self.conclusion)

    def run(self, print_to_console = True, show_figs = None, test_mode=False, **experiment_record_kwargs):
        """
        Run the experiment, and return the ExperimentRecord that is generated.
        Note, if you want the output of the function, you should just run the function directly.
        :param experiment_record_kwargs: See ExperimentRecord for kwargs
        """
        if self.versions is not None:
            assert self.current_version in self.versions, "Experiment %s: Your current version: '%s' is not in the list of versions: %s" % (self.name, self.current_version, self.versions.keys())
            kwargs = self.versions[self.current_version]
            name = self.name+'-'+(self.current_version if isinstance(self.current_version, str) else str(self.versions[self.current_version]))
        else:
            kwargs = {}
            name = self.name

        if test_mode:
            with TestMode():
                print '%s Testing Experiment: %s %s' % ('='*10, name, '='*10)
                with ExperimentRecord(name = name, print_to_console=print_to_console, show_figs=show_figs, **experiment_record_kwargs) as exp_rec:
                    self.function(**kwargs)
                print '%s Done Testing Experiment: %s %s' % ('-'*11, name, '-'*12)
        else:
            print '%s Running Experiment: %s %s' % ('='*10, name, '='*10)
            with ExperimentRecord(name = name, print_to_console=print_to_console, show_figs=show_figs, **experiment_record_kwargs) as exp_rec:
                self.function(**kwargs)
            print '%s Done Experiment: %s %s' % ('-'*11, name, '-'*12)
        return exp_rec

    def run_all(self, **kwargs):
        for v in (self.versions.keys() if isinstance(self.versions, dict) else xrange(len(self.versions))):
            self.current_version = v
            self.run(**kwargs)

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def test_all(self, **kwargs):
        self.run_all(test_mode=True, **kwargs)


if __name__ == '__main__':
    browse_experiment_records()
