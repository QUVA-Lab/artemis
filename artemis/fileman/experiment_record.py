import atexit
from collections import OrderedDict
from datetime import datetime
import inspect
import shlex
import pickle
import shutil
import tempfile
from decorator import contextmanager
import os
import re
from artemis.general.test_mode import is_test_mode, set_test_mode
from artemis.plotting.manage_plotting import WhatToDoOnShow
from artemis.plotting.saving_plots import SaveFiguresOnShow, show_saved_figure
from artemis.fileman.local_dir import format_filename, make_file_dir, get_local_path, make_dir
from artemis.fileman.persistent_print import PrintAndStoreLogger
import logging
logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')
ARTEMIS_LOGGER.setLevel(logging.INFO)


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

    def _get_experiment_listing(self):
        experiment_listing = OrderedDict()
        for i, (name, exp) in enumerate(GLOBAL_EXPERIMENT_LIBRARY.iteritems()):
            if exp.versions is None:
                experiment_listing['%s' % (i, )] = (name, None)
            else:
                assert len(exp.versions)<26, "Really?  You have more then 26 versions?  Time to code the z, aa, bb, ... system"
                for j, version in enumerate(exp.versions):
                    experiment_listing['%s%s' % (i, chr(ord('a')+j))] = (name, version)
        return experiment_listing

    def select_experiment(self):
        listing = self._get_experiment_listing()
        print '\n'.join(['%s : %s' % (identifier, name) if version is None else '%s: %s-%s' % (identifier, name, version) for identifier, (name, version) in listing.iteritems()])
        which_one = raw_input('Select Experiment >> ')
        if which_one.lstrip(' ').rstrip(' ') in listing:
            name, version = listing[which_one]
            GLOBAL_EXPERIMENT_LIBRARY[name].current_version = version  # HACK!
            return GLOBAL_EXPERIMENT_LIBRARY[name]
        else:
            raise Exception('No experiment with id: "%s"' % (which_one, ))

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
        # exp = Experiment(name=self.name, **kwargs)
        return register_experiment(name = self.name, **kwargs)

    def run(self):
        raise Exception("You tried to run experiment '%s', but it hasn't been made yet!" % (self.name, ))


ExperimentLibrary = _ExpLibClass()

GLOBAL_EXPERIMENT_LIBRARY = OrderedDict()


class ExperimentRecord(object):

    def __init__(self, experiment_directory):
        self._experiment_directory = experiment_directory

    def show_figures(self):
        for loc in self.get_figure_locs():
            show_saved_figure(loc)

    def get_log(self):
        log_file_path = os.path.join(self._experiment_directory, 'output.txt')
        assert os.path.exists(log_file_path), 'No output file found.  Maybe "%s" is not an experiment directory?' % (self._experiment_directory, )
        with open(log_file_path) as f:
            text = f.read()
        return text

    def get_figure_locs(self):
        return [os.path.join(self._experiment_directory, f) for f in os.listdir(self._experiment_directory) if f.startswith('fig-')]

    def show(self):
        print '{header}Showing Experiment{header}\n{info}\n{subborder}\n{log}\n{border}'.format(header="="*20, border="="*50, info=self.get_info_text(), subborder='-'*50, log=self.get_log())
        self.show_figures()

    def get_info(self):
        with open(os.path.join(self._experiment_directory, 'info.pkl')) as f:
            experiment_info = pickle.load(f)
        return experiment_info

    def get_info_text(self):
        return "name: {name}\nid: {id}\ndescription: {description}".format(**self.get_info())

    def get_result(self):
        result_loc = os.path.join(self._experiment_directory, 'output.txt')
        if os.path.exists(result_loc):
            with open(self._result_loc) as f:
                result = pickle.load(f)
            return result
        else:
            return None

    def set_result(self, result):
        file_path = get_local_experiment_path(os.path.join(self._experiment_directory, 'result.pkl'))
        make_file_dir(file_path)
        with open(file_path, 'w') as f:
            pickle.dump(result, f)
            print 'Saving Result for Experiment "%s"' % (self.get_identifier(), )

    def get_identifier(self):
        return self.get_info()['id']

    def get_dir(self):
        return self._experiment_directory

    def delete(self):
        shutil.rmtree(self._experiment_directory)


@contextmanager
def record_experiment(name = 'unnamed', filename = '%T-%N', description = '', print_to_console = True, show_figs = None,
            save_figs = True, saved_figure_ext = '.pdf', use_temp_dir = False):
    """
    :param name: Base-name of the experiment
    :param filename: Format of the filename (placeholders: %T is replaced by time, %N by name)
    :param experiment_dir: Relative directory (relative to data dir) to save this experiment when it closes
    :param print_to_console: If True, print statements still go to console - if False, they're just rerouted to file.
    :param show_figs: Show figures when the experiment produces them.  Can be:
        'hang': Show and hang
        'draw': Show but keep on going
        False: Don't show figures
    """

    now = datetime.now()
    if show_figs is None:
        show_figs = 'draw' if is_test_mode() else 'hang'

    assert show_figs in ('hang', 'draw', False)

    experiment_identifier = format_filename(file_string = filename, base_name=name, current_time = now)
    if use_temp_dir:
        experiment_directory = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(experiment_directory))
    else:
        experiment_directory = get_local_path('experiments/{identifier}'.format(identifier=experiment_identifier))
    make_dir(experiment_directory)
    make_file_dir(experiment_directory)
    log_file_name = os.path.join(experiment_directory, 'output.txt')

    with open(os.path.join(experiment_directory, 'info.pkl'), 'w') as f:
        pickle.dump({'name': name, 'id': experiment_identifier, 'description': description}, f)

    blocking_show_context = WhatToDoOnShow(show_figs)
    blocking_show_context.__enter__()
    log_capture_context = PrintAndStoreLogger(log_file_path = log_file_name, print_to_console = print_to_console)
    log_capture_context.__enter__()
    if save_figs:
        figure_save_context = SaveFiguresOnShow(path = os.path.join(experiment_directory, 'fig-%T-%L'+saved_figure_ext))
        figure_save_context.__enter__()

    _register_current_experiment(name, experiment_identifier)

    yield ExperimentRecord(experiment_directory)

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


def _get_matching_template_from_experiment_name(experiment_name, template = '%T-%N'):
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


def get_local_experiment_path(identifier):
    return os.path.join(get_local_path('experiments'), identifier)


def get_experiment_record(identifier):
    local_path = get_local_experiment_path(identifier)
    assert os.path.exists(local_path), "Couldn't find experiment '%s' at '%s'" % (identifier, local_path)
    return ExperimentRecord(local_path)


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


def get_latest_experiment_identifier(name, template = '%T-%N'):
    """
    Show results of the latest experiment matching the given template.
    :param name: The experiment name
    :param template: The template which turns a name into an experiment identifier
    :return: A string identifying the latest matching experiment, or None, if not found.
    """
    expr = _get_matching_template_from_experiment_name(name, template=template)
    matching_experiments = get_all_experiment_ids(expr)
    if len(matching_experiments) == 0:
        return None
    else:
        latest_experiment_id = sorted(matching_experiments)[-1]
        return latest_experiment_id


def show_latest_results(experiment_name, template = '%T-%N'):
    print GLOBAL_EXPERIMENT_LIBRARY[experiment_name]
    experiment_record_identifier =  get_latest_record_identifier(experiment_name, template)
    show_experiment(experiment_record_identifier)


def get_latest_record_identifier(experiment_name, template = None):
    if template is None:
        template = '%T-%N'
    experiment_record_identifier = get_latest_experiment_identifier(experiment_name, template)
    if experiment_record_identifier is None:
        raise Exception('No records for experiment "%s" exist.' % (experiment_name, ))
    return experiment_record_identifier


def get_lastest_result(experiment_name):
    return get_latest_experiment_record().get_result()


def get_latest_experiment_record(experiment_name):
    experiment_record_identifier = get_latest_experiment_identifier(experiment_name)
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

    def __init__(self, function, description='', conclusion = '', display_function = None, name = None, versions = None, current_version = None):
        if versions is not None:
            assert isinstance(versions, (list, dict))
        if isinstance(versions, list):
            assert isinstance(current_version, int)
        self.name = name
        self.function = function
        self.description = description
        self.conclusion = conclusion
        self.versions = versions
        self.current_version = current_version
        self.display_function = display_function

    def __str__(self):
        return 'Experiment: %s\n  Defined in: %s\n  Description: %s\n  Conclusion: %s' % \
            (self.name, inspect.getmodule(self.function).__name__, self.description, self.conclusion)

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
        version_to_run = self.current_version if version is None else version
        if self.versions is not None:
            if len(self.versions)==1:
                version_to_run = self.versions.keys()[0]
            assert version_to_run is not None, 'If you specify multiple versions, you have to pick a current version'
            assert version_to_run in self.versions, "Experiment %s: The version you're trying to run: '%s' is not in the list of versions: %s" % (self.name, version_to_run, self.versions.keys())
            kwargs = self.versions[version_to_run]
            name = self.name+'-'+(version_to_run if isinstance(version_to_run, str) else str(self.versions[version_to_run]))
        else:
            kwargs = {}
            name = self.name

        if test_mode is None:
            test_mode = is_test_mode()
        if keep_record is None:
            keep_record = not test_mode

        old_test_mode = is_test_mode()
        set_test_mode(test_mode)
        ARTEMIS_LOGGER.info('{border} {mode} Experiment: {name} {border}'.format(border = '='*10, mode = "Testing" if test_mode else "Running", name=self.name))
        with record_experiment(name = name, description=self.description, print_to_console=print_to_console, show_figs=show_figs, use_temp_dir=not keep_record, **experiment_record_kwargs) as exp_rec:
            results = self.function(**kwargs)
        exp_rec.set_result(results)
        ARTEMIS_LOGGER.info('{border} Done {mode} Experiment: {name} {border}'.format(border = '='*10, mode = "Testing" if test_mode else "Running", name=self.name))
        set_test_mode(old_test_mode)
        return exp_rec

    def display_last(self):
        assert self.display_function is not None, "You have not specified a display function for experiment: %s" % (self.name, )
        result = get_lastest_result(self.name)
        self.display_function(result)

    def run_all(self, **kwargs):
        for v in (self.versions.keys() if isinstance(self.versions, dict) else xrange(len(self.versions))):
            self.run(version = v, **kwargs)

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def test_all(self, **kwargs):
        self.run_all(test_mode=True, **kwargs)


# The following is an alternate interface to experiments.  It may be useful in things like notebooks where it is
# difficult to put all code within a big "with" statement.  See test_start_experiment

_CURRENT_EXPERIMENT_CONTEXT = None


def start_experiment(*args, **kwargs):
    global _CURRENT_EXPERIMENT_CONTEXT
    _CURRENT_EXPERIMENT_CONTEXT = record_experiment(*args, **kwargs)
    return _CURRENT_EXPERIMENT_CONTEXT.__enter__()


def end_current_experiment():
    global _CURRENT_EXPERIMENT_CONTEXT
    _CURRENT_EXPERIMENT_CONTEXT.__exit__(None, None, None)
    _CURRENT_EXPERIMENT_CONTEXT = None


if __name__ == '__main__':
    browse_experiment_records()
