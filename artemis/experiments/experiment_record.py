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
import inspect
import logging
import multiprocessing
import os
import pickle
import re
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pprint import pprint
from artemis.fileman.local_dir import format_filename, make_file_dir, get_local_path, make_dir
from artemis.fileman.persistent_ordered_dict import PersistentOrderedDict
from artemis.general.display import CaptureStdOut
from artemis.general.functional import infer_derived_arg_values, get_partial_chain
from artemis.general.hashing import compute_fixed_hash
from artemis.general.should_be_builtins import separate_common_items, izip_equal
from artemis.general.test_mode import is_test_mode, set_test_mode
try:
    from enum import Enum
except ImportError:
    raise ImportError("Failed to import the enum package. This was added in python 3.4 but backported back to 2.4.  To install, run 'pip install --upgrade pip enum34'")

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

    def __init__(self, display_function=None, comparison_function=None, one_liner_results=None, info=None, is_root=False):
        """
        :param display_function: A function that takes the results (whatever your experiment returns) and displays them.
        :param comparison_function: A function that takes an OrderedDict<experiment_name, experiment_return_value>.
            You can optionally define this function to compare the results of different experiments.
            You can use call this via the UI with the compare_results command.
        :param one_liner_results: A function that takes your results and returns a 1 line string summarizing them.
        :param info: Don't use this?
        :param is_root: True to make this a root experiment - so that it is not listed to be run itself.
        """
        self.display_function = display_function
        self.comparison_function = comparison_function
        self.info = info
        self.is_root = is_root
        self.one_liner_results = one_liner_results

    def __call__(self, f):
        f.is_base_experiment = True
        ex = Experiment(
            name=f.__name__,
            function=f,
            display_function=self.display_function,
            comparison_function = self.comparison_function,
            one_liner_results=self.one_liner_results,
            info=OrderedDict([('Root Experiment', f.__name__), ('Defined in', inspect.getmodule(f).__file__)]),
            is_root=self.is_root
        )
        return ex


GLOBAL_EXPERIMENT_LIBRARY = OrderedDict()


class ExpInfoFields(Enum):
    NAME = 'Name'
    ID = 'Identifier'
    DIR = 'Directory'
    TIMESTAMP = 'Timestamp'
    ARGS = 'Args'
    FUNCTION = 'Function'
    MODULE = 'Module'
    FILE = 'File'
    STATUS = 'Status'
    N_FIGS = '# Figures Generated'
    FIGS = 'Figure Locs'
    RUNTIME = 'Run Time'
    VERSION = 'Version'
    NOTES = 'Notes'


class ExpStatusOptions(Enum):
    STARTED = 'Running (or Killed)'
    ERROR = 'Error'
    STOPPED = 'Stopped by User'
    FINISHED = 'Ran Succesfully'


class ExperimentRecordInfo(object):
    def __init__(self, file_path, write_text_version=True):
        before, ext = os.path.splitext(file_path)
        assert ext == '.pkl', 'Your file-path must be a pickle'
        self._text_path = before + '.txt' if write_text_version else None
        self.persistent_obj = PersistentOrderedDict(file_path=file_path)

    def has_field(self, field):
        assert field in ExpInfoFields, 'Field must be a member of ExperimentRecordInfo.FIELDS'
        return field in self.persistent_obj

    def get_field(self, field):
        """
        :param field: A member of ExperimentRecordInfo.FIELDS
        :return:
        """
        assert field in ExpInfoFields, 'Field must be a member of ExperimentRecordInfo.FIELDS'
        return self.persistent_obj[field]

    def set_field(self, field, value):
        assert field in ExpInfoFields, 'Field must be a member of ExperimentRecordInfo.FIELDS'
        if field == ExpInfoFields.STATUS:
            assert value in ExpStatusOptions, 'Status value must be in: {}'.format(ExpStatusOptions)
        with self.persistent_obj as pod:
            pod[field] = value
        if self._text_path is not None:
            with open(self._text_path, 'w') as f:
                f.write(self.get_text())

    def add_note(self, note):
        if not self.has_field(ExpInfoFields.NOTES):
            self.set_field(ExpInfoFields.NOTES, [note])
        else:
            self.set_field(ExpInfoFields.NOTES, self.get_field(ExpInfoFields.NOTES) + [note])

    def get_text(self):
        if ExpInfoFields.VERSION not in self.persistent_obj:  # Old version... we must adapt
            return '\n'.join(
                '{}: {}'.format(key, self.get_field_text(key)) for key, value in self.persistent_obj.iteritems())
        else:
            return '\n'.join(
                '{}: {}'.format(key.value, self.get_field_text(key)) for key, value in self.persistent_obj.iteritems())

    def get_field_text(self, field, replacement_if_none=''):
        assert field in ExpInfoFields, 'Field must be a member of ExperimentRecordInfo.FIELDS'
        if not self.has_field(field):
            return replacement_if_none
        elif field is ExpInfoFields.STATUS:
            return self.get_field(field).value
        elif field is ExpInfoFields.ARGS:
            return ['{}={}'.format(k, v) for k, v in self.get_field(field)]
        else:
            return str(self.get_field(field))


class NoSavedResultError(Exception):
    def __init__(self, experiment_record_id):
        Exception.__init__(self, "Experiment Record {} has no saved result.".format(experiment_record_id))


class ExperimentRecord(object):

    ERROR_FILE_NAME = 'errortrace.txt'

    def __init__(self, experiment_directory):
        self._experiment_directory = experiment_directory
        self._info = ExperimentRecordInfo(os.path.join(experiment_directory, 'info.pkl'))

    @property
    def info(self):
        return self._info

    def show_figures(self):
        from artemis.plotting.saving_plots import show_saved_figure
        for loc in self.get_figure_locs():
            show_saved_figure(loc)

    def get_log(self):
        log_file_path = os.path.join(self._experiment_directory, 'output.txt')
        assert os.path.exists(log_file_path), 'No output file found.  Maybe "%s" is not an experiment directory?' % (
        self._experiment_directory,)
        with open(log_file_path) as f:
            text = f.read()
        return text

    def list_files(self, full_path=False):
        """
        List files in experiment directory, relative to root.
        :param full_path: If true, list file with the full local path
        :return: A list of strings indicating the file paths.
        """
        paths = [os.path.join(root, filename) for root, _, files in os.walk(self._experiment_directory) for filename in
                 files]
        if not full_path:
            dir_length = len(self._experiment_directory)
            paths = [f[dir_length + 1:] for f in paths]
        return paths

    def open_file(self, filename, *args, **kwargs):
        """
        Open a file within the experiment record folder.
        Example Usage:

            with record.open_file('filename.txt') as f:
                txt = f.read()

        :param filename: Path within experiment directory (it can include subdirectories)
        :param args, kwargs: Forwarded to python's "open" function
        :return: A file object
        """
        full_path = os.path.join(self._experiment_directory, filename)
        make_file_dir(full_path)  # Make the path if it does not yet exist
        return open(full_path, *args, **kwargs)

    def get_figure_locs(self, include_directory=True):
        locs = [f for f in os.listdir(self._experiment_directory) if f.startswith('fig-')]
        if include_directory:
            return [os.path.join(self._experiment_directory, f) for f in locs]
        else:
            return locs

    def show(self):
        print '{header} Showing Experiment {header}\n{info}\n{subborder}Logs {subborder}\n{log}\n{border}'.format(
            header="=" * 20, border="=" * 50, info=self.info.get_text(), subborder='-' * 20, log=self.get_log())
        self.show_figures()

    def get_info_text(self):
        return self.info.get_text()

    def has_result(self):
        return os.path.exists(os.path.join(self._experiment_directory, 'result.pkl'))

    def get_result(self):
        result_loc = os.path.join(self._experiment_directory, 'result.pkl')
        if os.path.exists(result_loc):
            with open(result_loc) as f:
                result = pickle.load(f)
            return result
        else:
            raise NoSavedResultError(self.get_identifier())

    def save_result(self, result):
        file_path = get_local_experiment_path(os.path.join(self._experiment_directory, 'result.pkl'))
        make_file_dir(file_path)
        with open(file_path, 'w') as f:
            pickle.dump(result, f, protocol=2)
            print 'Saving Result for Experiment "%s"' % (self.get_identifier(),)

    def get_identifier(self):
        root, identifier = os.path.split(self._experiment_directory)
        return identifier

    def get_one_liner(self):
        """
        :return: A one line description of the experiment resutls
        """
        if not self.has_result():
            return '<Experiment did not finish>'
        else:
            result = self.get_result()
            if result is None:
                return '<Experiment Returned no result>'
            else:
                try:
                    exp = self.get_experiment()
                except ExperimentNotFoundError:
                    return '<Experiment {} was not found>'.format(record_id_to_experiment_id(self.get_identifier()))
                except Exception as err:
                    return '<Error loading experiment>'
                one_liner = exp.get_one_liner(result)
                if one_liner is None:
                    return '<One-liner function not defined>'
                else:
                    return one_liner

    def get_experiment(self):
        return load_experiment(record_id_to_experiment_id(self.get_identifier()))

    def get_name(self):
        return self.get_identifier()[27:]  # NOTE: THIS WILL HAVE TO CHANGE IF WE USE A DIFFERENT DATA FORMAT!

    def get_dir(self):
        return self._experiment_directory

    def delete(self):
        shutil.rmtree(self._experiment_directory)

    @classmethod
    def from_identifier(cls, record_id):
        path = os.path.join(get_local_path(os.path.join('experiments', record_id)))
        return ExperimentRecord(path)

    def get_invalid_arg_note(self):
        """
        Return a string identifying ig the arguments for this experiment are still valid.
        :return:
        """
        experiment_id = record_id_to_experiment_id(self.get_identifier())
        if is_experiment_loadable(experiment_id):
            last_run_args = dict(self.info.get_field(ExpInfoFields.ARGS))
            current_args = dict(load_experiment(record_id_to_experiment_id(self.get_identifier())).get_args())
            validity = self.is_valid(last_run_args=last_run_args, current_args=current_args)
            if validity is False:
                last_arg_str, this_arg_str = [
                    ['{}:{}'.format(k, v) for k, v in argdict.iteritems()] if isinstance(argdict, dict) else
                    ['{}:{}'.format(k, v) for k, v in argdict]
                    for argdict in (last_run_args, current_args)]
                common, (old_args, new_args) = separate_common_items([last_arg_str, this_arg_str])
                notes = "No: Args changed!: {{{}}}->{{{}}}".format(','.join(old_args), ','.join(new_args))
            elif validity is None:
                notes = "Cannot Determine."
            else:
                notes = "Yes"
        else:
            notes = "<Experiment Not Currently Imported>"
        return notes

    def is_valid(self, last_run_args=None, current_args=None):
        """
        :return: True if the experiment arguments have not changed
            False if they have changed
            None if it cannot be determined because arguments are not hashable objects.
        """
        if last_run_args is None:
            last_run_args = dict(self.info.get_field(ExpInfoFields.ARGS))
        if current_args is None:
            current_args = dict(load_experiment(record_id_to_experiment_id(self.get_identifier())).get_args())
        try:
            return compute_fixed_hash(last_run_args) == compute_fixed_hash(current_args)
        except NotImplementedError:  # Happens when we have unhashable arguments
            return None

    def get_error_trace(self):
        """
        Get the error trace, or return None if there is no error trace.
        :return:
        """
        file_path = os.path.join(self._experiment_directory, self.ERROR_FILE_NAME)
        if os.path.exists(file_path):
            with open(file_path) as f:
                return f.read()
        else:
            return None

    def write_error_trace(self, print_too = True):
        file_path = os.path.join(self._experiment_directory, self.ERROR_FILE_NAME)
        assert not os.path.exists(file_path), 'Error trace has already been created in this experiment... Something fishy going on here.'
        with open(file_path, 'w') as f:
            error_text = traceback.format_exc()
            f.write(error_text)
        if print_too:
            print error_text

_CURRENT_EXPERIMENT_RECORD = None


@contextmanager
def record_experiment(identifier='%T-%N', name='unnamed', print_to_console=True, show_figs=None,
                      save_figs=True, saved_figure_ext='.pdf', use_temp_dir=False, date=None):
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
    # Note: matplotlib imports are internal in order to avoid trouble for people who may import this module without having
    # a working matplotlib (which can occasionally be tricky to install).
    if date is None:
        date = datetime.now()
    identifier = format_filename(file_string=identifier, base_name=name, current_time=date)

    if show_figs is None:
        show_figs = 'draw' if is_test_mode() else 'hang'

    assert show_figs in ('hang', 'draw', False)

    if use_temp_dir:
        experiment_directory = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(experiment_directory))
    else:
        experiment_directory = get_local_path('experiments/{identifier}'.format(identifier=identifier))

    make_dir(experiment_directory)
    from artemis.plotting.manage_plotting import WhatToDoOnShow
    global _CURRENT_EXPERIMENT_RECORD  # Register
    _CURRENT_EXPERIMENT_RECORD = ExperimentRecord(experiment_directory)
    capture_context = CaptureStdOut(log_file_path=os.path.join(experiment_directory, 'output.txt'),
                                    print_to_console=print_to_console)
    show_context = WhatToDoOnShow(show_figs)
    if save_figs:
        from artemis.plotting.saving_plots import SaveFiguresOnShow
        save_figs_context = SaveFiguresOnShow(path=os.path.join(experiment_directory, 'fig-%T-%L' + saved_figure_ext))
        with capture_context, show_context, save_figs_context:
            yield _CURRENT_EXPERIMENT_RECORD
    else:
        with capture_context, show_context:
            yield _CURRENT_EXPERIMENT_RECORD
    _CURRENT_EXPERIMENT_RECORD = None  # Deregister


def get_current_experiment_record():
    if _CURRENT_EXPERIMENT_RECORD is None:
        raise Exception("No experiment is currently running!")
    return _CURRENT_EXPERIMENT_RECORD


def get_current_experiment_id():
    """
    :return: A string identifying the current experiment
    """
    return get_current_experiment_record().get_identifier()


def get_current_experiment_name():
    """
    :return: A string containing the name of the current experiment
    """
    return get_current_experiment_record().get_name()


def get_current_experiment_dir():
    """
    The directory in which the results of the current experiment are recorded.
    """
    return get_current_experiment_record().get_dir()


def open_in_experiment_dir(filename, *args, **kwargs):
    """
    Open a file in the given experiment directory.  Usage:

    with open_in_experiment_dir('myfile.txt', 'w') as f:
        f.write('blahblahblah')

    :param filename: The name of the file, relative to your experiment directory,
    :param args,kwargs: See python built-in "open" function
    :yield: The file object
    """
    return get_current_experiment_record().open_file(filename, *args, **kwargs)


def run_experiment(name, exp_dict=GLOBAL_EXPERIMENT_LIBRARY, **experiment_record_kwargs):
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


def record_id_to_experiment_id(record_id):
    return record_id[27:]


def record_id_to_timestamp(record_id):
    return record_id[:26]


def run_experiment_ignoring_errors(name, **kwargs):
    try:
        run_experiment(name, **kwargs)
    except Exception as err:
        traceback.print_exc()


def _get_matching_template_from_experiment_name(experiment_name, template='%T-%N'):
    # Potentially obselete, though will keep around in case it gets useful one day.
    named_template = template.replace('%N', re.escape(experiment_name))
    expr = named_template.replace('%T', '\d\d\d\d\.\d\d\.\d\d\T\d\d\.\d\d\.\d\d\.\d\d\d\d\d\d')
    expr = '^' + expr + '$'
    return expr


def delete_experiment_with_id(experiment_identifier):
    if experiment_exists(experiment_identifier):
        load_experiment_record(experiment_identifier).delete()


def get_local_experiment_path(identifier):
    return os.path.join(get_local_path('experiments'), identifier)


def experiment_exists(identifier):
    local_path = get_local_experiment_path(identifier)
    return os.path.exists(local_path)


def merge_experiment_dicts(*dicts):
    """
    Merge dictionaries of experiments, checking that names are unique.
    """
    merge_dict = OrderedDict()
    for d in dicts:
        assert not any(k in merge_dict for k in d), "Experiments %s has been defined twice." % (
        [k for k in d.keys() if k in merge_dict],)
        merge_dict.update(d)
    return merge_dict


def filter_experiment_ids(ids, expr=None, names=None):
    if expr is not None:
        ids = [e for e in ids if expr in e]
    if names is not None:
        ids = [eid for eid in ids if record_id_to_experiment_id(eid) in names]
    return ids


def get_all_record_ids(experiment_ids=None, filters=None):
    """
    :param experiment_ids: A list of experiment names
    :param filters: A list or regular expressions for matching experiments.
    :return: A list of experiment identifiers.
    """
    expdir = get_local_path('experiments')
    ids = [e for e in os.listdir(expdir) if os.path.isdir(os.path.join(expdir, e))]
    ids = filter_experiment_ids(ids=ids, names=experiment_ids)
    if filters is not None:
        for expr in filters:
            ids = filter_experiment_ids(ids=ids, expr=expr)
    ids = sorted(ids)
    return ids


def experiment_id_to_record_ids(experiment_identifier, filter_status = None):
    """
    :param experiment_identifier: The name of the experiment
    :param filter_status: An ExpStatusOptions enum, indicating that you only want experiments with this status.
        e.g. ExpStatusOptions.FINISHED
    :return: A list of records for this experiment, temporal order
    """
    matching_records = get_all_record_ids(experiment_ids=[experiment_identifier])
    if filter_status is not None:
        assert filter_status in ExpStatusOptions, 'filter status must be one of the ExpStatusOptions'
        matching_records = [record for record in  matching_records if load_experiment_record(record).info.get_field(ExpInfoFields.STATUS) is filter_status]
    return sorted(matching_records)


def experiment_id_to_latest_record_id(experiment_identifier, filter_status = None):
    """
    Show results of the latest experiment matching the given template.
    :param name: The experiment name
    :param template: The template which turns a name into an experiment identifier
    :return: A string identifying the latest matching experiment, or None, if not found.
    """

    all_records = experiment_id_to_record_ids(experiment_identifier, filter_status=filter_status)
    return all_records[-1] if len(all_records)>0 else None


def experiment_id_to_latest_result(experiment_id):
    return load_latest_experiment_record(experiment_id).get_result()


def load_latest_experiment_record(experiment_name, filter_status=None):
    experiment_record_identifier = experiment_id_to_latest_record_id(experiment_name, filter_status=filter_status)
    return None if experiment_record_identifier is None else load_experiment_record(experiment_record_identifier)


def has_experiment_record(experiment_identifier):
    return os.path.exists(get_local_experiment_path(identifier=experiment_identifier))


def load_experiment_record(experiment_identifier):
    """
    Load an ExperimentRecord based on the identifier
    :param experiment_identifier: A string identifying the experiment
    :return: An ExperimentRecord object
    """
    full_path = get_local_experiment_path(identifier=experiment_identifier)
    return ExperimentRecord(full_path)


def _register_experiment(experiment):
    GLOBAL_EXPERIMENT_LIBRARY[experiment.name] = experiment


def clear_experiment_records(ids=None):
    """
    Delete all experiments with ids in the list, or all experiments if ids is None.
    :param ids: A list of experiment ids, or None to remove all.
    """
    # Credit: http://stackoverflow.com/questions/185936/delete-folder-contents-in-python
    folder = get_local_path('experiments')

    if ids is None:
        ids = os.listdir(folder)

    for exp_id in ids:
        exp_path = os.path.join(folder, exp_id)
        try:
            if os.path.isfile(exp_path):
                os.unlink(exp_path)
            elif os.path.isdir(exp_path):
                shutil.rmtree(exp_path)
        except Exception as e:
            print(e)


def get_experiment_info(name):
    experiment = load_experiment(name)
    return str(experiment)


class ExperimentNotFoundError(Exception):
    def __init__(self, experiment_id):
        Exception.__init__(self,
                           'Experiment "{}" could not be loaded, either because it has not been imported, or its definition was removed.'.format(
                               experiment_id))


def load_experiment(experiment_id):
    try:
        return GLOBAL_EXPERIMENT_LIBRARY[experiment_id]
    except KeyError:
        raise ExperimentNotFoundError(experiment_id)


def is_experiment_loadable(experiment_id):
    return experiment_id in GLOBAL_EXPERIMENT_LIBRARY


def _kwargs_to_experiment_name(kwargs):
    return ','.join('{}={}'.format(argname, kwargs[argname]) for argname in sorted(kwargs.keys()))


keep_record_by_default = None


@contextmanager
def experiment_testing_context():
    """
    Use this context when testing the experiment/experiment_record infrastructure.
    Should only really be used in test_experiment_record.py
    """
    ids = get_all_record_ids()
    global keep_record_by_default
    old_val = keep_record_by_default
    keep_record_by_default = True
    yield
    keep_record_by_default = old_val

    def clean_on_close():
        new_ids = set(get_all_record_ids()).difference(ids)
        clear_experiment_records(list(new_ids))

    atexit.register(
        clean_on_close)  # We register this on exit to avoid race conditions with system commands when we open figures externally


def save_figure_in_experiment(name, fig=None, default_ext='.pdf'):
    '''
    Saves the given figure in the experiment directory. If no figure is passed, plt.gcf() is saved instead.
    :param name: The name of the figure to be saved
    :param fig: The figure to be saved, can be None
    :param default_ext: See artemis.plotting.saving_plots.save_figure() for information
    :return: The path to the figure
    '''
    import matplotlib.pyplot as plt
    from artemis.plotting.saving_plots import save_figure
    if fig is None:
        fig = plt.gcf()
    save_path = os.path.join(get_current_experiment_dir(), name)
    save_figure(fig, path=save_path, default_ext=default_ext)
    return save_path


class Experiment(object):
    """
    An experiment.  In general you should not use this class directly.  Use the experiment_function decorator, and
    create variants using decorated_function.add_variant()
    """

    def __init__(self, function=None, display_function=pprint, comparison_function=None, one_liner_results=None, info=None, conclusion=None,
                 name=None, is_root=False):
        """
        :param function: The function defining the experiment
        :param display_function: A function that can be called to display the results returned by function.
            This can be useful if you just want to re-run the display for already-computed and saved results.
            To do this, go experiment.save_last()
        :param conclusion: <Deprecated> will be removed in future
        :param name: Nmae of this experiment.
        """
        self.name = name
        self.function = function
        self.display_function = display_function
        self.one_liner_results = one_liner_results
        self.comparison_function = comparison_function
        self.variants = OrderedDict()
        if info is None:
            info = OrderedDict()
        else:
            assert isinstance(info, dict)
        if conclusion is not None:
            info['Conclusion'] = conclusion
        self.info = info
        self._notes = []
        self.is_root = is_root
        if not is_root:
            _register_experiment(self)

    def __call__(self, *args, **kwargs):
        """ Run the function as normal, without recording or anything.  You can also modify with arguments. """
        return self.function(*args, **kwargs)

    def __str__(self):
        return 'Experiment: %s\n  Description: %s' % \
               (self.name, self.info)

    def get_args(self):
        """
        :param to_root: If True, find all args of this experiment down to the root experiment.
            If False, just return the args that differentiate this variant from its parent.
        :return: A dictionary of arguments to the experiment
        """
        return infer_derived_arg_values(self.function)

    def get_root_function(self):
        return get_partial_chain(self.function)[0]

    def run(self, print_to_console=True, show_figs=None, test_mode=None, keep_record=None, raise_exceptions=True,
            display_results=True, **experiment_record_kwargs):
        """
        Run the experiment, and return the ExperimentRecord that is generated.

        :param print_to_console: Print to console (as well as logging to file)
        :param show_figs: Show figures (as well as saving to file)
        :param test_mode: Run in "test_mode".  This sets the global "test_mode" flag when running the experiment.  This
            flag can be used to, for example, shorten a training session to verify that the code runs.  Can be:
                True: Run in test mode
                False: Don't run in test mode:
                None: Keep the current state of the global "is_test_mode()" flag.
        :param keep_record: Keep the folder that results are saved into.
                True: Results are saved into a folder
                False: Results folder is deleted at the end.
                None: If "test_mode" is true, then delete results at end, otherwise save them.
        :param raise_exceptions: True to raise any exception that occurs when running the experiment.  False to catch it,
            print the error, and move on.
        :param experiment_record_kwargs: Passed to the "record_experiment" context.
        :return: The ExperimentRecord object, if keep_record is true, otherwise None
        """
        if test_mode is None:
            test_mode = is_test_mode()
        if keep_record is None:
            keep_record = keep_record_by_default if keep_record_by_default is not None else not test_mode

        old_test_mode = is_test_mode()
        set_test_mode(test_mode)
        ARTEMIS_LOGGER.info('{border} {mode} Experiment: {name} {border}'.format(border='=' * 10,
                                                                                 mode="Testing" if test_mode else "Running",
                                                                                 name=self.name))
        EIF = ExpInfoFields
        date = datetime.now()
        with record_experiment(name=self.name, print_to_console=print_to_console, show_figs=show_figs,
                use_temp_dir=not keep_record, date=date, **experiment_record_kwargs) as exp_rec:
            start_time = time.time()
            try:
                exp_rec.info.set_field(ExpInfoFields.NAME, self.name)
                exp_rec.info.set_field(ExpInfoFields.ID, exp_rec.get_identifier())
                exp_rec.info.set_field(ExpInfoFields.DIR, exp_rec.get_dir())
                exp_rec.info.set_field(EIF.ARGS, self.get_args().items())
                root_function = self.get_root_function()
                exp_rec.info.set_field(EIF.FUNCTION, root_function.__name__)
                exp_rec.info.set_field(EIF.TIMESTAMP, str(date))
                exp_rec.info.set_field(EIF.MODULE, inspect.getmodule(root_function).__name__)
                exp_rec.info.set_field(EIF.FILE, inspect.getmodule(root_function).__file__)
                exp_rec.info.set_field(EIF.STATUS, ExpStatusOptions.STARTED)
                results = self.function()
                exp_rec.info.set_field(EIF.STATUS, ExpStatusOptions.FINISHED)
            except KeyboardInterrupt:
                exp_rec.info.set_field(EIF.STATUS, ExpStatusOptions.STOPPED)
                exp_rec.write_error_trace(print_too=False)
                raise
            except Exception:
                exp_rec.info.set_field(EIF.STATUS, ExpStatusOptions.ERROR)
                exp_rec.write_error_trace(print_too=not raise_exceptions)
                if raise_exceptions:
                    raise
                else:
                    return exp_rec
            finally:
                exp_rec.info.set_field(EIF.RUNTIME, time.time() - start_time)
                fig_locs = exp_rec.get_figure_locs(include_directory=False)
                exp_rec.info.set_field(EIF.N_FIGS, len(fig_locs))
                exp_rec.info.set_field(EIF.FIGS, fig_locs)

        exp_rec.save_result(results)
        for n in self._notes:
            exp_rec.info.add_note(n)
        if display_results and self.display_function is not None:
            self.display_function(results)
        ARTEMIS_LOGGER.info('{border} Done {mode} Experiment: {name} {border}'.format(border='=' * 10, mode="Testing" if test_mode else "Running", name=self.name))
        set_test_mode(old_test_mode)
        return exp_rec

    def _create_experiment_variant(self, args, kwargs, is_root):
        assert len(args) in (0, 1), "When creating an experiment variant, you can either provide one unnamed argument (the experiment name), or zero, in which case the experiment is named after the named argumeents.  See add_variant docstring"
        name = args[0] if len(args) == 1 else _kwargs_to_experiment_name(kwargs)
        assert name not in self.variants, 'Variant "%s" already exists.' % (name,)
        ex = Experiment(
            name=self.name + '.' + name,
            function=partial(self.function, **kwargs),
            display_function=self.display_function,
            comparison_function=self.comparison_function,
            one_liner_results=self.one_liner_results,
            is_root=is_root
        )
        self.variants[name] = ex
        return ex

    def add_variant(self, *args, **kwargs):
        """
        Add a variant to this experiment, and register it on the list of experiments.
        There are two ways you can do this:

            # Name the experiment explicitely, then list the named arguments
            my_experiment_function.add_variant('big_a', a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.big_a'

            # Allow the experiment to be named automatically, and just list the named arguments
            my_experiment_function.add_variant(a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.a==10000'

        :return: The experiment.
        """
        return self._create_experiment_variant(args, kwargs, is_root=False)

    def add_root_variant(self, *args, **kwargs):
        """
        Add a variant to this experiment, but do NOT register it on the list of experiments.
        There are two ways you can do this:

            # Name the experiment explicitely, then list the named arguments
            my_experiment_function.add_root_variant('big_a', a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.big_a'

            # Allow the experiment to be named automatically, and just list the named arguments
            my_experiment_function.add_root_variant(a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.a==10000'

        :return: The experiment.
        """
        return self._create_experiment_variant(args, kwargs, is_root=True)

    def add_note(self, note):
        """
        :param note:
        :return:
        """
        self._notes.append(str(note))
        return self

    def get_variant(self, *args, **kwargs):
        """
        Get a variant on this experiment.
        :param name: A the name of the variant
        :param path: Optionally, a list of names of subvariants (to call up a nested experiment)
        :return:
        """
        if len(args)==0:
            name = _kwargs_to_experiment_name(kwargs)
        else:
            assert len(args)==1, 'You can only provide 1 unnamed argument to get_variant: the variant name.'
            name, = args
            assert len(kwargs)==0, 'If you provide a variant name ({}), there is no need to specify the keyword arguments. ({})'.format(name, kwargs)
        assert name in self.variants, "No variant '{}' exists.  Existing variants: {}".format(name, self.variants.keys())
        return self.variants[name]

    def get_unnamed_variant(self, **kwargs):
        return self.get_variant(_kwargs_to_experiment_name(kwargs))

    def display_last(self, result='___FINDLATEST', err_if_none=True):
        if result == '___FINDLATEST':
            result = experiment_id_to_latest_result(self.name)
        if err_if_none:
            assert result is not None, "No result was computed for the last run of '%s'" % (self.name,)
        if result is None:
            if err_if_none:
                raise Exception("No result was computed for the last run of '%s'" % (self.name,))
            else:
                print "<No result saved from last run>"
        elif self.display_function is None:
            print result
        else:
            self.display_function(result)

    def get_one_liner(self, results):
        return self.one_liner_results(results) if self.one_liner_results is not None else str(results).replace('\n', ';')

    def display_or_run(self):
        """
        Display the last results, or, if the experiment has not been run yet, run it and then display the results.
        A word of caution: This function does NOT check that the parameters of the last experiment are the same as the
        current parameters.

        :return:
        """
        if experiment_id_to_latest_record_id(self.name) is None:
            self.run()
        else:
            result = experiment_id_to_latest_result(self.name)
            if result is not None and self.display_function is not None:
                self.display_last()
            else:
                self.run()

    def get_all_variants(self, include_roots=False, include_self=False):
        """
        Return a list of variants of this experiment
        :param include_roots:
        :return:
        """
        variants = []
        if include_self and (not self.is_root or include_roots):
            variants.append(self)
        for name, v in self.variants.iteritems():
            variants += v.get_all_variants(include_roots=include_roots, include_self=True)
        return variants

    def run_all(self):
        """
        Run this experiment (if not a root-experiment) and all variants (if not roots).
        """
        experiments = self.get_all_variants()
        for ex in experiments:
            ex.run()

    def run_all_multiprocess(self):
        experiments = self.get_all_variants()
        p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        p.map(run_experiment, [ex.name for ex in experiments])

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def test_all(self, **kwargs):
        self.run_all(test_mode=True, **kwargs)

    def compare_results(self, experiment_ids, error_if_no_result = True):
        if self.comparison_function is None:
            print 'Cannot compare results, because you have not specified any comparison function for this experiment.  Use @ExperimentFunction(comparison_function = my_func)'
            return
        results = load_lastest_experiment_results(experiment_ids, error_if_no_result=error_if_no_result)
        self.comparison_function(results)

    def get_all_ids(self):
        """
        Get all identifiers of this experiment that have been run.
        :return:
        """
        return get_all_record_ids(experiment_ids=[self.name])

    def clear_records(self):
        """
        Delete all records from this experiment
        """
        clear_experiment_records(ids=self.get_all_ids())

    def get_name(self):
        return self.name


def load_lastest_experiment_results(experiment_ids, error_if_no_result = True):
    results = OrderedDict()
    for eid in experiment_ids:
        record = load_latest_experiment_record(eid, filter_status=ExpStatusOptions.FINISHED)
        if record is None:
            if error_if_no_result:
                raise Exception("Experiment {} had no result.  Run this experiment to completion before trying to compare its results.".format(eid))
            else:
                ARTEMIS_LOGGER.warn('Experiment {} had no records.  Not including this in results'.format(eid))
        else:
            results[eid] = record.get_result()
    if len(results)==0:
        ARTEMIS_LOGGER.warn('None of your experiments had any results.  Your comparison function will probably show no meaningful result.')
    return results


def make_record_comparison_table(record_ids, args_to_show=None, results_extractor = None, print_table = False):
    """
    Make a table comparing the arguments and results of different experiment records.  You can use the output
    of this function with the tabulate package to make a nice readable table.

    :param record_ids: A list of record ids whose results to compare
    :param args_to_show: A list of arguments to show.  If none, it will just show all arguments
        that differ between experiments.
    :param results_extractor: A dict<str->callable> where the callables take the result of the
        experiment as an argument and return an entry in the table.  For example:
    :param print_table: Optionally, import tabulate and print the table here and now.
    :return: headers, rows
        headers is a list of of headers for the top of the table
        rows is a list of lists filling in the information.

    example usage:

        headers, rows = make_record_comparison_table(
            record_ids = [experiment_id_to_latest_record_id(eid) for eid in [
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.version=mlp',
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.full-gd.n_steps=1',
                'demo_fast_weight_mlp.multilayer_baseline.1epoch.full-gd.n_steps=20',
                ]],
            results_extractor={
                'Test': lambda result: result.get_best('test').score.get_score('test'),
                'Train': lambda result: result.get_best('test').score.get_score('train'),
                }
             )
        import tabulate
        print tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt)
    """

    records = [load_experiment_record(rid) for rid in record_ids]
    args = [rec.info.get_field(ExpInfoFields.ARGS) for rec in records]
    if args_to_show is None:
        common, separate = separate_common_items(args)
        args_to_show = [k for k, v in separate[0]]

    if results_extractor is None:
        results_extractor = {'Result': str}
    else:
        assert isinstance(results_extractor, dict)

    headers = args_to_show + results_extractor.keys()

    rows = []
    for record, record_args in izip_equal(records, args):
        arg_dict = dict(record_args)
        args_vals = [arg_dict[k] for k in args_to_show]
        results = record.get_result()
        rows.append(args_vals+[f(results) for f in results_extractor.values()])

    if print_table:
        import tabulate
        print tabulate.tabulate(rows, headers=headers, tablefmt='simple')


    return headers, rows


def clear_all_experiments():
    GLOBAL_EXPERIMENT_LIBRARY.clear()


@contextmanager
def capture_created_experiments():
    """
    A convenient way to cross-breed experiments.  If you define experiments in this block, you can capture them for
    later use (for instance by modifying them)
    :return:
    """
    current_len = len(GLOBAL_EXPERIMENT_LIBRARY)
    new_experiments = []
    yield new_experiments
    for ex in GLOBAL_EXPERIMENT_LIBRARY.values()[current_len:]:
        new_experiments.append(ex)
