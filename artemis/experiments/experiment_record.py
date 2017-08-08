import atexit
import logging
import os
import pickle
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict
from contextlib import contextmanager, nested
from datetime import datetime

from artemis.fileman.local_dir import format_filename, make_file_dir, get_local_path, make_dir
from artemis.fileman.persistent_ordered_dict import PersistentOrderedDict
from artemis.general.display import CaptureStdOut
from artemis.general.hashing import compute_fixed_hash
from artemis.general.test_mode import is_test_mode

try:
    from enum import Enum
except ImportError:
    raise ImportError("Failed to import the enum package. This was added in python 3.4 but backported back to 2.4.  To install, run 'pip install --upgrade pip enum34'")

logging.basicConfig()
ARTEMIS_LOGGER = logging.getLogger('artemis')
ARTEMIS_LOGGER.setLevel(logging.INFO)

__author__ = 'peter'


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
    USER = 'User'
    MAC = 'MAC Address'


class ExpStatusOptions(Enum):
    STARTED = 'Running (or Killed)'
    ERROR = 'Error'
    STOPPED = 'Stopped by User'
    FINISHED = 'Ran Succesfully'
    CORRUPT = 'Corrupt'


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

    def add_note(self, note):  # Currently unused
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
    """
    A Record of a run of an Experiment.  This object allows you to access data stored in the directory that was created
    for that run of the experiment.  Experiment Records are stored in ``~/artemis/experiments/``.
    """

    ERROR_FILE_NAME = 'errortrace.txt'

    def __init__(self, experiment_directory):
        self._experiment_directory = experiment_directory
        self._info = None

    @property
    def info(self):
        """
        :return: An ExperimentRecordInfo object, containing info about the experiment (name, runtime, etc)
        """
        if self._info is None:
            self._info = ExperimentRecordInfo(os.path.join(self._experiment_directory, 'info.pkl'))
        return self._info

    def __reduce__(self):
        return self.__class__, (self._experiment_directory, )

    def show_figures(self, hang=False):
        """
        Show all figures that were saved during the run of the experiment.
        :param hang: If True, and figures were saved matplotlib figures, hang execution until they are closed.
        """
        from artemis.plotting.saving_plots import show_saved_figure
        for i, loc in enumerate(self.get_figure_locs()):
            show_saved_figure(loc, title='{} Fig {}'.format(self.get_id(), i + 1))
        if hang and len(self.get_figure_locs())>0:
            from matplotlib import pyplot as plt
            plt.show()

    def get_log(self):
        """
        :return: The stdout generated during the run of this experiment.
        """
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

    def get_info_text(self):
        return self.info.get_text()

    def has_result(self):
        return os.path.exists(os.path.join(self._experiment_directory, 'result.pkl'))

    def get_result(self, err_if_none = True):
        result_loc = os.path.join(self._experiment_directory, 'result.pkl')
        if os.path.exists(result_loc):
            with open(result_loc) as f:
                result = pickle.load(f)
            return result
        elif err_if_none:
            raise NoSavedResultError(self.get_id())
        else:
            return None

    def save_result(self, result):
        file_path = get_local_experiment_path(os.path.join(self._experiment_directory, 'result.pkl'))
        make_file_dir(file_path)
        with open(file_path, 'w') as f:
            pickle.dump(result, f, protocol=2)
            print 'Saving Result for Experiment "%s"' % (self.get_id(),)

    def get_id(self):
        """
        Get the id of this experiment record.  Generally in format '<datetime>-<experiment_name>'
        :return:
        """
        root, identifier = os.path.split(self._experiment_directory)
        return identifier

    def get_experiment(self):
        """
        Load the experiment associated with this record.
        Note that this will raise an ExperimentNotFoundError if the experiment has not been imported.
        :return: An Experiment object
        """
        from artemis.experiments.experiments import load_experiment
        return load_experiment(self.get_experiment_id())

    def get_experiment_id(self):
        return self.get_id()[27:]

    def get_dir(self):
        return self._experiment_directory

    def get_args(self):
        """
        Get the arguments with which this record was run.
        :return: A list of 2-tuples of (arg_name, arg_value)
        """
        return self.info.get_field(ExpInfoFields.ARGS)

    def get_status(self):
        try:
            return self.info.get_field(ExpInfoFields.STATUS)
        except KeyError:
            return ExpStatusOptions.CORRUPT

    def load_figures(self):
        """
        :return: A list of matplotlib figures generated in the experiment.  The figures will not be drawn yet, so you
            will have to call plt.show() to draw them or plt.draw() to draw them.
        """
        locs = self.get_figure_locs()
        figs = []
        for fig_path in locs:
            assert fig_path.endswith('.pkl'), 'Figure {} was not saved as a pickle, so it cannot be reloaded.'.format(fig_path)
            with open(fig_path) as f:
                figs.append(pickle.load(f))
        return figs

    def delete(self):
        """
        Delete this experiment record from disk.
        """
        shutil.rmtree(self._experiment_directory)

    def args_valid(self, last_run_args=None, current_args=None):
        """
        :return: True if the experiment arguments have not changed
            False if they have changed
            None if it cannot be determined because arguments are not hashable objects.
        """
        if last_run_args is None:
            last_run_args = dict(self.info.get_field(ExpInfoFields.ARGS))
        if current_args is None:
            current_args = dict(self.get_experiment().get_args())
        try:
            return compute_fixed_hash(last_run_args, try_objects=True) == compute_fixed_hash(current_args, try_objects=True)
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
def hold_current_experiment_record(experiment_record):
    global _CURRENT_EXPERIMENT_RECORD
    assert _CURRENT_EXPERIMENT_RECORD is None, "It seems that you are trying to start an experiment withinin an experiment.  This is not allowed!"
    _CURRENT_EXPERIMENT_RECORD = experiment_record
    try:
        yield
    finally:
        _CURRENT_EXPERIMENT_RECORD = None


def is_matplotlib_imported():
    return 'matplotlib' in sys.modules


@contextmanager
def record_experiment(identifier='%T-%N', name='unnamed', print_to_console=True, show_figs=None,
                      save_figs=True, saved_figure_ext='.fig.pkl', use_temp_dir=False, date=None):
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
        experiment_directory = get_local_experiment_path(identifier)

    make_dir(experiment_directory)
    this_record = ExperimentRecord(experiment_directory)

    # Create context that sets the current experiment record
    # and the context which captures stdout (print statements) and logs them.
    contexts = [
        hold_current_experiment_record(this_record),
        CaptureStdOut(log_file_path=os.path.join(experiment_directory, 'output.txt'), print_to_console=print_to_console)
        ]

    if is_matplotlib_imported():
        from artemis.plotting.manage_plotting import WhatToDoOnShow
        # Add context that modifies how matplotlib figures are shown.
        contexts.append(WhatToDoOnShow(show_figs))
        if save_figs:
            from artemis.plotting.saving_plots import SaveFiguresOnShow
            # Add context that saves figures when show is called.
            contexts.append(SaveFiguresOnShow(path=os.path.join(experiment_directory, 'fig-%T-%L' + saved_figure_ext)))

    with nested(*contexts):
        yield this_record


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


def record_id_to_experiment_id(record_id):
    return load_experiment_record(record_id).get_experiment_id()


def delete_experiment_with_id(experiment_identifier):
    if experiment_exists(experiment_identifier):
        load_experiment_record(experiment_identifier).delete()


def get_experiment_dir():
    path = get_local_path('experiments')
    if not os.path.exists(path):
        make_dir(path)
    return path


def get_local_experiment_path(identifier):
    return os.path.join(get_experiment_dir(), identifier)


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


def filter_experiment_ids(record_ids, expr=None, experiment_ids=None):
    if expr is not None:
        record_ids = [e for e in record_ids if expr in e]
    if experiment_ids is not None:
        record_ids = [record_id for record_id in record_ids if ExperimentRecord(record_id).get_experiment_id() in experiment_ids]
    return record_ids


def get_all_record_ids(experiment_ids=None, filters=None):
    """
    :param experiment_ids: A list of experiment names
    :param filters: A list or regular expressions for matching experiments.
    :return: A list of experiment identifiers.
    """
    expdir = get_experiment_dir()
    ids = [e for e in os.listdir(expdir) if os.path.isdir(os.path.join(expdir, e))]
    ids = filter_experiment_ids(record_ids=ids, experiment_ids=experiment_ids)
    if filters is not None:
        for expr in filters:
            ids = filter_experiment_ids(record_ids=ids, expr=expr)
    ids = sorted(ids)
    return ids


def experiment_id_to_record_ids(experiment_identifier):
    """
    :param experiment_identifier: The name of the experiment
    :param filter_status: An ExpStatusOptions enum, indicating that you only want experiments with this status.
        e.g. ExpStatusOptions.FINISHED
    :return: A list of records for this experiment, temporal order
    """
    matching_records = get_all_record_ids(experiment_ids=[experiment_identifier])
    return sorted(matching_records)


def has_experiment_record(experiment_identifier):
    return len(experiment_id_to_record_ids(experiment_identifier)) != 0


def load_experiment_record(record_id):
    """
    Load an ExperimentRecord based on the identifier
    :param record_id: A string identifying the experiment record
    :return: An ExperimentRecord object
    """
    path = os.path.join(get_experiment_dir(), record_id)
    return ExperimentRecord(path)


def clear_experiment_records(ids):
    """
    Delete all experiments with ids in the list, or all experiments if ids is None.
    :param ids: A list of experiment ids, or None to remove all.
    """
    folder = get_experiment_dir()
    for exp_id in ids:
        exp_path = os.path.join(folder, exp_id)
        ExperimentRecord(exp_path).delete()


def save_figure_in_record(name, fig=None, default_ext='.pkl'):
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
