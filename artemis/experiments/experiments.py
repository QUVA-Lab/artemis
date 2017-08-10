import atexit
import inspect
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from getpass import getuser
from uuid import getnode
from artemis.experiments.experiment_record import ARTEMIS_LOGGER, \
    ExpInfoFields, record_experiment, ExpStatusOptions, experiment_id_to_record_ids, load_experiment_record, \
    get_all_record_ids, clear_experiment_records
from artemis.general.functional import infer_derived_arg_values, get_partial_chain
from artemis.general.test_mode import is_test_mode, set_test_mode


class Experiment(object):
    """
    An experiment.  In general you should not use this class directly.  Use the experiment_function decorator, and
    create variants using decorated_function.add_variant()
    """

    def __init__(self, function=None, display_function=None, comparison_function=None, one_liner_function=None,
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
        self._display_function = display_function
        self._one_liner_results = one_liner_function
        self._comparison_function = comparison_function
        self.variants = OrderedDict()
        self._notes = []
        self.is_root = is_root
        if not is_root:
            _register_experiment(self)

    @property
    def display_function(self):
        return self._display_function

    @property
    def one_liner_function(self):
        return self._one_liner_results

    @property
    def comparison_function(self):
        return self._comparison_function

    def __call__(self, *args, **kwargs):
        """ Run the function as normal, without recording or anything.  You can also modify with arguments. """
        return self.function(*args, **kwargs)

    def __str__(self):
        return 'Experiment {}'.format(self.name)

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
                exp_rec.info.set_field(ExpInfoFields.ID, exp_rec.get_id())
                exp_rec.info.set_field(ExpInfoFields.DIR, exp_rec.get_dir())
                exp_rec.info.set_field(EIF.ARGS, self.get_args().items())
                root_function = self.get_root_function()
                exp_rec.info.set_field(EIF.FUNCTION, root_function.__name__)
                exp_rec.info.set_field(EIF.TIMESTAMP, str(date))
                module = inspect.getmodule(root_function)
                exp_rec.info.set_field(EIF.MODULE, module.__name__)
                exp_rec.info.set_field(EIF.FILE, module.__file__ if hasattr(module, '__file__') else '<unknown>')
                exp_rec.info.set_field(EIF.STATUS, ExpStatusOptions.STARTED)
                exp_rec.info.set_field(EIF.USER, getuser())
                exp_rec.info.set_field(EIF.MAC, ':'.join(("%012X" % getnode())[i:i+2] for i in range(0, 12, 2)))
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
            one_liner_function=self.one_liner_function,
            is_root=is_root
        )
        self.variants[name] = ex
        return ex

    def add_variant(self, variant_name = None, **kwargs):
        """
        Add a variant to this experiment, and register it on the list of experiments.
        There are two ways you can do this:

        .. code-block:: python

            # Name the experiment explicitely, then list the named arguments
            my_experiment_function.add_variant('big_a', a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.big_a'

            # Allow the experiment to be named automatically, and just list the named arguments
            my_experiment_function.add_variant(a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.a=10000'

        :param variant_name: Optionally, the name of the experiment
        :param kwargs: The named arguments which will differ from the base experiment.
        :return: The experiment.
        """
        return self._create_experiment_variant(() if variant_name is None else (variant_name, ), kwargs, is_root=False)

    def add_root_variant(self, variant_name=None, **kwargs):
        """
        Add a variant to this experiment, but do NOT register it on the list of experiments.
        There are two ways you can do this:

        .. code-block:: python

            # Name the experiment explicitely, then list the named arguments
            my_experiment_function.add_root_variant('big_a', a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.big_a'

            # Allow the experiment to be named automatically, and just list the named arguments
            my_experiment_function.add_root_variant(a=10000)
            assert my_experiment_function.get_name()=='my_experiment_function.a=10000'

        :param variant_name: Optionally, the name of the experiment
        :param kwargs: The named arguments which will differ from the base experiment.
        :return: The experiment.
        """
        return self._create_experiment_variant(() if variant_name is None else (variant_name, ), kwargs, is_root=True)

    def get_id(self):
        return self.name

    def get_variant(self, variant_name=None, **kwargs):
        """
        Get a variant on this experiment.

        :param variant_name: The name of the variant, if it has one
        :param kwargs: Otherwise, the named arguments which were used to define the variant.
        :return: An Experiment object
        """
        if variant_name is None:
            variant_name = _kwargs_to_experiment_name(kwargs)
        else:
            assert len(kwargs)==0, 'If you provide a variant name ({}), there is no need to specify the keyword arguments. ({})'.format(variant_name, kwargs)
        assert variant_name in self.variants, "No variant '{}' exists.  Existing variants: {}".format(variant_name, self.variants.keys())
        return self.variants[variant_name]

    def get_records(self, only_completed=False):
        records = [load_experiment_record(rid) for rid in experiment_id_to_record_ids(self.name)]
        if only_completed:
            records = [record for record in records if record.get_status()==ExpStatusOptions.FINISHED]
        return records

    def browse(self, **kwargs):
        from artemis.experiments.ui import browse_experiments
        browse_experiments(root_experiment=self, **kwargs)

    # Above this line is the core api....
    # -----------------------------------
    # Below this line are a bunch of convenience functions.

    def has_record(self, completed=True, valid=True):
        """
        Return true if the experiment has a record, otherwise false.
        :param completed: Check that the record is completed.
        :param valid: Check that the record is valid (arguments match current experiment arguments)
        :return: True/False
        """
        records = self.get_records(only_completed=completed)
        if valid:
            records = [record for record in records if record.args_valid()]
        return len(records)>0

    def get_all_variants(self, include_roots=False, include_self=True):
        """
        Return a list of variants of this experiment
        :param include_roots: Include "root" experiments
        :param include_self: Include this experiment (unless include_roots is false and this this experiment is a root)
        :return: A list of experiments.
        """
        variants = []
        if include_self and (not self.is_root or include_roots):
            variants.append(self)
        for name, v in self.variants.iteritems():
            variants += v.get_all_variants(include_roots=include_roots, include_self=True)
        return variants

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def get_latest_record(self, only_completed=False, err_if_none = True):
        """
        Return the ExperimentRecord from the latest run of this Experiment.

        :param only_completed: Only search among records of that have run to completion.
        :param err_if_none: If True, raise an error if no record exists.  Otherwise, just return None in this case.
        :return: An ExperimentRecord object
        """
        records = self.get_records(only_completed=only_completed)
        if len(records)==0:
            if err_if_none:
                raise Exception('No{} records for experiment "{}"'.format(' completed' if only_completed else '', self.name))
            else:
                return None
        else:
            return sorted(records, key=lambda x: x.get_id())[-1]

    def get_variant_records(self, only_completed=False, only_last=False, flat=False):
        """
        Get the collection of records associated with all variants of this Experiment.

        :param only_completed: Only search among records of that have run to completion.
        :param only_last: Just return the most recent record.
        :param flat: Just return a list of records
        :return: if not flat (default) An OrderedDict<experiment_id: ExperimentRecord>.
            otherwise, if flat: a list<ExperimentRecord>
        """
        variants = self.get_all_variants(include_self=True)

        if only_last:
            exp_record_dict = OrderedDict((ex.name, ex.get_latest_record(only_completed=only_completed, err_if_none=False)) for ex in variants)
            if flat:
                return [record for record in exp_record_dict.values() if record is not None]
            else:
                return exp_record_dict
        else:
            exp_record_dict = OrderedDict((ex.name, ex.get_records(only_completed=only_completed)) for ex in variants)
            if flat:
                return [record for records in exp_record_dict.values() for record in records]
            else:
                return exp_record_dict


GLOBAL_EXPERIMENT_LIBRARY = OrderedDict()


class ExperimentNotFoundError(Exception):
    def __init__(self, experiment_id):
        Exception.__init__(self,'Experiment "{}" could not be loaded, either because it has not been imported, or its definition was removed.'.format(experiment_id))


def clear_all_experiments():
    GLOBAL_EXPERIMENT_LIBRARY.clear()


@contextmanager
def capture_created_experiments():
    """
    A convenient way to cross-breed experiments.  If you define experiments in this block, you can capture them for
    later use (for instance by modifying them). e.g.:

    .. code-block:: python

        @experiment_function
        def add_two_numbers(a=1, b=2):
            return a+b

        with capture_created_experiments() as exps:
            add_two_numbers.add_variant(a=2)
            add_two_numbers.add_variant(a=3)

        for ex in exps:
            ex.add_variant(b=4)

    :rtype: Generator[:class:`Experiment`]
    """
    current_len = len(GLOBAL_EXPERIMENT_LIBRARY)
    new_experiments = []
    yield new_experiments
    for ex in GLOBAL_EXPERIMENT_LIBRARY.values()[current_len:]:
        new_experiments.append(ex)


def _register_experiment(experiment):
    assert experiment.name not in GLOBAL_EXPERIMENT_LIBRARY, 'You have already registered an experiment named {} in {}'.format(experiment.name, inspect.getmodule(experiment.get_root_function()).__name__)
    GLOBAL_EXPERIMENT_LIBRARY[experiment.name] = experiment


def get_experiment_info(name):
    experiment = load_experiment(name)
    return str(experiment)


def load_experiment(experiment_id):
    try:
        return GLOBAL_EXPERIMENT_LIBRARY[experiment_id]
    except KeyError:
        raise ExperimentNotFoundError(experiment_id)


def is_experiment_loadable(experiment_id):
    assert isinstance(experiment_id, basestring), 'Expected a string for experiment_id, not {}'.format(experiment_id)
    return experiment_id in GLOBAL_EXPERIMENT_LIBRARY


def _kwargs_to_experiment_name(kwargs):
    return ','.join('{}={}'.format(argname, kwargs[argname]) for argname in sorted(kwargs.keys()))


@contextmanager
def hold_global_experiment_libary(new_lib = None):
    if new_lib is None:
        new_lib = {}

    global GLOBAL_EXPERIMENT_LIBRARY
    oldlib = GLOBAL_EXPERIMENT_LIBRARY
    GLOBAL_EXPERIMENT_LIBRARY = new_lib
    yield GLOBAL_EXPERIMENT_LIBRARY
    GLOBAL_EXPERIMENT_LIBRARY = oldlib


keep_record_by_default = None


@contextmanager
def experiment_testing_context(close_figures_at_end = True, new_experiment_lib = False):
    """
    Use this context when testing the experiment/experiment_record infrastructure.
    Should only really be used in test_experiment_record.py
    """
    ids = get_all_record_ids()
    global keep_record_by_default
    old_val = keep_record_by_default
    keep_record_by_default = True
    if new_experiment_lib:
        with hold_global_experiment_libary():
            yield
    else:
        yield
    keep_record_by_default = old_val

    if close_figures_at_end:
        from matplotlib import pyplot as plt
        plt.close('all')

    def clean_on_close():
        new_ids = set(get_all_record_ids()).difference(ids)
        clear_experiment_records(list(new_ids))

    atexit.register(
        clean_on_close)  # We register this on exit to avoid race conditions with system commands when we open figures externally