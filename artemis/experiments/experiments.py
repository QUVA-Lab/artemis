import atexit
import inspect
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from six import string_types

from artemis.experiments.experiment_record import ExpStatusOptions, experiment_id_to_record_ids, load_experiment_record, \
    get_all_record_ids, clear_experiment_records
from artemis.experiments.experiment_record import run_and_record
from artemis.experiments.experiment_record_view import compare_experiment_records, show_record
from artemis.experiments.hyperparameter_search import parameter_search
from artemis.general.display import sensible_str

from artemis.general.functional import get_partial_root, partial_reparametrization, \
    advanced_getargspec, PartialReparametrization
from artemis.general.should_be_builtins import izip_equal
from artemis.general.test_mode import is_test_mode


class Experiment(object):
    """
    An experiment.  In general you should not use this class directly.  Use the experiment_function decorator, and
    create variants using decorated_function.add_variant()
    """

    def __init__(self, function=None, show=None, compare=None, one_liner_function=None, result_parser = None,
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
        self._show = show_record if show is None else show
        self._one_liner_results = sensible_str if one_liner_function is None else one_liner_function
        self._result_parser = (lambda result: [('Result', self.one_liner_function(result))]) if result_parser is None else result_parser
        self._compare = compare_experiment_records if compare is None else compare
        self.variants = OrderedDict()
        self._notes = []
        self.is_root = is_root
        self._tags= set()

        if not is_root:
            all_args, varargs_name, kargs_name, defaults = advanced_getargspec(function)
            undefined_args = [a for a in all_args if a not in defaults]
            assert len(undefined_args)==0, "{} is not a root-experiment, but arguments {} are undefined.  Either provide a value for these arguments or define this as a root_experiment (see {})."\
                .format(self, undefined_args, 'X.add_root_variant(...)' if isinstance(function, partial) else 'X.add_config_root_variant(...)' if isinstance(function, PartialReparametrization) else '@experiment_root')

        _register_experiment(self)

    @property
    def show(self):
        return self._show

    @property
    def one_liner_function(self):
        return self._one_liner_results

    @property
    def compare(self):
        return self._compare

    @compare.setter
    def compare(self, val):
        self._compare = val

    @property
    def result_parser(self):
        return self._result_parser

    def __call__(self, *args, **kwargs):
        """ Run the function as normal, without recording or anything.  You can also modify with arguments. """
        return self.function(*args, **kwargs)

    def __str__(self):
        return 'Experiment {}'.format(self.name)

    def get_args(self):
        """
        :return OrderedDict[str, Any]: An OrderedDict of arguments to the experiment
        """
        all_arg_names, _, _, defaults = advanced_getargspec(self.function)
        return OrderedDict((name, defaults[name]) for name in all_arg_names)

    def get_root_function(self):
        return get_partial_root(self.function)

    def is_generator(self):
        return inspect.isgeneratorfunction(self.get_root_function())

    def call(self, *args, **kwargs):
        """
        Call the experiment function without running as an experiment.  If the experiment is a function, this is the same
        as just result = my_exp_func().  If it's defined as a generator, it loops and returns the last result.
        :return: The last result
        """
        if self.is_generator():
            result = None
            for x in self(*args, **kwargs):
                result = x
        else:
            result = self(*args, **kwargs)
        return result

    def run(self, print_to_console=True, show_figs=None, test_mode=None, keep_record=None, raise_exceptions=True,
            display_results=False, notes = (), **experiment_record_kwargs):
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

        for exp_rec in self.iterator(print_to_console=print_to_console, show_figs=show_figs, test_mode=test_mode, keep_record=keep_record,
                    raise_exceptions=raise_exceptions, display_results=display_results, notes=notes, **experiment_record_kwargs):
            pass

        return exp_rec

    def iterator(self, print_to_console=True, show_figs=None, test_mode=None, keep_record=None, raise_exceptions=True,
            display_results=False, notes = (), **experiment_record_kwargs):

        if keep_record is None:
            keep_record = keep_record_by_default if keep_record_by_default is not None else not test_mode

        exp_rec = None
        for exp_rec in run_and_record(
                function = self.function,
                experiment_id=self.name,
                print_to_console=print_to_console,
                show_figs=show_figs,
                test_mode=test_mode,
                keep_record=keep_record,
                raise_exceptions=raise_exceptions,
                notes=notes,
                **experiment_record_kwargs
                ):
            yield exp_rec
        assert exp_rec is not None, 'Should nevah happen.'
        if display_results:
            self.show(exp_rec)
        return

    def _create_experiment_variant(self, args, kwargs, is_root):
        # TODO: For non-root variants, assert that all args are defined
        assert len(args) in (0, 1), "When creating an experiment variant, you can either provide one unnamed argument (the experiment name), or zero, in which case the experiment is named after the named argumeents.  See add_variant docstring"
        name = args[0] if len(args) == 1 else _kwargs_to_experiment_name(kwargs)
        assert isinstance(name, str), 'Name should be a string.  Not: {}'.format(name)
        assert name not in self.variants, 'Variant "%s" already exists.' % (name,)
        assert '/' not in name, 'Experiment names cannot have "/" in them: {}'.format(name)
        ex = Experiment(
            name=self.name + '.' + name,
            function=partial(self.function, **kwargs),
            show=self._show,
            compare=self._compare,
            one_liner_function=self.one_liner_function,
            result_parser=self._result_parser,
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
        :return Experiment: The experiment.
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
        :return Experiment: The experiment.
        """
        return self._create_experiment_variant(() if variant_name is None else (variant_name, ), kwargs, is_root=True)

    def copy_variants(self, other_experiment):
        """
        Copy over the variants from another experiment.

        :param other_experiment: An Experiment Object
        """
        base_args = other_experiment.get_args()
        for variant in other_experiment.get_variants():
            if variant is not self:
                variant_args = variant.get_args()
                different_args = {k: v for k, v in variant_args.items() if base_args[k]!=v}
                name_diff = variant.get_id()[len(other_experiment.get_id())+1:]
                v = self.add_variant(name_diff, **different_args)
                v.copy_variants(variant)

    def _add_config(self, name, arg_constructors, is_root):
        assert isinstance(name, str), 'Name should be a string.  Not: {}'.format(name)
        assert name not in self.variants, 'Variant "%s" already exists.' % (name,)
        assert '/' not in name, 'Experiment names cannot have "/" in them: {}'.format(name)
        configured_function = partial_reparametrization(self.function, **arg_constructors)
        ex = Experiment(
            name=self.name + '.' + name,
            function=configured_function,
            show=self._show,
            compare=self._compare,
            one_liner_function=self.one_liner_function,
            is_root=is_root
        )
        self.variants[name] = ex
        return ex

    def add_config_variant(self, name, **arg_constructors):
        """
        Add a variant where you redefine the constructor for arguments to the experiment.  e.g.

            @experiment_function
            def demo_smooth_out_signal(smoother, signal):
                y = [smoother.update(xt) for xt in signal]
                plt.plot(y)
                return y

            demo_average_sequence.add_config_variant('exp_smooth', averager = lambda decay=0.1: ExponentialMovingAverage(decay))

        This creates a variant "exp_smooth" which can be parameterized by a "decay" argument.

        :param name: Name of the variant
        :param kwargs: The constructors for the arguments which you'd like to configure.
        :return: A new experiment.
        """
        return self._add_config(name, arg_constructors=arg_constructors, is_root=False)

    def add_config_root_variant(self, name, **arg_constructors):
        """
        Add a config variant which requires additional parametrization.  (See add_config_variant)
        """
        return self._add_config(name, arg_constructors=arg_constructors, is_root=True)

    def get_id(self):
        """
        :return: A string uniquely identifying this experiment.
        """
        return self.name

    def get_variant(self, variant_name=None, **kwargs):
        """
        Get a variant on this experiment.

        :param str variant_name: The name of the variant, if it has one
        :param kwargs: Otherwise, the named arguments which were used to define the variant.
        :return Experiment: An Experiment object
        """
        if variant_name is None:
            variant_name = _kwargs_to_experiment_name(kwargs)
        else:
            assert len(kwargs)==0, 'If you provide a variant name ({}), there is no need to specify the keyword arguments. ({})'.format(variant_name, kwargs)
        assert variant_name in self.variants, "No variant '{}' exists.  Existing variants: {}".format(variant_name, list(self.variants.keys()))
        return self.variants[variant_name]

    def get_records(self, only_completed=False):
        """
        Get all records associated with this experiment.

        :param only_completed: Only include records that have run to completion.
        :return: A list of ExperimentRecord objects.
        """
        records = [load_experiment_record(rid) for rid in experiment_id_to_record_ids(self.name)]
        if only_completed:
            records = [record for record in records if record.get_status()==ExpStatusOptions.FINISHED]
        return records

    def browse(self, command=None, catch_errors = False, close_after = False, filterexp=None, filterrec = None,
            view_mode ='full', raise_display_errors=False, run_args=None, keep_record=True, truncate_result_to=100,
            cache_result_string = False, remove_prefix = None, display_format='nested', **kwargs):
        """
        Open up the UI, which allows you to run experiments and view their results.

        :param command: Optionally, a string command to pass directly to the UI.  (e.g. "run 1")
        :param catch_errors: Catch errors that arise while running experiments
        :param close_after: Close after issuing one command.
        :param filterexp: Filter the experiments with this selection (see help for how to use)
        :param filterrec: Filter the experiment records with this selection (see help for how to use)
        :param view_mode: How to view experiments {'full', 'results'} ('results' leads to a narrower display).
        :param raise_display_errors: Raise errors that arise when displaying the table (otherwise just indicate that display failed in table)
        :param run_args: A dict of named arguments to pass on to Experiment.run
        :param keep_record: Keep a record of the experiment after running.
        :param truncate_result_to: An integer, indicating the maximum length of the result string to display.
        :param cache_result_string: Cache the result string (useful when it takes a very long time to display the results
            when opening up the menu - often when results are long lists).
        :param remove_prefix: Remove the common prefix on the experiment ids in the display.
        :param display_format: How experements and their records are displayed: 'nested' or 'flat'.  'nested' might be
            better for narrow console outputs.
        """
        from artemis.experiments.ui import ExperimentBrowser
        experiments = get_ordered_descendents_of_root(root_experiment=self)
        browser = ExperimentBrowser(experiments=experiments, catch_errors=catch_errors, close_after=close_after,
            filterexp=filterexp, filterrec=filterrec, view_mode=view_mode, raise_display_errors=raise_display_errors,
            run_args=run_args, keep_record=keep_record, truncate_result_to=truncate_result_to, cache_result_string=cache_result_string,
            remove_prefix=remove_prefix, display_format=display_format, **kwargs)
        browser.launch(command=command)

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

    def get_variants(self):
        return self.variants.values()

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
        for name, v in self.variants.items():
            variants += v.get_all_variants(include_roots=include_roots, include_self=True)
        return variants

    def test(self, **kwargs):
        self.run(test_mode=True, **kwargs)

    def get_latest_record(self, only_completed=False, if_none = 'skip'):
        """
        Return the ExperimentRecord from the latest run of this Experiment.

        :param only_completed: Only search among records of that have run to completion.
        :param err_if_none: If True, raise an error if no record exists.  Otherwise, just return None in this case.
        :return ExperimentRecord: An ExperimentRecord object
        """
        assert if_none in ('skip', 'err', 'run')
        records = self.get_records(only_completed=only_completed)
        if len(records)==0:
            if if_none=='run':
                return self.run()
            elif if_none=='err':
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
            exp_record_dict = OrderedDict((ex.name, ex.get_latest_record(only_completed=only_completed, if_none='skip')) for ex in variants)
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

    def add_parameter_search(self, name='parameter_search', space = None, n_calls=100, search_params = None, scalar_func=None):
        """
        :param name: Name of the Experiment to be created
        :param dict[str, skopt.space.Dimension] space: A dict mapping param name to Dimension.
            e.g.    space=dict(a = Real(1, 100, 'log-uniform'), b = Real(1, 100, 'log-uniform'))
        :param Callable[[Any], float] scalar_func: Takes the return value of the experiment and turns it into a scalar
            which we aim to minimize.
        :param dict[str, Any] search_params: Args passed to parameter_search
        :return Experiment: A new experiment which runs the search and yields current-best parameters with every iteration.
        """
        assert space is not None, "You must specify a parameter search space.  See this method's documentation"
        if name is None:  # TODO: Set name=None in the default after deadline
            name = 'parameter_search[{}]'.format(','.join(space.keys()))

        if search_params is None:
            search_params = {}

        def objective(**current_params):
            output = self.call(**current_params)
            if scalar_func is not None:
                output = scalar_func(output)
            return output

        from artemis.experiments import ExperimentFunction

        @ExperimentFunction(name = self.name + '.'+ name, show = show_parameter_search_record, one_liner_function=parameter_search_one_liner)
        def search_exp():
            if is_test_mode():
                nonlocal n_calls
                n_calls = 3  # When just verifying that experiment runs, do the minimum

            for iter_info in parameter_search(objective, n_calls=n_calls, space=space, **search_params):
                info = dict(names=list(space.keys()), x_iters =iter_info.x_iters, func_vals=iter_info.func_vals, score = iter_info.func_vals, x=iter_info.x, fun=iter_info.fun)
                latest_info = {name: val for name, val in izip_equal(info['names'], iter_info.x_iters[-1])}
                print(f'Latest: {latest_info}, Score: {iter_info.func_vals[-1]:.3g}')
                yield info

        self.variants[name] = search_exp
        search_exp.tag('psearch')  # Secret feature that makes it easy to select all parameter experiments in ui with "filter tag:psearch"
        return search_exp

    def tag(self, tag):
        """
        Add a "tag" - a string identifying the experiment as being in some sort of group.
        You can use tags in the UI with 'filter tag:my_tag' to select experiments with a given tag
        :param tag:
        :return:
        """
        self._tags.add(tag)
        return self

    def get_tags(self):
        return self._tags


def show_parameter_search_record(record):
    from tabulate import tabulate
    result = record.get_result()
    table = tabulate([list(xs)+[fun] for xs, fun in zip(result['x_iters'], result['func_vals'])], headers=list(result['names'])+['score'])
    print(table)


def parameter_search_one_liner(result):
    return f'{len(result["x_iters"])} Runs : ' + ', '.join(f'{k}={v:.3g}' for k, v in izip_equal(result['names'], result['x'])) + f' : Score = {result["fun"]:.3g}'


_GLOBAL_EXPERIMENT_LIBRARY = OrderedDict()


class ExperimentNotFoundError(Exception):
    def __init__(self, experiment_id):
        Exception.__init__(self,'Experiment "{}" could not be loaded, either because it has not been imported, or its definition was removed.'.format(experiment_id))


def clear_all_experiments():
    _GLOBAL_EXPERIMENT_LIBRARY.clear()


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
            add_two_numbers.add_variant(a=3)

        for ex in exps:
            ex.add_variant(b=4)

    :rtype: Generator[:class:`Experiment`]
    """
    current_len = len(_GLOBAL_EXPERIMENT_LIBRARY)
    new_experiments = []
    yield new_experiments
    for ex in list(_GLOBAL_EXPERIMENT_LIBRARY.values())[current_len:]:
        new_experiments.append(ex)


def _register_experiment(experiment):
    assert experiment.name not in _GLOBAL_EXPERIMENT_LIBRARY, 'You have already registered an experiment named {} in {}'.format(experiment.name, inspect.getmodule(experiment.get_root_function()).__name__)
    _GLOBAL_EXPERIMENT_LIBRARY[experiment.name] = experiment


def get_nonroot_global_experiment_library():
    return OrderedDict((name, exp) for name, exp in _GLOBAL_EXPERIMENT_LIBRARY.items() if not exp.is_root)


def get_ordered_descendents_of_root(root_experiment):
    """
    :param Experiment root_experiment: An experiment which has variants
    :return List[Experiment]: A list of the descendents (i.e. variants and subvariants) of the root experiment, in the
        order in which they were created
    """
    descendents_of_root = set(ex for ex in root_experiment.get_all_variants(include_self=True))
    return [ex for ex in get_nonroot_global_experiment_library().values() if ex in descendents_of_root]


def get_experiment_info(name):
    experiment = load_experiment(name)
    return str(experiment)


def load_experiment(experiment_id):
    try:
        return _GLOBAL_EXPERIMENT_LIBRARY[experiment_id]
    except KeyError:
        raise ExperimentNotFoundError(experiment_id)


def is_experiment_loadable(experiment_id):
    assert isinstance(experiment_id, string_types), 'Expected a string for experiment_id, not {}'.format(experiment_id)
    return experiment_id in _GLOBAL_EXPERIMENT_LIBRARY


def _kwargs_to_experiment_name(kwargs):
    string = ','.join('{}={}'.format(argname, kwargs[argname]) for argname in sorted(kwargs.keys()))
    string = string.replace('/', '_SLASH_')
    return string


@contextmanager
def hold_global_experiment_libary(new_lib = None):
    if new_lib is None:
        new_lib = OrderedDict()

    global _GLOBAL_EXPERIMENT_LIBRARY
    oldlib = _GLOBAL_EXPERIMENT_LIBRARY
    _GLOBAL_EXPERIMENT_LIBRARY = new_lib
    yield _GLOBAL_EXPERIMENT_LIBRARY
    _GLOBAL_EXPERIMENT_LIBRARY = oldlib


def get_global_experiment_library():
    return _GLOBAL_EXPERIMENT_LIBRARY


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

    atexit.register(clean_on_close)  # We register this on exit to avoid race conditions with system commands when we open figures externally
