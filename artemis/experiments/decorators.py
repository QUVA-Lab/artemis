from collections import OrderedDict

from artemis.experiments.experiment_record_view import show_record, compare_experiment_records
from artemis.experiments.experiments import Experiment
from artemis.general.display import sensible_str
from artemis.general.should_be_builtins import uniquify_duplicates, izip_equal


def experiment_function(f):
    """
    Use this decorator (@experiment_function) on a function that you want to run.  e.g.

    .. code-block:: python

        @experiment_function
        def demo_my_experiment(a=1, b=2, c=3):
            ...

    This turns your function demo_my_experiment into an experiment.  It can still be called as a normal function, but
    it now has can also be called with the methods of an Experiment object (eg. demo_my_experiment.run()).

    """
    return ExperimentFunction()(f)


def experiment_root(f):
    """
    Use this decorator on a function that you want to build variants off of:

    .. code-block:: python

        @experiment_root
        def demo_my_experiment(a, b=2, c=3):
            ...

    The root experiment is not runnable by itself, and will not appear in the list in the browse experiments UI, but
    you can call ``demo_my_experiment.add_variant(...)`` to create runnable variants.
    """
    return ExperimentFunction(is_root=True)(f)


class ExperimentFunction(object):
    """
    This is the most general decorator.  You can use this to add details on the experiment.
    """

    def __init__(self, show = show_record, compare = compare_experiment_records, display_function=None, comparison_function=None, one_liner_function=sensible_str, is_root=False):
        """
        :param show:  A function that is called when you "show" an experiment record in the UI.  It takes an experiment
            record as an argument.
        :param compare: A function that is called when you "compare" a set of experiment records in the UI.
        :param display_function: [Deprecated] A function that takes the results (whatever your experiment returns) and displays them.
        :param comparison_function: [Deprecated] A function that takes an OrderedDict<experiment_name, experiment_return_value>.
            You can optionally define this function to compare the results of different experiments.
            You can use call this via the UI with the compare_experiment_results command.
        :param one_liner_function: A function that takes your results and returns a 1 line string summarizing them.
        :param is_root: True to make this a root experiment - so that it is not listed to be run itself.
        """
        self.show = show
        self.compare = compare

        if display_function is not None:
            assert show is show_record, "You can't set both display function and show.  (display_function is deprecated)"
            show = lambda rec: display_function(rec.get_result())

        if comparison_function is not None:
            assert compare is compare_experiment_records, "You can't set both display function and show.  (display_function is deprecated)"

            def compare(records):
                record_experiment_ids_uniquified = uniquify_duplicates(rec.get_experiment_id() for rec in records)
                comparison_function(OrderedDict((unique_rid, rec.get_result()) for unique_rid, rec in izip_equal(record_experiment_ids_uniquified, records)))

        self.show = show
        self.compare = compare
        self.is_root = is_root
        self.one_liner_function = one_liner_function

    def __call__(self, f):
        f.is_base_experiment = True
        ex = Experiment(
            name=f.__name__,
            function=f,
            show=self.show,
            compare = self.compare,
            one_liner_function=self.one_liner_function,
            is_root=self.is_root
        )
        return ex