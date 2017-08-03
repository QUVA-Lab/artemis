from artemis.experiments.experiments import Experiment


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

    def __init__(self, display_function=None, comparison_function=None, one_liner_function=None, is_root=False):
        """
        :param display_function: A function that takes the results (whatever your experiment returns) and displays them.
        :param comparison_function: A function that takes an OrderedDict<experiment_name, experiment_return_value>.
            You can optionally define this function to compare the results of different experiments.
            You can use call this via the UI with the compare_experiment_results command.
        :param one_liner_function: A function that takes your results and returns a 1 line string summarizing them.
        :param is_root: True to make this a root experiment - so that it is not listed to be run itself.
        """
        self.display_function = display_function
        self.comparison_function = comparison_function
        self.is_root = is_root
        self.one_liner_function = one_liner_function

    def __call__(self, f):
        f.is_base_experiment = True
        ex = Experiment(
            name=f.__name__,
            function=f,
            display_function=self.display_function,
            comparison_function = self.comparison_function,
            one_liner_function=self.one_liner_function,
            is_root=self.is_root
        )
        return ex