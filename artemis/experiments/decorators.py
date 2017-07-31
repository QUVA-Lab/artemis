from artemis.experiments.experiments import Experiment


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

    def __init__(self, display_function=None, comparison_function=None, one_liner_function=None, info=None, is_root=False):
        """
        :param display_function: A function that takes the results (whatever your experiment returns) and displays them.
        :param comparison_function: A function that takes an OrderedDict<experiment_name, experiment_return_value>.
            You can optionally define this function to compare the results of different experiments.
            You can use call this via the UI with the compare_results command.
        :param one_liner_function: A function that takes your results and returns a 1 line string summarizing them.
        :param info: Don't use this?
        :param is_root: True to make this a root experiment - so that it is not listed to be run itself.
        """
        self.display_function = display_function
        self.comparison_function = comparison_function
        self.info = info
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