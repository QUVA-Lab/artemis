from collections import OrderedDict
from functools import partial

from artemis.experiments.experiment_record import record_experiment
from artemis.experiments.experiments import Experiment, get_global_experiment_library


# ALTERNATE INTERFACES.
# We keep these for backwards compatibility and to show how else we could organize the experiment API.
# These are out of use, as we at Artemis inc. prefer the @experiment_function decorator.


# Register interface:

def register_experiment(name, function, description = None, conclusion = None, versions = None, current_version = None, **kwargs):
    """
    This is the old interface to experiments.  We keep it, for now, for the sake of
    backwards-compatibility.

    In the future, use the @experiment_function decorator instead.
    """
    info = OrderedDict()
    if description is not None:
        info['Description'] = description
    if conclusion is not None:
        info['Conclusion'] = conclusion

    if versions is not None:
        if current_version is None:
            assert len(versions)==1
            current_version = versions.keys()[0]
        assert current_version is not None
        function = partial(function, **versions[current_version])

    assert name not in get_global_experiment_library(), 'An experiment with name "%s" has already been registered!' % (name, )
    experiment = Experiment(name = name, function=function, info=info, **kwargs)
    get_global_experiment_library()[name] = experiment
    return experiment


# Start/Stop Interface:

_CURRENT_EXPERIMENT_CONTEXT = None


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
        assert experiment_name not in get_global_experiment_library(), "Experiment %s is already in the library" % (experiment_name, )
        self.__dict__[experiment_name] = experiment
        get_global_experiment_library()[experiment_name] = experiment

    def get_experiments(self):
        return get_global_experiment_library()

    def __getattr__(self, name):
        if name in get_global_experiment_library():
            return get_global_experiment_library()[name]
        else:
            return _ExperimentConstructor(name)


class _ExperimentConstructor(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        if self.name in get_global_experiment_library():
            raise Exception("You tried to run create experiment '%s', but it already exists in the library.  Give it another name!" % (self.name, ))
        return register_experiment(name = self.name, **kwargs)

    def run(self):
        raise Exception("You tried to run experiment '%s', but it hasn't been made yet!" % (self.name, ))


ExperimentLibrary = _ExpLibClass()


