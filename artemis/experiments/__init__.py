from decorators import experiment_function, experiment_root, ExperimentFunction
from experiments import capture_created_experiments, load_experiment
from experiment_record import load_experiment_record


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

    my_experiment.browse()


You may not want an experiment to be runnable (and show up in browse_experiments) - you may just want it
to serve as a basis on which to make variants.  In the latter case, you can decorate with

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