


Experiment API
===================================


You can use the UI for most things related to running and viewing the results of experiments.  (See :doc:`experiments` for how to do that).

This document shows the methods for interacting programatically with the experiment interface.

######################
Creating Experiments
######################

Experiment Decorators are turn your python functions into experiments.  When using decorators, **be sure that your experiment
function is uniquely named** (ie, no other function in your code has the same name).
This is important because when results are saved, the name of the function is used
to identify what experiment the results belong to.


.. autofunction:: artemis.experiments.experiment_function


.. autofunction:: artemis.experiments.experiment_root


.. autoclass:: artemis.experiments.ExperimentFunction
    :members: __init__


.. autofunction:: artemis.experiments.capture_created_experiments


######################
The Experiment
######################

The above decorators return Experiment Objects, which have the following API...

.. autoclass:: artemis.experiments.experiments.Experiment
    :members:

######################
The Experiment Record
######################

When you run an Experiment, a folder is created in which the stdout, results, figures, and
other info are stored.  The ExperimentRecord object provides an API for accessing the contents of
this folder.

.. autoclass:: artemis.experiments.experiment_record.ExperimentRecord
    :members:


