

Artemis Experiments Documentation
===================================

The Artemis Experiment Framework helps you to keep track of your experiments and their results.  It is an alternative to `Sacred <http://sacred.readthedocs.io/en/latest/>`_, with the goal of being more intuitive to use. 

For details on the Experiment API, see :doc:`experiment-api`.  For a in introduction to the framework, read on...

######################
A Basic Example
######################


Using this module, you can turn your main function into an "Experiment", which, when run, stores all console output, plots,
and computed results to disk (in ~/.artemis/experiments)

Any function that can be called alone with no arguments can be turned into an experiment using the @experiment_function
decorator:

.. code-block:: python

    from artemis.experiments import experiment_function

    @experiment_function
    def multiply_3_numbers(a=1, b=2, c=3):
        return a*b*c

This turns the function into an Experiment object, which, in addition to still being a callable function, has methods ``run()``, ``add_variant(...)`` and ``get_variant()``.   It's important to give this function a unique name (rather than ``main()``, or something) because this name is used to link the experiment to the records that it has produced.

If we want to run our experiment, and save all text outputs and plots to disk, we can call the ``run`` method:

.. code-block:: python

    record = multiply_3_numbers.run()

Before we get to reviewing the results, we may want to create a "variant" on this experiment, with a different set of parameters.  For this, we can use the ``add_variant`` method: 

.. code-block:: python

    multiply_3_numbers.add_variant('higher-ab', a=4, b=5)

If we want to access this variant later, we can call ``get_variant``:.

.. code-block:: python

    ex = multiply_3_numbers.get_variant('higher-ab')

To open up a menu where you can see and run all experiments (and their variants) that have been created we run:

.. code-block:: python

    multiply_3_numbers.browse()

This will give us an output that looks something like this::

    ==================== Experiments ====================
      E#  R#    Name                          All Runs                    Duration         Status           Valid    Result
    ----  ----  ----------------------------  --------------------------  ---------------  ---------------  -------  --------
       0  0     multiply_3_numbers            2017-08-03 10:34:51.150555  0.0213599205017  Ran Succesfully  Yes      6
       1        multiply_3_numbers.higher-ab  <No Records>                -                -                -        -
    -----------------------------------------------------
    Enter command or experiment # to run (h for help) >>


This indicates that we have a saved record of our experiment (created when we called ``multiply_3_numbers.run()``), but
none of the variant ``higher-ab``.  In the UI, we can run this variant by entering ``run 1``::

    Enter command or experiment # to run (h for help) >> run 1

After running, we will see the status of our experiments updated::

    ==================== Experiments ====================
      E#    R#  Name                          All Runs                      Duration  Status           Valid      Result
    ----  ----  ----------------------------  --------------------------  ----------  ---------------  -------  --------
       0     0  multiply_3_numbers            2017-08-03 10:34:51.150555   0.0213599  Ran Succesfully  Yes             6
       1     0  multiply_3_numbers.higher-ab  2017-08-03 10:38:45.836260   0.0350862  Ran Succesfully  Yes            60
    -----------------------------------------------------
    Enter command or experiment # to run (h for help) >>


From the UI we have access to a variety of commands for showing and comparing experiments.  For example, `argtable` prints
a table comparing the results of the different experiments::

    Enter command or experiment # to run (h for help) >> argtable all
        -------------------------------------------------------  ------------------  ---------------  -----------  --------------  ------
                                                                 Function            Run Time         Common Args  Different Args  Result
        2017.08.03T10.34.51.150555-multiply_3_numbers            multiply_3_numbers  0.0213599205017  c=3          a=1, b=2        6
        2017.08.03T10.38.45.836260-multiply_3_numbers.higher-ab  multiply_3_numbers  0.0350861549377  c=3          a=4, b=5        60
        -------------------------------------------------------  ------------------  ---------------  -----------  --------------  ------



######################
More Examples
######################

* `An example demonstrating Artemis's Experiment framework on a simple MNIST classification task <https://github.com/QUVA-Lab/artemis/blob/master/artemis/examples/demo_mnist_logreg.py>`_
* `Step-by-step tutorial on using Artemis to organize your Experiments <https://rawgit.com/petered/data/master/gists/experiment_tutorial.html>`_
