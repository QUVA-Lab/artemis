

Artemis Plotting
===================================

######################
Live Plots with dbplot
######################

``dbplot`` is an easy way to create a live plot of your data.

For example, to create live updating plots of a random grid:

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot
    import numpy as np

    for _ in xrange(100):
        dbplot(np.random.randn(20, 10), 'random data')


A plot will come up showing the random data.

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot
    import numpy as np

    for _ in xrange(100):
        dbplot(np.random.randn(20, 10), 'random line data', plot_type='line')


You can include multiple plots:

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot
    import numpy as np

    for _ in xrange(100):
        dbplot(np.random.randn(20, 2), 'random line data', plot_type='line')
        dbplot(np.random.randn(10, 10), 'random grid data')


######################
dbplot documentation
######################

.. autofunction:: artemis.plotting.db_plotting.dbplot

######################
Plotting Demos
######################

* `A demo of showing how to make various kinds of live updating plots. </artemis/plotting/demo_dbplot.py>`_
* `A demo repo showing how to use Artemis from your code <https://github.com/QUVA-Lab/demo_repo)>`_
* `A guide on using Artemis for remote plotting <https://github.com/QUVA-Lab/artemis/blob/master/artemis/remote/README.md)>`_

