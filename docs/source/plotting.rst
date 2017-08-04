

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

    for _ in xrange(50):
        dbplot(np.random.randn(20, 10), 'random data')


A plot will come up showing the random data.

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot
    import numpy as np

    for _ in xrange(50):
        dbplot(np.random.randn(20, 10), 'random line data', plot_type='line')


You can include multiple plots:

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot
    import numpy as np

    for _ in xrange(50):
        dbplot(np.random.randn(20, 2), 'random line data', plot_type='line')
        dbplot(np.random.randn(10, 10), 'random grid data')


If you plot many things, you may want to "hold" your plots, so that they all update together.  This speeds up the rate of
plotting:

.. code-block:: python

    from artemis.plotting.db_plotting import dbplot, hold_dbplots
    import numpy as np

    for _ in xrange(50):
        with hold_dbplots():
            dbplot(np.random.randn(20, 2), 'random line data', plot_type='line')
            dbplot(np.random.randn(10, 10), 'random grid data')
            dbplot(np.random.randn(4), 'random line history')
            dbplot(np.random.rand(3), 'random bars', plot_type='bar')
            dbplot([np.random.rand(20, 20), np.random.randn(20, 16, 3)], 'multi image')


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


######################
Browser-Plotting
######################

After installing, you should have a file ``~/.artemisrc``.

To use the web backend, edit the ``backend`` field to ``matplotlib-web``.

To try it you can run the commands described above for dbplot.