from collections import OrderedDict
from artemis.plotting.live_plotting import LiveStream
from artemis.plotting.plotting_backend import LinePlot,ImagePlot

__author__ = 'peter'


PLOT_DATA = OrderedDict()
STREAM = None


def dbplot(data, name = None, plot_constructor = None, **kwargs):
    """
    Quick plot of some variable - you can call this in a loop and it will know to update the
    same plot.  See test_db_plotting.py for examples.

    :param data: The data to plot
    :param name: The name of this plot (you need to specify this if you want to make more than one plot)
    :param plot_mode: Affects the type of plots generated.
        'live' is more appropriate when you're monitoring something and want an online plot with memory
        'static' is better for step-by-step debugging, or plotting from the debugger.
    """

    if not isinstance(name, str):
        name = str(name)

    if name not in PLOT_DATA and plot_constructor is not None:
        if isinstance(plot_constructor, str):
            plot_constructor = {
                'line': LinePlot,
                'img': ImagePlot,
                }[plot_constructor]

        assert hasattr(plot_constructor, '__call__'), 'Plot constructor must be callable!'
        stream = get_dbplot_stream(**kwargs)
        # Following is a kludge - the data is flattened in LivePlot, so we reference
        # it by the "flattened" key.
        flattened_key = "['%s']" % name

        stream.add_plot_type(flattened_key, plot_constructor())

    set_plot_data_and_update(name, data, **kwargs)


def set_plot_data_and_update(name, data, **kwargs):
    PLOT_DATA[name] = data
    stream = get_dbplot_stream(**kwargs)
    flattened_key = "['%s']" % name
    stream.update(flattened_key)


def get_dbplot_stream(**kwargs):
    global STREAM
    if STREAM is None:
        STREAM = LiveStream(lambda: PLOT_DATA, **kwargs)
    return STREAM
