from collections import namedtuple, OrderedDict
from abc import abstractmethod
from artemis.general.nested_structures import flatten_struct
from artemis.plotting.drawing_plots import redraw_figure
from artemis.plotting.easy_plotting import plot_data_dict
from matplotlib import pyplot as plt

from artemis.plotting.matplotlib_backend import get_plot_from_data

__author__ = 'peter'


class BaseStream(object):
    def __init__(self, update_every = 1, figure = None):
        self._plots = None
        self._counter = -1
        self._update_every = update_every
        self._plot_keys = None
        self._fig = figure
        # TODO: Allow plots to be updated every iteration but only rendered every N'th iteration.  Important for streaming.

    def update(self, name = None):
        """
        Update all plots.  Note that when calling this function, we assume that you've updated the data returned by the
        callback.  When you call update multiple times without updating the data it can lead to incorrect results on
        plots with history (samples will be repeated)
        :param name: The name of the plot to update.  If not specified, updates all plots.
        """
        self._counter += 1
        if self._counter % self._update_every != 0:
            return

        data_dict = self._get_data_structure()  # dict : str -> IPlot

        if self._plots is None or set(data_dict.keys()) != self._plot_keys:
            # Note - this causes us to reset all plots (including ones with history
            # every time a new plot comes in, but that's ok for now).
            if self._fig is not None:
                self._fig.clf()
            else:
                self._fig = plt.figure()
            self._plots = self._get_plots_from_first_data(data_dict)
            self._plot_keys = set(self._plots.keys())
            plot_data_dict(data_dict, plots = self._plots, hang = False, figure = self._fig)
        else:
            if name is None:  # Update all plots
                for k, v in data_dict.items():
                    self._plots[k].update(v)
                    self._plots[k].plot()
            else:
                self._plots[name].update(data_dict[name])
                self._plots[name].plot()

        redraw_figure(self._fig)

    @abstractmethod
    def _get_data_structure(self):
        """
        :return a dict<s    tr: data> where data is some form of plottable data
        """

    @abstractmethod
    def _get_plots_from_first_data(self, first_data):
        """
        :return: a dict<str: IPlot> containing the plots corresponding to each element of the data.
        """


class LiveStream(BaseStream):
    """
    Lets you automatically generate live plots from some arbitrary data structure returned by a callback.
    """

    def __init__(self, callback, custom_handlers = {}, plot_mode = 'live', plot_types = {}, update_every=1,
                 figure = None, **plot_preference_kwargs):
        """
        :param callback: Some function that takes no arguments and returns some object.
        :param custom_handlers: A dict<type: function>.  If there's an object of one of the listed types
            returned from your callback, the function will take that object and return plot data from it.
        :param plot_mode: {'live', 'static', 'image'} - Determines what kind of plots to make for the data.
            See get_plot_from_data.
        :param update_every: Use this to only update the plot periodically - generally for speed.
        :param plot_preference_kwargs: Get passed down to get_plot_from_data
        """
        assert hasattr(callback, '__call__'), 'Your callback must be callable.'
        self._callback = callback
        self._custom_handlers=custom_handlers
        self._plot_mode = plot_mode
        self._plot_preference_kwargs = plot_preference_kwargs
        self._plot_types = plot_types
        BaseStream.__init__(self, update_every=update_every, figure = figure)

    def add_plot_type(self, key, plot_type):
        """
        You can pre-specify a plot to use for a given key.  The fact that you can add them dynamically is used in
        dbplot, where the data structure can change dynamically.
        """
        self._plot_types[key] = plot_type

    def _get_data_structure(self):
        struct = self._callback()
        assert struct is not None, 'Your plotting-data callback returned None.  Probably you forgot to include a return statememnt.'

        flat_struct = flatten_struct(struct, custom_handlers=self._custom_handlers, detect_duplicates=False)  # list<*tuple<str, data>>
        return OrderedDict(flat_struct)

    def _get_plots_from_first_data(self, first_data):
        return {k: get_plot_from_data(v, mode = self._plot_mode, **self._plot_preference_kwargs)
            if k not in self._plot_types else self._plot_types[k] for k, v in first_data.items()}


LivePlot = namedtuple('PlotBuilder', ['plot', 'cb'])


class LiveCanal(BaseStream):
    """
    Lets you make live plots by defining a dict of LivePlot objects, which contain the plot type and the data callback.
    LiveCanal gives you more control over your plots than LiveStream
    """

    def __init__(self, live_plots, **kwargs):
        """
        :param live_plots: A dict<str: (LivePlot OR function)>.  If the value is a LivePlot, you specify the type of
            plot to create.  Otherwise, you just specify a callback function, and the plot type is determined automatically
            based on the data.
        :param kwargs: Passed up to BaseStream
        """
        self._live_plots = live_plots
        self._callbacks = {k: lp.cb if isinstance(lp, LivePlot) else lp for k, lp in live_plots.items()}
        BaseStream.__init__(self, **kwargs)

    def _get_data_structure(self):
        return {k: cb() for k, cb in self._callbacks.items()}

    def _get_plots_from_first_data(self, first_data):
        # first_data = dict(first_data)
        return {k: pb.plot if isinstance(pb, LivePlot) else get_plot_from_data(first_data[k], mode = 'live') for k, pb in self._live_plots.items()}
