from abc import ABCMeta, abstractmethod
from itertools import cycle

from matplotlib.image import AxesImage

from six import string_types

from artemis.config import get_artemis_config_value
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.data_conversion import (put_data_in_grid, RecordBuffer, data_to_image, put_list_of_images_in_array,
    UnlimitedRecordBuffer)
from matplotlib import pyplot as plt
import numpy as np
from six.moves import xrange


__author__ = 'peter'


class IPlot(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    def update_and_plot(self, data):
        self.update(data)
        self.plot()


class HistoryFreePlot(IPlot):

    def update(self, data):
        self._last_full_plot_data = data

    def plot(self):
        self._plot_last_data(self._last_full_plot_data)

    @abstractmethod
    def _plot_last_data(self, data):
        pass


colour_cycle = 'bgrmcyk'


class ImagePlot(HistoryFreePlot):

    def __init__(self, interpolation = 'nearest', show_axes = False, show_clims = True, clims = None, only_grow_clims = False,
            channel_first = False, aspect = 'auto', cmap = 'gray', is_colour_data = None):

        """
        :param interpolation: How to interpolate array to form the image {'none', 'nearest', ''bilinear', 'bicubic', ... (see plt.imshow)}
        :param show_axes: Show axes marks (numbers along the side showing pixel locations)
        :param show_clims: Print the range of the colour scale at the bottom
        :param clims: (lower, upper) limit to colour-scale (or None to set it to the range of the data)
        :param aspect: 'auto' to stretch the image to the shape of the plot, 'equal' to maintain aspect ratio
        :param cmap: Colourmap {'gray', 'jet', ...}
        :param is_colour_data: Identify whether the image consists of colour data.  Usually, if you leave this at None,
            it will figure out the correct thing automatically, but this can, for instance, be used to distinquish a case
            where an image of shape shape is (12, 18, 3) should be interpreted as a (12x18) colour image or 12 18x3
            black and white images.  (Default in this case would be colour image)
        """
        self._plot = None
        self._interpolation = interpolation
        self._show_axes = show_axes
        self._clims = clims
        self._aspect = aspect
        self._cmap = cmap
        self._is_colour_data = is_colour_data
        self.show_clims = show_clims
        self.only_grow_clims = only_grow_clims
        self.channel_first = channel_first
        if only_grow_clims:
            self._old_clims = (float('inf'), -float('inf'))

    def _plot_last_data(self, data):

        if self.channel_first:
            if data.ndim==3:
                assert data.shape[0] in (1, 3)
                data = np.rollaxis(data, 0, 3)
            elif data.ndim==4:
                assert data.shape[1] in (1, 3)
                data = np.rollaxis(data, 1, 4)
            else:
                print('Image should be 3D (3, size_y, size_x) or 4D (n_images, 3, size_y, size_x) for channel-first-mode.')

        if len(data)==0:
            plottable_data = np.zeros((16, 16, 3), dtype = np.uint8)
            clims = (0, 1)
        else:
            if isinstance(data, list):
                data = put_list_of_images_in_array(data, fill_colour=(np.nan, np.nan, np.nan))
            elif data.ndim == 1:
                data = data[None]

            clims = ((np.nanmin(data), np.nanmax(data)) if data.size != 0 else (0, 1)) if self._clims is None else self._clims

            if self.only_grow_clims:
                clims = min(self._old_clims[0], clims[0]), max(self._old_clims[1], clims[1])
                self._old_clims = clims

            if self._is_colour_data is None:
                self._is_colour_data = data.shape[-1]==3

            plottable_data = put_data_in_grid(data, clims = clims, cmap = self._cmap, is_color_data = self._is_colour_data, fill_colour = np.array((0, 0, 128)), nan_colour = np.array((0, 0, 128))) \
                if not (self._is_colour_data and data.ndim==3 or data.ndim==2) else \
                data_to_image(data, clims = clims, cmap = self._cmap, nan_colour = np.array((0, 0, 128)))

        if self._plot is None:
            self._plot = plt.imshow(plottable_data, interpolation = self._interpolation, aspect = self._aspect, cmap = self._cmap)
            if not self._show_axes:
                self._plot.axes.tick_params(labelbottom = 'off')
                self._plot.axes.get_yaxis().set_visible(False)
        else:
            self._plot.set_array(plottable_data)
        if self.show_clims:
            self._plot.axes.set_xlabel('{:.3g} <> {:.3g}'.format(*clims))


class MovingImagePlot(ImagePlot):

    def __init__(self, buffer_len = 100, **kwargs):
        ImagePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)

    def update(self, data):
        if np.isscalar(data):
            data = np.array([data])
        elif data.ndim != 1 and data.size == np.max(data.shape):
            data = data.flatten()
        else:
            assert data.ndim == 1
        buffer_data = self._buffer(data)
        ImagePlot.update(self, buffer_data.T)


class LinePlot(HistoryFreePlot):

    def __init__(self, x_axis_type = 'lin', y_axis_type = 'lin', x_bounds = (None, None), y_bounds = (None, None), y_bound_extend = (.05, .05),
                 x_bound_extend = (0, 0), make_legend = None, axes_update_mode = 'fit', add_end_markers = False, legend_entries = None,
                 legend_entry_size = 8, color=None, linewidth=None, linestyle=None, plot_kwargs = None, allow_axis_offset = False):
        """
        :param y_axis_type: 'lin' (only one supported now)
        :param x_bounds: A tuple of (lower_bound, upper_bound), where None means automatic
        :param y_bounds: A tuple of (lower_bound, upper_bound), where None means automatic
        :param y_bound_extend: A tuple of (float, float) indicating how much to extend the automatically determined bounds.
        :param x_bound_extend: A tuple of (float, float) indicating how much to extend the automatically determined bounds.
        :param make_legend: True to make a legend
            'fit': Fit to current curve
            'expand': Expand axes to contain curve (useful if plotting over pre-existing curve)
        :param add_end_markers: Add a circle to indicate a start, and an "X" to indicate the end of each line.
        :param legend_entries: Entries of the legend.  Should be one per line in the data.
        :param legend_entry_size: Font size for legend entries.
        :param plot_kwargs:
        """
        # assert y_axis_type == 'lin', 'Changing axis scaling not supported yet'
        self.x_axis_type = x_axis_type
        self.y_axis_type = y_axis_type
        self._plots = None
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.x_bound_extend = x_bound_extend
        self.y_bound_extend = y_bound_extend
        self.make_legend=make_legend
        if plot_kwargs is None:
            plot_kwargs = {}
        if color is not None:
            plot_kwargs['color'] = color
        if linewidth is not None:
            plot_kwargs['linewidth'] = linewidth
        if linestyle is not None:
            plot_kwargs['linestyle'] = linestyle
        self.plot_kwargs = plot_kwargs
        self.axes_update_mode = axes_update_mode
        self.add_end_markers = add_end_markers
        self._end_markers = []
        self.legend_entries = [legend_entries] if isinstance(legend_entries, string_types) else legend_entries
        self.legend_entry_size = legend_entry_size
        self.allow_axis_offset = allow_axis_offset

    def _plot_last_data(self, data):
        """
        :param data: Can be:
            An array of y_data (x_data is assumed to be np.arange(data.shape[0])
            A 2-tuple of (x_data, y_data)
        """

        # Format all data as (list<x_data>, list<y_data>)
        if isinstance(data, tuple) and len(data)==2:
            x_data, y_data = data
        else:
            x_data = None
            y_data = data
        if isinstance(y_data, np.ndarray):
            n_lines = 1 if y_data.ndim==1 else y_data.shape[1]
            x_data = [np.arange(1, y_data.shape[0]+1) if self.x_axis_type=='log' else np.arange(y_data.shape[0])] * n_lines if x_data is None else [x_data] * n_lines if x_data.ndim==1 else x_data.T
            # x_data = y_data.T if y_data.ndim==2 else y_data[None] if y_data.ndim==1 else bad_value(y_data.ndim)
            y_data = y_data.T if y_data.ndim==2 else y_data[None] if y_data.ndim==1 else bad_value(y_data.ndim, "Line plot data must be 1D or 2D, not {}D".format(y_data.ndim))
        else:  # List of arrays
            if all(d.ndim==0 for d in data):  # Turn it into one line
                x_data = np.arange(1, y_data.shape[0]+1) if self.x_axis_type=='log' else np.arange(y_data.shape[0])[None, :]
                y_data = np.array(data)[None, :]
            else:  # List of arrays becomes a list of lines
                x_data = [np.arange(1, y_data.shape[0]+1) if self.x_axis_type=='log' else np.arange(y_data.shape[0]) for d in y_data] if x_data is None else x_data
        assert len(x_data)==len(y_data), "The number of lines in your x-data (%s) does not match the number in your y-data (%s)" % (len(x_data), len(y_data))

        lower, upper = (np.nanmin(y_data) if self.y_bounds[0] is None else self.y_bounds[0], np.nanmax(y_data)+1e-9 if self.y_bounds[1] is None else self.y_bounds[1])
        left, right = (np.nanmin(x_data) if self.x_bounds[0] is None else self.x_bounds[0], np.nanmax(x_data)+1e-9 if self.x_bounds[1] is None else self.x_bounds[1])

        if left==right:
            right+=1e-9

        # Expand x_bound:
        delta = right-left if left-right >0 else 1e-9
        left -= self.x_bound_extend[0]*delta
        right += self.x_bound_extend[1]*delta

        # Expand y_bound:
        if self.y_axis_type=='lin':
            delta = upper-lower if upper-lower >0 else 1e-9
            lower -= self.y_bound_extend[0]*delta
            upper += self.y_bound_extend[1]*delta
        elif self.y_axis_type=='log':
            lower *= (1-self.y_bound_extend[0])
            upper *= (1+self.y_bound_extend[1])
        else:
            raise Exception(self.y_axis_type)

        if self._plots is None:
            self._plots = []
            plt.gca().autoscale(enable=False)
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(self.allow_axis_offset)
            plt.gca().get_xaxis().get_major_formatter().set_useOffset(self.allow_axis_offset)
            if self.x_axis_type!='lin':
                plt.gca().set_xscale(self.x_axis_type)
            if self.y_axis_type!='lin':
                plt.gca().set_yscale(self.y_axis_type)

            if isinstance(self.plot_kwargs, dict):
                plot_kwargs = [self.plot_kwargs]*len(x_data)
            elif isinstance(self.plot_kwargs, (list, tuple)):
                assert len(self.plot_kwargs)==len(x_data), "You provided a list of {0} plot kwargs, but {1} lines".format(len(self.plot_kwargs), len(x_data))
                plot_kwargs = self.plot_kwargs
            for i, (xd, yd, legend_entry) in enumerate(zip(x_data, y_data, self.legend_entries if self.legend_entries is not None else [None]*len(x_data))):
                p, =plt.plot(xd, yd, label = legend_entry, **plot_kwargs[i])
                self._plots.append(p)
                _update_axes_bound(p.axes, (left, right), (lower, upper), self.axes_update_mode)
                if self.add_end_markers:
                    colour = p.get_color()
                    self._end_markers.append((plt.plot(xd[[0]], yd[[0]], marker='.', markersize=20, color=colour)[0], plt.plot(xd[0], yd[0], marker='x', markersize=10, mew=4, color=colour)[0]))

            if (self.make_legend is True) or (self.make_legend is None and (self.legend_entries is not None or len(y_data)>1)):
                plt.legend(loc='best', framealpha=0.5, prop={'size':self.legend_entry_size})
                # entries = [str(i) for i in xrange(len(y_data))] if self.legend_entries is None else self.legend_entries
                # assert len(self._plots) == len(self.legend_entries), 'You have %s plots but you specified %s entries for the legend: %s' % (len(self._plots), len(entries), entries)
                # handles, labels = plt.gca().get_legend_handles_labels()
                # if len(handles)==0:
                #     plt.legend(handles + self._plots, labels+entries, loc='best', prop={'size':8})
                # else:
                #     plt.gca().set_legend_handles_labels(handles + self._plots, labels+entries)

        else:
            for i, (p, xd, yd) in enumerate(zip(self._plots, x_data, y_data)):
                p.set_xdata(xd)
                p.set_ydata(yd)
                _update_axes_bound(p.axes, (left, right), (lower, upper), self.axes_update_mode)
                if self.add_end_markers:
                    self._end_markers[i][0].set_xdata(xd[[0]])
                    self._end_markers[i][0].set_ydata(yd[[0]])
                    self._end_markers[i][1].set_xdata(xd[[-1]])
                    self._end_markers[i][1].set_ydata(yd[[-1]])

        # plt.legend(loc='best', framealpha=0.5, prop={'size': self.legend_entry_size})


def _update_axes_bound(ax, left_and_right, lower_and_upper, mode = 'fit'):
    left, right = left_and_right
    lower, upper = lower_and_upper
    if mode=='fit':
        ax.set_xbound(left, right)
        ax.set_ybound(lower, upper)
    elif mode=='expand':
        old_left, old_right = ax.get_xbound()
        old_lower, old_upper = ax.get_ybound()
        if (old_left, old_right, old_lower, old_upper) == (0, 1, 0, 1):  # Virgin axes (probably)... overwrite them
            ax.set_xbound(left, right)
            ax.set_ybound(lower, upper)
        else:
            ax.set_xbound(min(old_left, left), max(old_right, right))
            ax.set_ybound(min(old_lower, lower), max(old_upper, upper))
    else:
        raise Exception('No axis update mode: "%s"' % (mode, ))


class BoundingBoxPlot(LinePlot):

    def __init__(self, axes_update_mode='expand', **kwargs):
        super(BoundingBoxPlot, self).__init__(axes_update_mode=axes_update_mode, **kwargs)
        self._image_handle = None
        self._last_data_shape = None

    def update(self, data):
        """
        :param data: A (left, bottom, right, top) bounding box.
        """
        if self._image_handle is None:
            self._image_handle = next(c for c in plt.gca().get_children() if isinstance(c, AxesImage))

        data_shape = self._image_handle.get_array().shape # Hopefully this isn't copying
        if data_shape != self._last_data_shape:
            extent =(-.5, data_shape[1]-.5, data_shape[0]-.5, -.5)
            self._image_handle.set_extent(extent)
            plt.gca().set_xlim(extent[:2])
            plt.gca().set_ylim(extent[2:])
            self._last_data_shape = data_shape

        l, b, r, t = data
        # x = np.array([l+.5, l+.5, r+.5, r+.5, l+.5])  # Note: should we be adding .5? The extend already subtracts .5
        # y = np.array([t+.5, b+.5, b+.5, t+.5, t+.5])
        x = np.array([l, l, r, r, l])  # Note: should we be adding .5? The extend already subtracts .5
        y = np.array([t, b, b, t, t])

        LinePlot.update(self, (x, y))


class MovingPointPlot(LinePlot):

    def __init__(self, buffer_len=None, **kwargs):
        """
        :param buffer_len: An integar to keep a fixed-length window, or None to keep an expanding buffer
        :param kwargs:
        :return:
        """
        LinePlot.__init__(self, **kwargs)
        self._buffer = UnlimitedRecordBuffer() if buffer_len is None else RecordBuffer(buffer_len)
        self.x_data = np.arange(-buffer_len+1, 1) if buffer_len is not None else None

    def update(self, data):
        if not np.isscalar(data) or isinstance(data, np.ndarray):
            data = np.array(data)
        buffer_data = self._buffer(data)
        x_data = np.arange(len(buffer_data)) if self.x_data is None else self.x_data
        LinePlot.update(self, (x_data, buffer_data))

    def plot(self):
        LinePlot.plot(self)


class Moving2DPointPlot(LinePlot):

    def __init__(self, buffer_len=None, **kwargs):
        LinePlot.__init__(self, add_end_markers=True, **kwargs)
        self._y_buffer = UnlimitedRecordBuffer() if buffer_len is None else RecordBuffer(buffer_len)
        self._x_buffer = UnlimitedRecordBuffer() if buffer_len is None else RecordBuffer(buffer_len)

    def update(self, x_data_and_y_data):
        x_data, y_data = x_data_and_y_data
        x_buffer_data = self._x_buffer(x_data)
        y_buffer_data = self._y_buffer(y_data)

        valid_sample_start = np.argmax(~np.any(np.isnan(y_buffer_data.reshape(y_buffer_data.shape[0], -1)), axis=1))
        LinePlot.update(self, (x_buffer_data[valid_sample_start:], y_buffer_data[valid_sample_start:]))

    def plot(self):
        LinePlot.plot(self)


class TextPlot(IPlot):

    def __init__(self, max_history = 8, horizontal_alignment = 'left', vertical_alignment = 'bottom', size = 'medium'):
        """
        :param horizontal_alignment: {'left', 'center', 'right'}
        :param vertical_alignment: {'top', 'center', 'bottom', 'baseline'}
        :param size: [size in points | "xx-small" | "x-small" | "small" | "medium" | "large" | "x-large" | "xx-large" ]
        :return:
        """
        self._buffer = RecordBuffer(buffer_len = max_history, initial_value='')
        self._max_history = 10
        self._text_plot = None
        self.horizontal_alignment = horizontal_alignment
        self.vertical_alignment = vertical_alignment
        self.size = size
        self._x_offset = {'left': 0.05, 'center': 0.5, 'right': 0.95}[self.horizontal_alignment]
        self._y_offset = {'bottom': 0.05, 'center': 0.5, 'top': 0.95}[self.vertical_alignment]

    def update(self, string):
        if not isinstance(string, string_types):
            string = str(string)
        history = self._buffer(string)
        self._full_text = '\n'.join(history)

    def plot(self):
        if self._text_plot is None:
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_axis_off()
            self._text_plot = ax.text(self._x_offset, self._y_offset, self._full_text, horizontalalignment=self.horizontal_alignment, verticalalignment=self.vertical_alignment, size = self.size)
        else:
            self._text_plot.set_text(self._full_text)


class BarPlot(HistoryFreePlot):

    def __init__(self, fill_width=0.8, ):
        self._plots = None
        self.fill_width = fill_width

    def _plot_last_data(self, data):
        """
        :param data: Can be:
            A vector of values
            A (n_values, n_measures) array of values
        """
        if data.ndim==1:
            data = data[:, None]

        if self._plots is None:
            n_values, n_sources = data.shape
            width = self.fill_width/n_sources
            x_centers = np.arange(n_values)
            self._plots = [plt.bar(x_centers+width*i-self.fill_width/2., data[:, i], width=width, color=c, edgecolor=c) for i, c in zip(xrange(n_sources), cycle(colour_cycle))]
            plt.xticks(x_centers)
        else:
            for i, plot in enumerate(self._plots):
                for rect, h in zip(plot, data[:, i]):
                    rect.set_height(h)
            _update_axes_bound(rect.axes, (-.5, data.shape[0]-.5), (min(0, np.min(data)), max(0, np.max(data))), mode='fit')


class HistogramPlot(IPlot):

    def __init__(self, edges, mode = 'mass', plot_type = 'bar', cumulative = False):
        assert mode in ('mass', 'density')
        edges = np.array(edges)
        self._edges = edges
        self._mode = mode
        self._binvals = np.ones(len(edges)-1)/len(edges)
        self._n_points = 0
        self._plot = None
        self._widths = np.diff(edges)
        self._lefts = edges[:-1]
        self._plot_type=plot_type
        self._cumulative = cumulative

    def update(self, data):

        if isinstance(data, (list, tuple)):
            data = np.array(data)

        # Update data
        new_n_points = self._n_points + data.size
        this_hist, _ = np.histogram(data, self._edges)
        frac = (float(data.size)/self._n_points) if self._n_points > 0 else 1
        self._binvals += this_hist * frac
        self._binvals /= max(1, np.sum(self._binvals))
        self._n_points = new_n_points
        # DIsplay
        heights = self._binvals if self._mode == 'mass' else self._binvals/self._widths
        if self._cumulative:
            heights = np.cumsum(heights)
        self._last_heights = heights

    def plot(self):
        heights = self._last_heights
        if self._plot_type == 'bar':
            if self._plot is None:
                self._plot = plt.bar(self._lefts, heights, width = self._widths)
            else:
                for rect, h in zip(self._plot, heights):
                    rect.set_height(h)
        elif self._plot_type == 'line':
            if self._plot is None:
                self._plot = plt.plot(self._edges[:-1], heights)
            else:
                self._plot[0].set_ydata(heights)

        self._plot[0].axes.set_ybound(0, np.max(heights)*1.05)


class CumulativeLineHistogram(HistogramPlot):

    def __init__(self, edges):
        HistogramPlot.__init__(self, edges, mode = 'mass', plot_type='line', cumulative=True)


def get_plot_from_data(data, mode, **plot_preference_kwargs):

    return \
        get_live_plot_from_data(data, **plot_preference_kwargs) if mode == 'live' else \
        get_static_plot_from_data(data, **plot_preference_kwargs) if mode == 'static' else \
        ImagePlot(**plot_preference_kwargs) if mode == 'image' else \
        bad_value(mode, 'Unknown plot modee: %s' % (mode, ))


def get_live_plot_from_data(data, line_to_image_threshold = 8, cmap = 'gray'):

    # TODO: Maybe refactor that so that plot objects contain their own "data validation" code, and we can
    # simply ask plots in sequence whether they can handle the data.
    if isinstance(data, string_types):
        return TextPlot()

    if isinstance(data, list):
        if all(isinstance(d, np.ndarray) for d in data):
            if all(d.ndim in (2, 3) for d in data):
                return ImagePlot()
            elif all(d.ndim==1 for d in data):
                return LinePlot()
            elif all(d.ndim==0 for d in data):
                return LinePlot()
            else:
                raise Exception("Don't know how to deal with a list of arrays with shapes: %s" % ([d.shape for d in data], ))
        elif all(np.isscalar(d) for d in data):
            return MovingPointPlot()
    elif isinstance(data, tuple):
        if len(data)==2:
            if np.isscalar(data[0]) and np.isscalar(data[1]):
                return Moving2DPointPlot()
            else:
                return LinePlot()
        else:
            raise Exception("Don't know what to do with a length-%s tuple of data" % (len(data)))

    is_scalar = np.isscalar(data) or data.shape == ()
    if is_scalar:
        data = np.array(data)

    is_1d = not is_scalar and data.size == np.max(data.shape)
    few_values = data.size < line_to_image_threshold

    if is_scalar or is_1d and few_values:
        return MovingPointPlot()
    elif is_1d:
        return MovingImagePlot()
    elif data.ndim == 2 and data.shape[1]<line_to_image_threshold:
        return LinePlot()
    elif data.ndim in (2, 3, 4, 5):
        return ImagePlot(cmap=cmap)
    else:
        raise NotImplementedError('We have no way to plot data of shape %s.  Make one!' % (data.shape, ))


def get_static_plot_from_data(data, line_to_image_threshold=8, cmap = 'gray'):

    if isinstance(data, string_types):
        return TextPlot()

    is_scalar = np.isscalar(data) or data.shape == ()
    if is_scalar or data.size==1:
        return TextPlot()

    is_1d = not is_scalar and data.size == np.max(data.shape)
    if is_1d:
        n_unique = len(np.unique(data))
        if n_unique == 2:
            return ImagePlot(cmap=cmap)
        else:
            return LinePlot()
    elif data.ndim == 2 and data.shape[1] < line_to_image_threshold:
        return LinePlot()
    else:
        return ImagePlot(cmap=cmap)


_PLOTTING_SERVER = get_artemis_config_value(section='plotting', option='plotting_server', default_generator="")
_USE_SERVER = _PLOTTING_SERVER != ""


def is_server_plotting_on():
    return _USE_SERVER


def set_server_plotting(state):
    global _USE_SERVER
    _USE_SERVER = state


def get_plotting_server_address():
    return _PLOTTING_SERVER


BACKEND = get_artemis_config_value(section='plotting', option='backend')
