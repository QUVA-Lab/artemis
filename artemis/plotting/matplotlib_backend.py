from abc import ABCMeta, abstractmethod
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.data_conversion import put_data_in_grid, RecordBuffer, data_to_image, put_list_of_images_in_array
from matplotlib import pyplot as plt
import numpy as np

__author__ = 'peter'


class IPlot(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class HistoryFreePlot(IPlot):

    def update(self, data):
        self._last_data = data

    def plot(self):
        self._plot_last_data(self._last_data)

    @abstractmethod
    def _plot_last_data(self, data):
        pass


class ImagePlot(HistoryFreePlot):

    def __init__(self, interpolation = 'nearest', show_axes = False, show_clims = True, clims = None, only_grow_clims = True, aspect = 'auto', cmap = 'gray', is_colour_data = None):
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
        if only_grow_clims:
            self._old_clims = (float('inf'), -float('inf'))

    def _plot_last_data(self, data):

        if len(data)==0:
            plottable_data = np.zeros((16, 16, 3), dtype = np.uint8)
            clims = (0, 1)
        else:
            if isinstance(data, list):
                data = put_list_of_images_in_array(data)

            if data.ndim == 1:
                data = data[None]

            clims = ((np.nanmin(data), np.nanmax(data)) if data.size != 0 else (0, 1)) if self._clims is None else self._clims

            if self.only_grow_clims:
                clims = min(self._old_clims[0], clims[0]), max(self._old_clims[1], clims[1])
                self._old_clims = clims

            if self._is_colour_data is None:
                self._is_colour_data = data.shape[-1]==3

            plottable_data = put_data_in_grid(data, clims = clims, cmap = self._cmap, is_color_data = self._is_colour_data) \
                if not (self._is_colour_data and data.ndim==3 or data.ndim==2) else \
                data_to_image(data, clims = clims, cmap = self._cmap)

        if self._plot is None:
            self._plot = plt.imshow(plottable_data, interpolation = self._interpolation, aspect = self._aspect, cmap = self._cmap)
            if not self._show_axes:
                self._plot.axes.tick_params(labelbottom = 'off')
                self._plot.axes.get_yaxis().set_visible(False)
        else:
            self._plot.set_array(plottable_data)
        if self.show_clims:
            self._plot.axes.set_xlabel('%.2f - %.2f' % clims)


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
        ImagePlot.update(self, buffer_data)


class LinePlot(HistoryFreePlot):

    def __init__(self, y_axis_type = 'lin', x_bounds = (None, None), y_bounds = (None, None)):
        assert y_axis_type == 'lin', 'Changing axis scaling not supported yet'
        self._plots = None
        self._oldvlims = (float('inf'), -float('inf'))
        self._oldhlims = (float('inf'), -float('inf'))
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def _plot_last_data(self, data):
        """

        :param data: Can be:
            A 2-tuple
        :return:
        """

        if isinstance(data, tuple) and len(data)==2:
            x_data, y_data = data
        else:
            x_data = np.arange(len(data))
            y_data = data

        lower, upper = (np.nanmin(y_data) if self.y_bounds[0] is None else self.y_bounds[0], np.nanmax(y_data) if self.y_bounds[1] is None else self.y_bounds[1])
        left, right = (np.nanmin(x_data) if self.x_bounds[0] is None else self.x_bounds[0], np.nanmax(x_data) if self.x_bounds[1] is None else self.x_bounds[1])
        # left, right = (np.nanmin(x_data), np.nanmax(x_data)) if self._yscale is None else self._yscale

        if self._plots is None:
            self._plots = plt.plot(x_data, y_data)
            for p, d in zip(self._plots, y_data[None] if y_data.ndim==1 else y_data.T):
                p.axes.set_xbound(left, right)
                if lower != upper:  # This happens in moving point plots when there's only one point.
                    p.axes.set_ybound(lower, upper)
        else:
            for p, d in zip(self._plots, y_data[None] if y_data.ndim==1 else y_data.T):
                p.set_xdata(x_data)
                p.set_ydata(d)

                if (lower, upper) != self._oldvlims:
                    p.axes.set_ybound(lower, upper)

                if (left, right) != self._oldhlims:
                    p.axes.set_xbound(left, right)

        self._oldvlims = lower, upper
        self._oldhlims = left, right


class MovingPointPlot(LinePlot):

    def __init__(self, buffer_len=100, expanding=True, **kwargs):
        LinePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)
        self.expanding = expanding
        self.x_data = np.arange(-buffer_len, 1)

    def update(self, data):
        if not np.isscalar(data):
            data = data.flatten()

        buffer_data = self._buffer(data)
        if self.expanding:
            buffer_data = buffer_data[np.argmax(~np.any(np.isnan(buffer_data.reshape(buffer_data.shape[0], -1)), axis=1)):]
            x_data = self.x_data[-len(buffer_data):]
        else:
            x_data = self.x_data
        LinePlot.update(self, (x_data, buffer_data))

    def plot(self):
        LinePlot.plot(self)


class Moving2DPointPlot(LinePlot):

    def __init__(self, buffer_len=100, **kwargs):
        LinePlot.__init__(self, **kwargs)
        self._y_buffer = RecordBuffer(buffer_len)
        self._x_buffer = RecordBuffer(buffer_len)
        self.x_data = np.arange(-buffer_len, 1)

    def update(self, (x_data, y_data)):

        x_buffer_data = self._x_buffer(x_data)
        y_buffer_data = self._y_buffer(y_data)

        valid_sample_start = np.argmax(~np.any(np.isnan(y_buffer_data.reshape(y_buffer_data.shape[0], -1)), axis=1))
        # if self.expanding:
        #     y_buffer_data = y_buffer_data[np.argmax(~np.any(np.isnan(y_buffer_data.reshape(y_buffer_data.shape[0], -1)), axis=1)):]
        #     x_data = self.x_data[-len(y_buffer_data):]
        # else:
        #     x_data = self.x_data
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
        if not isinstance(string, basestring):
            string = str(string)
        history = self._buffer(string)
        self._full_text = '\n'.join(history)

    def plot(self):
        if self._text_plot is None:
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_axis_off()

            # self._text_plot = ax.text(0.05, 0.05, self._full_text, horizontalalignment=self.horizontal_alignment, verticalalignment=self.vertical_alignment, size = self.size)
            self._text_plot = ax.text(self._x_offset, self._y_offset, self._full_text, horizontalalignment=self.horizontal_alignment, verticalalignment=self.vertical_alignment, size = self.size)
        else:
            self._text_plot.set_text(self._full_text)


# class BarPlot(HistoryFreePlot):
#
#     def _plot_last_data(self, data):
#         pass


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

    if isinstance(data, basestring):
        return TextPlot()

    if isinstance(data, list) and all(isinstance(d, np.ndarray) and d.ndim in (2, 3) for d in data):
        return ImagePlot()

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

    if isinstance(data, basestring):
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
