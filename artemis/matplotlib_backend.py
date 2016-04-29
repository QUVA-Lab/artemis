from abc import ABCMeta, abstractmethod
# from general.should_be_builtins import bad_value
from plotting.data_conversion import put_data_in_grid, RecordBuffer, scale_data_to_8_bit, data_to_image
import numpy as np
__author__ = 'peter'




def set_default_figure_size(width, height):
    """
    :param width: Width (in inches, for some reason)
    :param height: Height (also in inches.  One inch is about 2.54cm)
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height


class IPlot(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        pass


class ImagePlot(object):

    def __init__(self, interpolation = 'nearest', show_axes = False, clims = None, aspect = 'auto', cmap = 'gray'):
        self._plot = None
        self._interpolation = interpolation
        self._show_axes = show_axes
        self._clims = clims
        self._aspect = aspect
        self._cmap = cmap

    def update(self, data):

        if data.ndim == 1:
            data = data[None]

        clims = ((np.nanmin(data), np.nanmax(data)) if data.size != 0 else (0, 1)) if self._clims is None else self._clims

        plottable_data = put_data_in_grid(data, clims = clims, cmap = self._cmap) \
            if not (data.ndim == 2 or data.ndim == 3 and data.shape[2] == 3) else \
            data_to_image(data, clims = clims, cmap = self._cmap)

        if self._plot is None:
            self._plot = imshow(plottable_data, interpolation = self._interpolation, aspect = self._aspect, cmap = self._cmap)
            if not self._show_axes:
                # self._plot.axes.get_xaxis().set_visible(False)
                self._plot.axes.tick_params(labelbottom = 'off')
                self._plot.axes.get_yaxis().set_visible(False)
            # colorbar()

        else:
            self._plot.set_array(plottable_data)
        self._plot.axes.set_xlabel('%.2f - %.2f' % clims)
            # self._plot.axes.get_caxis


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


class LinePlot(object):

    def __init__(self, yscale = None):
        self._plots = None
        self._yscale = yscale
        self._oldlims = (float('inf'), -float('inf'))

    def update(self, data):

        lower, upper = (np.nanmin(data), np.nanmax(data)) if self._yscale is None else self._yscale

        if self._plots is None:
            self._plots = plot(np.arange(-data.shape[0]+1, 1), data)
            for p, d in zip(self._plots, data[None] if data.ndim==1 else data.T):
                p.axes.set_xbound(-len(d), 0)
                if lower != upper:  # This happens in moving point plots when there's only one point.
                    p.axes.set_ybound(lower, upper)
        else:
            for p, d in zip(self._plots, data[None] if data.ndim==1 else data.T):
                p.set_ydata(d)
                if lower!=self._oldlims[0] or upper!=self._oldlims[1]:
                    p.axes.set_ybound(lower, upper)

        self._oldlims = lower, upper


class MovingPointPlot(LinePlot):

    def __init__(self, buffer_len=100, **kwargs):
        LinePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)

    def update(self, data):
        if not np.isscalar(data):
            data = data.flatten()

        buffer_data = self._buffer(data)
        LinePlot.update(self, buffer_data)


class TextPlot(IPlot):

    def __init__(self, max_history = 8):
        self._buffer = RecordBuffer(buffer_len = max_history, initial_value='')
        self._max_history = 10
        self._text_plot = None

    def update(self, string):
        if not isinstance(string, basestring):
            string = str(string)
        history = self._buffer(string)
        full_text = '\n'.join(history)
        if self._text_plot is None:
            ax = gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._text_plot = ax.text(0.05, 0.05, full_text)
        else:
            self._text_plot.set_text(full_text)


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
        if self._plot_type == 'bar':
            if self._plot is None:
                self._plot = bar(self._lefts, heights, width = self._widths)
            else:
                for rect, h in zip(self._plot, heights):
                    rect.set_height(h)
        elif self._plot_type == 'line':
            if self._plot is None:
                self._plot = plot(self._edges[:-1], heights)
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
