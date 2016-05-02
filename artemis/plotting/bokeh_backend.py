
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.data_conversion import put_data_in_grid, data_to_image, RecordBuffer, UnlimitedRecordBuffer
import numpy as np

from bokeh.models import GridPlot
from bokeh.document import Document
from bokeh.client import push_session
from bokeh.plotting import Figure, figure
from bokeh.palettes import Spectral6



_SESSION_COUNTER = 0
_ALL_PLOTS = {}
_CURRENT_SESSION = None
_CURRENT_PLOT = None
_URL = "default"
_SUBPLOT_DICT = {}
_GRIDPLOT = None

def set_url(url):
    global _URL
    _URL = url


def get_name():
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    return "Figure_" + str(_SESSION_COUNTER)


class FakeFigure():
    def __init__(self, session):
        self.session = session

    def clf(self):
        global _CURRENT_SESSION
        _CURRENT_SESSION.document.clear()
        global _CURRENT_PLOT
        _CURRENT_PLOT = None
        global _ALL_PLOTS
        _ALL_PLOTS = {}
        global _SESSION_COUNTER
        _SESSION_COUNTER = 0
        global _SUBPLOT_DICT
        _SUBPLOT_DICT = {}
        global _GRIDPLOT
        _GRIDPLOT = None


def figure(*args,**kwargs):
    return FakeFigure(_session(*args, **kwargs))

def _session(name=None):
    if name == None:
        name = get_name()
    doc = Document()
    session = push_session(document=doc, session_id=name, url=_URL)
    print ("Session id: %s" %(name))
    session.show()
    global _CURRENT_SESSION
    _CURRENT_SESSION = session
    return session

def _get_or_make_session():
    return figure() if _CURRENT_SESSION is None else _CURRENT_SESSION

def _plot(model = None, **kwargs):
    session = _get_or_make_session()
    # Build Gridplot:
    num = _SUBPLOT_DICT["num"]
    rows = _SUBPLOT_DICT["rows"]
    cols = _SUBPLOT_DICT["cols"]
    global _GRIDPLOT
    if num == 1:
        children = [[None for _ in xrange(_SUBPLOT_DICT["cols"])] for _ in xrange(_SUBPLOT_DICT["rows"])]
    else:
        children = _GRIDPLOT.children
    newplot = model(**kwargs)
    global _CURRENT_PLOT
    _CURRENT_PLOT = newplot
    global _ALL_PLOTS
    _ALL_PLOTS[newplot._id] = newplot
    row_coordinate = int(np.ceil(float(num) / cols) - 1)
    col_coordinate = num % cols -1 if num % cols != 0 else cols -1
    children[row_coordinate][col_coordinate] = newplot
    _GRIDPLOT = GridPlot()
    _GRIDPLOT.children = children
    _CURRENT_SESSION.document.clear()
    _GRIDPLOT._detach_document()
    print _GRIDPLOT
    _CURRENT_SESSION.document.add_root(_GRIDPLOT)
    # global _CURRENT_PLOT
    # _CURRENT_PLOT = newplot
    return newplot



    # plot = model(**kwargs)
    # global _CURRENT_SESSION
    # _CURRENT_SESSION.document.add_root(plot)

def _get_or_make_plot(model, **kwargs):
    session = _get_or_make_session()
    return _plot(model, **kwargs) if _CURRENT_PLOT is None else _CURRENT_PLOT


def make_plot(model, **kwargs):
    return _plot(model, **kwargs)

def subplot(rows, cols, num, **kwargs):

    global _SUBPLOT_DICT
    _SUBPLOT_DICT = {"rows":rows, "cols":cols, "num":num, "kwargs":kwargs}


def draw():
    pass

def show():
    pass

def isinteractive():
    pass

def interactive(bool):
    pass

def gca():
    return _CURRENT_PLOT

def title(s, *args, **kwargs):
    gca().title = s

def plot_data_format(*args, **kwargs):
    '''
    This function should take data as variable as possible and put it into a multi_line plot

    :param args:
    :param kwargs:
    :return:
    '''

    if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
        x_data, y_data = args[:2]
    else:
        x_data = np.arange(len(args[0]))
        y_data = args[0]
    # print (x_data)
    # print (y_data)

    if x_data.ndim == 1:
        x_data = np.expand_dims(x_data,0)
    if y_data.ndim == 1:
        y_data = np.expand_dims(y_data,0)

    if y_data.shape != x_data.shape:
        if y_data.T.shape == x_data.shape:
            x_data = x_data.T
        elif y_data.shape[0] % x_data.shape[0] == 0:
                x_data = np.repeat(x_data,y_data.shape[0],0)
        else:
            print ("x-axis data and y-axis data not correct for multi-line plot")
    assert y_data.shape == x_data.shape

    # Remove all nan values because updating the data dict can't deal with nan values.
    # y_data_list = [(y_data[i][~np.isnan(y_data[i])]).tolist() for i in xrange(y_data.shape[0])]
    # x_data_list = [(x_data[i][~np.isnan(y_data[i])]).tolist() for i in xrange(y_data.shape[0])]
    y_data = np.nan_to_num(y_data)

    return x_data.tolist(), y_data.tolist()
    # return x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]

    # return gca().multi_line(xs = x_data.tolist(), ys = y_data.tolist(), color = Spectral6[:y_data.shape[0]], line_width = 2 )


def _glyph_already_on_document(glyph):
    global _ALL_PLOTS
    existing_renderers  = []
    for p in _ALL_PLOTS.values():
        existing_renderers = existing_renderers + p.renderers
    return glyph._id in [r._id for r in existing_renderers]

class LinePlot(object):
    def __init__(self, yscale = None, **kwargs):
        self._plots = None
        self.kw = kwargs

    def update(self, data):
        # print(data)
        x_data, y_data = plot_data_format(np.arange(-data.shape[0]+1, 1), data.T)

        if self._plots is None:
            ax = make_plot(Figure, **self.kw)
            self._plots = gca().multi_line(xs = x_data, ys = y_data, color = Spectral6[:len(y_data)], line_width = 2 )
            # self._plots = plot(np.arange(-data.shape[0]+1, 1), data.T, **self.kw)
        else:
            _get_or_make_plot(Figure, **self.kw)
            if not _glyph_already_on_document(self._plots):
                _CURRENT_PLOT.renderers.append(self._plots)
            self._plots.data_source.data["xs"] = x_data
            self._plots.data_source.data["ys"] = y_data
            print self._plots.data_source.data

class HistoryPlot(LinePlot):
    def __init__(self, **kwargs):
        LinePlot.__init__(self,**kwargs)
        self._buffer = UnlimitedRecordBuffer()

    def update(self,data):
        if not np.isscalar(data):
            data = data.flatten()
        buffer_data = self._buffer(data)
        LinePlot.update(self,buffer_data)



class MovingPointPlot(LinePlot):

    def __init__(self, buffer_len=100, **kwargs):
        LinePlot.__init__(self, **kwargs)
        self._buffer = RecordBuffer(buffer_len)

    def update(self, data):
        if not np.isscalar(data):
            data = data.flatten()
        buffer_data = self._buffer(data)
        LinePlot.update(self, buffer_data)



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
