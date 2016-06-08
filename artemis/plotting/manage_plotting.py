from matplotlib import pyplot as plt
__author__ = 'peter'

"""
This allows you to attach callbacks to pyplots show/draw functions.  You may, for instance, want to save a figure
whenever plt.show is called.  After importing this module, you will be able to do this with the following:

    from artemis.plotting.manage_plotting import ShowContext

    def my_show_fcn(fig):
        ... save figure or whatever you want to do

    with ShowContext(my_show_fcn, clear_others=False):
        ... everything called in here will result in your function being called before show.  Set clear_others to false
        to remove the existing action of plt.show() (which displays the figure).

Alternatively, if you wish to globally set a callback and never change it, you can go:

    from artemis.plotting.manage_plotting import set_show_callback

    set_show_callback(my_show_fcn)


You can also do the same for drawing figures (showing without hanging on the figure) using DrawContext, set_draw_callback.
"""

_ORIGINAL_SHOW_CALLBACK = plt.show
_ORIGINAL_PLT_DRAW = plt.draw


def draw(fig=None):
    _ORIGINAL_PLT_DRAW()
    plt.pause(0.00001)  # Required to display, at least on some backends.


_ORIGINAL_DRAW_CALLBACK = draw

class FigureCallBackManager(object):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, fig=None):
        if fig is None:
            fig = plt.gcf()
        for cb in self.callbacks:
            cb(fig)

    def insert_callback(self, cb, idx = 0):
        old_callbacks = list(self.callbacks)
        assert cb not in self.callbacks, "You've included the same callback object already"
        self.callbacks.insert(idx, cb)
        return old_callbacks

    def set_callback(self, cb):
        """
        :param cb: Set a callback and remove all others.
        """
        old_callbacks = list(self.callbacks)
        self.callbacks = [cb]
        return old_callbacks

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def remove_callback(self, callbacks):
        self.callbacks.remove(callbacks)

    def clear_callbacks(self):
        self.callbacks = []

show_callback = FigureCallBackManager([_ORIGINAL_SHOW_CALLBACK])
plt.show=show_callback

draw_callback = FigureCallBackManager([_ORIGINAL_DRAW_CALLBACK])
plt.draw=draw_callback
draw = draw_callback


class ShowContext(object):

    def __init__(self, callback, clear_others = False):
        self.callback = callback
        self.clear_others = clear_others

    def __enter__(self):
        self.old = show_callback.set_callback(self.callback) if self.clear_others else show_callback.insert_callback(self.callback)

    def __exit__(self, exc_type, exc_val, exc_tb):
        show_callback.set_callbacks(self.old)


class DrawContext(object):

    def __init__(self, callback, clear_others = False):
        self.callback = callback
        self.clear_others = clear_others

    def __enter__(self):
        self.old = draw_callback.set_callback(self.callback) if self.clear_others else draw_callback.insert_callback(self.callback)

    def __exit__(self, exc_type, exc_val, exc_tb):
        draw_callback.set_callbacks(self.old)


def set_show_callback(callback):
    return show_callback.set_callback(callback)


def set_draw_callback(callback):
    return draw_callback.set_callback(callback)


def insert_show_callback(callback):
    return show_callback.insert_callback(callback)


def insert_draw_callback(callback):
    return draw_callback.insert_callback(callback)
