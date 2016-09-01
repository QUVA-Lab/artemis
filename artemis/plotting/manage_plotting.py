from contextlib import contextmanager
from artemis.general.test_mode import is_test_mode
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

_WHAT_TO_DO_ON_SHOW = 'hang'


def redraw_figure(fig=None):
    plt.draw()
    _ORIGINAL_SHOW_CALLBACK(block=False)


def show_figure(*args, **kwargs):

    if is_test_mode():
        redraw_figure()  # Designed to
    elif _WHAT_TO_DO_ON_SHOW=='hang':
        _ORIGINAL_SHOW_CALLBACK(*args, **kwargs)
    elif _WHAT_TO_DO_ON_SHOW=='draw':
        redraw_figure()
    elif _WHAT_TO_DO_ON_SHOW is False:
        pass

class FigureCallBackManager(object):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, fig=None, **kwargs):
        if fig is None:
            fig = plt.gcf()
        for cb in self.callbacks:
            cb(fig, **kwargs)

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

show_callback = FigureCallBackManager([show_figure])
plt.show=show_callback

draw_callback = FigureCallBackManager([redraw_figure])
# plt.draw=draw_callback
redraw_figure = draw_callback


@contextmanager
def WhatToDoOnShow(state):
    """
    Deterine whether or not show blocks.
    *Note: In test mode, this is overridden, and show will never block.
    :param state: One of:
        'hang': Show and hang
        'draw': Show but keep on going
        False: Don't show figures
    :return:
    """
    assert state in ('hang', 'draw', False)
    global _WHAT_TO_DO_ON_SHOW
    old_block_val = _WHAT_TO_DO_ON_SHOW
    _WHAT_TO_DO_ON_SHOW = state
    yield
    _WHAT_TO_DO_ON_SHOW = old_block_val


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
