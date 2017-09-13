from contextlib import contextmanager
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

    def new_show(*args, **kwargs):

        if state == 'hang':
            _ORIGINAL_SHOW_CALLBACK(*args, **kwargs)
        elif state == 'draw':
            plt.draw()
            plt.pause(0.00001)
        elif state == False:
            pass

    with ShowContext(new_show, clear_others=True):
        yield


class ShowContext(object):

    def __init__(self, callback, clear_others = False):

        def show_wrapper(*args, **kwargs):
            if 'block' in kwargs and kwargs['block'] is False:
                _ORIGINAL_SHOW_CALLBACK(*args, **kwargs)
            else:
                callback(*args, **kwargs)
                if not clear_others:
                    self.old(*args, **kwargs)

        self.show_wrapper = show_wrapper

    def __enter__(self):
        self.old = plt.show
        plt.show = self.show_wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self.old


class DrawContext(object):
    # TODO: Integrate this with redraw_figure(fig)
    # It mainly just useful for saving updating plots in experiments.

    def __init__(self, callback, clear_others = False):
        self.callback = callback
        self.clear_others = clear_others

    def __enter__(self):
        self.old = plt.draw
        plt.draw = self.callback if self.clear_others else self._draw_with_others

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.draw = self.old

    def _draw_with_others(self, *args, **kwargs):
        self.old(*args, **kwargs)
        self.callback(*args, **kwargs)


def set_show_callback(callback):
    """
    Perminantly changes the action of plt.show().  WARNING: This function can lead to some really confusing bugs.  Only
    use if you really know what you're doing.
    :param callback:
    """
    plt.show = callback


def set_draw_callback(callback):
    """
    Perminantly changes the action of plt.draw().  WARNING: This function can lead to some really confusing bugs.  Only
    use if you really know what you're doing.
    :param callback:
    """
    plt.draw = callback


@contextmanager
def delay_show():
    """
    Delay any calls to plt.show() until the end of this block
    :return:
    """
    has_shown = [False]
    def show_sub(*args, **kwargs):
        has_shown[0]=True
    with ShowContext(show_sub, clear_others=True):
        yield
    if has_shown[0]:
        plt.show()
