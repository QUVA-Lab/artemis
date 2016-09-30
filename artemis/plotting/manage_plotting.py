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

# _WHAT_TO_DO_ON_SHOW = 'hang'


# def redraw_figure(fig=None):
#     plt.draw()
#     _ORIGINAL_SHOW_CALLBACK(block=False)
#     plt.pause(0.0001)


# def show_figure(*args, **kwargs):
#
#     if is_test_mode():
#         redraw_figure()  # Designed to
#     elif _WHAT_TO_DO_ON_SHOW=='hang':
#         _ORIGINAL_SHOW_CALLBACK(*args, **kwargs)
#     elif _WHAT_TO_DO_ON_SHOW=='draw':
#         redraw_figure()
#     elif _WHAT_TO_DO_ON_SHOW is False:
#         pass

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

# show_callback = FigureCallBackManager([show_figure])
# draw_callback = FigureCallBackManager([redraw_figure])
# redraw_figure = draw_callback


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
    #
    #
    # global _WHAT_TO_DO_ON_SHOW
    # old_block_val = _WHAT_TO_DO_ON_SHOW
    # _WHAT_TO_DO_ON_SHOW = state
    # yield
    # _WHAT_TO_DO_ON_SHOW = old_block_val
#
# @contextmanager
# def ShowContext(callback, clear_others = False):
#
#     old_show_fcn = plt.show
#     if clear_others:
#         plt.show = callback
#     else:
#         def show_and_others(*args, **kwargs):
#             old_show_fcn(*args, **kwargs)
#             callback(*args, **kwargs)
#         plt.show = show_and_others
#     yield
#     plt.show = old_show_fcn
#
#
# @contextmanager
# def DrawContext(callback, clear_others = False):
#
#     old_draw_fcn = plt.draw
#     if clear_others:
#         plt.draw = callback
#     else:
#         def draw_and_others():
#             old_draw_fcn()
#             callback()
#         plt.draw = draw_and_others
#     yield
#     plt.draw = old_draw_fcn


class ShowContext(object):

    def __init__(self, callback, clear_others = False):
        self.callback = callback
        self.clear_others = clear_others

    def __enter__(self):
        self.old = plt.show
        plt.show = self.callback if self.clear_others else self._show_with_others

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self.old

    def _show_with_others(self, *args, **kwargs):
        if 'block' in kwargs and kwargs['block'] is False:  # This is treated a special case.  plt.pause() calls plt.show(block=False), which would result in an infinite loop if we didn't do this.
            _ORIGINAL_SHOW_CALLBACK(*args, **kwargs)
        else:
            self.callback(*args, **kwargs)
            self.old(*args, **kwargs)


class DrawContext(object):

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
