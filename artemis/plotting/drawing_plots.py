__author__ = 'peter'
from matplotlib import pyplot as plt


_has_drawn = set()


def redraw_figure(fig = None):
    """
    Redraw a matplotlib figure.
    :param fig: A matplotlib figure.
    """

    if fig is None:
        fig = plt.gcf()

    # Matplotlib is not made for speed by default, but it seems that minor hacks
    # can speed up the rendering a LOT:
    # See https://www.google.nl/search?q=speeding+up+matplotlib&gws_rd=cr&ei=DsD0V9-eGs6ba_jAtaAF
    # There is still potential for more speedup.

    # Slow way:  ~6.6 FPS in demo_dbplot (as measured on linux box)
    # plt.draw()
    # plt.pause(0.00001)

    # Faster way:  ~ 10.5 FPS
    # fig.canvas.draw()
    # plt.show(block=False)

    #  Superfast way ~22.3 FPS  # But crashes when using camera with opencv!s
    if fig in _has_drawn:
        plt.draw()
        _has_drawn.add(fig)
    else:
        fig.canvas.flush_events()
    plt.show(block=False)
