__author__ = 'peter'
from matplotlib import pyplot as plt


_has_drawn = set()


def redraw_figure(fig):
    """
    Redraw a matplotlib figure.
    :param fig: A matplotlib figure.
    """

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

    #  Superfast way ~22.3 FPS
    if fig in _has_drawn:
        fig.canvas.draw()
        _has_drawn.add(fig)
    else:
        fig.canvas.flush_events()
    plt.show(block=False)
