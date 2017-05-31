from artemis.config import get_artemis_config_value
from matplotlib import pyplot as plt
__author__ = 'peter'

_plotting_mode = get_artemis_config_value(section='plotting', option='mode')

if _plotting_mode == 'safe':

    def redraw_figure(fig=None):
        plt.draw()
        plt.pause(0.00001)

elif _plotting_mode == 'fast':

    def redraw_figure(fig=None):
        if fig is None:
            fig = plt.gcf()
        fig.canvas.flush_events()
        plt.show(block=False)
        plt.show(block=False)

else:
    raise Exception("Unknown plotting mode: {}".format(_plotting_mode))
