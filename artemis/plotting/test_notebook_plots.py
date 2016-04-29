import shutil

import os
from fileman.notebook_plots import always_link_figures
from fileman.saving_plots import get_saved_figure_locs, get_local_figures_dir, clear_saved_figure_locs
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'peter'


def test_link_figures():

    clear_saved_figure_locs()
    test_dir = os.path.join(get_local_figures_dir(), 'testing')

    try:  # Remove dir if not already removed
        shutil.rmtree(test_dir)
    except OSError:
        pass

    always_link_figures(subdir = 'testing', block = False, name = 'test_fig')

    plt.plot(np.random.randn(100, 3))
    plt.show()
    figures = get_saved_figure_locs()
    assert len(figures) == 1 and os.path.exists(figures[0]) and figures[0].endswith('testing/test_fig.pdf')

    try:  # Always good to clean up after yourself.  Your mama isn't going to do it for you.
        shutil.rmtree(test_dir)
    except OSError:
        pass


if __name__ == '__main__':
    test_link_figures()
