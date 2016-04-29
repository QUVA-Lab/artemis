import shutil
from fileman.local_dir import get_local_path

import os
from fileman.saving_plots import always_save_figures, get_saved_figure_locs, get_local_figures_dir, \
    clear_saved_figure_locs
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'peter'


def test_save_figures():
    
    clear_saved_figure_locs()
    test_dir = os.path.join(get_local_figures_dir(), 'testing')

    try:  # Remove dir if not already removed
        shutil.rmtree(test_dir)
    except OSError:
        pass

    always_save_figures(subdir = 'testing', block = False, name = 'test_fig')

    plt.plot(np.random.randn(100, 3))
    plt.show()
    figures = get_saved_figure_locs()
    assert len(figures) == 1 and os.path.exists(get_local_path(figures[0])) and figures[0].endswith('testing/test_fig.pdf')

    try:  # Always good to clean up after yourself.
        shutil.rmtree(test_dir)
    except OSError:
        pass


if __name__ == '__main__':
    test_save_figures()
