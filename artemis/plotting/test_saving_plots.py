from shutil import rmtree
from artemis.fileman.local_dir import get_local_path
from artemis.plotting.drawing_plots import redraw_figure
from artemis.plotting.saving_plots import save_figure, show_saved_figure, save_figures_on_close
import os
import matplotlib.pyplot as plt
import numpy as np
__author__ = 'peter'


def test_save_and_show_figure():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_local_path('tests/test_fig.pdf')
    save_figure(fig, path = path)
    show_saved_figure(path)


def test_save_and_show_figure_2():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_local_path('tests/test_fig')
    path = save_figure(fig, path = path)
    show_saved_figure(path)


def test_save_and_show_figure_3():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_local_path('tests/test_fig.with.strangely.formatted.ending')
    path = save_figure(fig, path = path, ext='pdf')
    show_saved_figure(path)


def test_save_figures_on_close():

    path = get_local_path('tests/save_on_close_test')
    if os.path.exists(path):
        rmtree(path)

    with save_figures_on_close(path=path, only_new_figs=True):
        plt.figure()
        plt.plot(np.sin(np.linspace(0, 10, 40)))
        plt.figure()
        plt.plot(np.cos(np.linspace(0, 10, 40)))
    assert os.path.exists(os.path.join(path, 'fig-0.pdf'))
    assert os.path.exists(os.path.join(path, 'fig-1.pdf'))
    assert not os.path.exists(os.path.join(path, 'fig-2.pdf'))
    rmtree(path)


if __name__ == '__main__':
    test_save_and_show_figure()
    test_save_figures_on_close()
