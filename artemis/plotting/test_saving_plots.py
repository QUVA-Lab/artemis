from artemis.fileman.local_dir import get_artemis_data_path
from artemis.plotting.saving_plots import save_figure, show_saved_figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

__author__ = 'peter'


@pytest.mark.skipif(True, reason='Opens pdf with browser, which causes test server to hang.  Not a terribly important test anyway.')
def test_save_and_show_figure():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_artemis_data_path('tests/test_fig.pdf')
    save_figure(fig, path = path)
    show_saved_figure(path)


@pytest.mark.skipif(True, reason='Opens pdf with browser, which causes test server to hang.  Not a terribly important test anyway.')
def test_save_and_show_figure_2():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_artemis_data_path('tests/test_fig')
    path = save_figure(fig, path = path)
    show_saved_figure(path)


@pytest.mark.skipif(True, reason='Opens pdf with browser, which causes test server to hang.  Not a terribly important test anyway.')
def test_save_and_show_figure_3():

    fig = plt.figure()
    plt.imshow(np.random.randn(10, 10))
    plt.title('Test Figure')
    path = get_artemis_data_path('tests/test_fig.with.strangely.formatted.ending')
    path = save_figure(fig, path = path, ext='pdf')
    show_saved_figure(path)


if __name__ == '__main__':
    test_save_and_show_figure()
