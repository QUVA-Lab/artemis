import numpy as np
from matplotlib import pyplot as plt
from artemis.plotting.expanding_subplots import select_subplot


def test_expanding_subplots(block=False):

    select_subplot('agfdsfgdg')
    plt.plot(np.sin(np.linspace(0, 10, 100)))

    select_subplot('dsxfdsgf')
    plt.imshow(np.random.randn(10, 10))

    select_subplot('agfdsfgdg')
    plt.plot(np.cos(np.linspace(0, 10, 100)))

    plt.show(block = block)


def test_closing_fig(block=False):
    """
    Test that if we close the figure, we recognise that fact, and that we therefore need to recreate the subplot afterwards.
    """
    fig = plt.figure()

    select_subplot('fdsfsd')
    plt.plot(np.sin(np.linspace(0, 10, 100)))

    plt.close(fig)
    select_subplot('fdsfsd')
    plt.plot(np.cos(np.linspace(0, 10, 100)))

    plt.show(block = block)


def test_positioning_plots(block=False):

    select_subplot(position=(0, 1))
    plt.plot(np.random.randn(10))
    select_subplot(position=(1, 1))
    plt.plot(np.random.randn(10))
    select_subplot(position=(1, 0))
    plt.plot(np.random.randn(10))

    plt.show(block=block)


if __name__ == '__main__':
    test_closing_fig()
    test_expanding_subplots()
    test_positioning_plots()