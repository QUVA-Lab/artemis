import numpy as np
from matplotlib import pyplot as plt
from artemis.plotting.expanding_subplots import set_named_subplot


def test_expanding_subplots(block=False):

    set_named_subplot('agfdsfgdg')
    plt.plot(np.sin(np.linspace(0, 10, 100)))

    set_named_subplot('dsxfdsgf')
    plt.imshow(np.random.randn(10, 10))

    set_named_subplot('agfdsfgdg')
    plt.plot(np.cos(np.linspace(0, 10, 100)))

    plt.show(block = block)


def test_closing_fig(block=False):
    """
    Test that if we close the figure, we recognise that fact, and that we therefore need to recreate the subplot afterwards.
    """
    fig = plt.figure()

    set_named_subplot('fdsfsd')
    plt.plot(np.sin(np.linspace(0, 10, 100)))

    plt.close(fig)
    set_named_subplot('fdsfsd')
    plt.plot(np.cos(np.linspace(0, 10, 100)))

    plt.show(block = block)



if __name__ == '__main__':
    test_closing_fig()
    test_expanding_subplots()
