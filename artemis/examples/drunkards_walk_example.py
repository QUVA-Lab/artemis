import numpy as np
from artemis.experiments import ExperimentFunction
from matplotlib import pyplot as plt
from six.moves import xrange


def display_drunkards_walk(drunkards):
    plt.plot(drunkards[:, :, 0], drunkards[:, :, 1])
    plt.grid()
    plt.xlabel('$\Delta$ Longitude (arcseconds)')
    plt.ylabel('$\Delta$ Latitude (arcseconds)')
    plt.show()


def compare_drunkards_walk(dict_of_drunkards):
    plot_handles = []
    for i, (exp_name, drunkards) in enumerate(dict_of_drunkards.items()):
        plot_handles.append(plt.plot(drunkards[:, :, 0], drunkards[:, :, 1], color='C{}'.format(i)))
    plt.grid()
    plt.xlabel('$\Delta$ Longitude (arcseconds)')
    plt.ylabel('$\Delta$ Latitude (arcseconds)')
    plt.legend([p[0] for p in plot_handles], dict_of_drunkards.keys())
    plt.show()


@ExperimentFunction(display_function=display_drunkards_walk, comparison_function=compare_drunkards_walk)
def demo_drunkards_walk(n_steps=500, n_drunkards=5, homing_instinct = 0, n_dim=2, seed=1234):
    """
    Release several drunkards in a field to randomly stumble around.  Record their progress.
    """
    rng = np.random.RandomState(seed)
    drunkards = np.zeros((n_steps+1, n_drunkards, n_dim))
    for t in xrange(1, n_steps+1):
        drunkards[t] = drunkards[t-1]*(1-homing_instinct) + rng.randn(n_drunkards, n_dim)
        if t%100==0:
            print('Status at step {}: Mean: {}, STD: {}'.format(t, drunkards[t].mean(), drunkards[t].std()))
    return drunkards


demo_drunkards_walk.add_variant(homing_instinct = 0.01)
demo_drunkards_walk.add_variant(homing_instinct = 0.1)


if __name__ == '__main__':
    demo_drunkards_walk.browse()
    # Try
    #   run all
    #   compare all
    #   display 1
