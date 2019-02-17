from matplotlib import pyplot as plt
import numpy as np


def plot_sample_mean_and_var(*x_and_ys, var_rep ='std', fill_alpha = 0.25, **plot_kwargs):
    """
    Given a collection of signals, plot their mean and fill a range around the mean.  Example:
            x = np.arange(-5, 5)
            ys = np.random.randn(20, len(x_data)) + x ** 2 - 2
            plot_sample_mean_and_var(x, ys, var_rep='std')
    :param x_and_ys: You can provide either x and the y-signals or just the y-signals
    :param var_rep: How to represent the variance.  Options are:
        'std': Standard Deviation
        'sterr': Standard Error of the Mean
        'lim': Min/max
    :param fill_alpha:
    :param plot_kwargs:
    :return:
    """
    if len(x_and_ys)==2:
        x, ys = x_and_ys
    else:
        assert len(x_and_ys) == 1, "You must provide unnamed arguments in order (ys) or (x, ys)"
        ys, = x_and_ys
        x = range(len(ys[0]))

    assert var_rep in ('std', 'sterr', 'lim')

    mean = np.mean(ys, axis=0)

    if var_rep == 'std':
        std = np.std(ys, axis=0)
        lower, upper = mean-std, mean+std
    elif var_rep == 'sterr':
        sterr = np.std(ys, axis=0)/np.sqrt(len(ys))
        lower, upper = mean-sterr, mean+sterr
    elif var_rep == 'lim':
        lower, upper = np.min(ys, axis=0), np.max(ys, axis=0)
    else:
        raise NotImplementedError(var_rep)

    mean_handel, = plt.plot(x, mean, **plot_kwargs)
    fill_handle = plt.fill_between(x, lower, upper, color=mean_handel.get_color(), alpha=fill_alpha)
    return mean_handel, fill_handle


if __name__ == '__main__':
    x_data = np.arange(-5, 5)
    ys1 = np.random.randn(20, len(x_data)) + x_data ** 2 - 2
    plot_sample_mean_and_var(x_data, ys1, var_rep='std')

    ys2 = np.random.randn(20, len(x_data)) + .9 * x_data ** 2 - 2
    plot_sample_mean_and_var(x_data, ys2, var_rep='std')

    ys3 = np.random.randn(20, len(x_data)) + .7 * x_data ** 2 - 2
    plot_sample_mean_and_var(x_data, ys3, var_rep='std')
    plt.show()
