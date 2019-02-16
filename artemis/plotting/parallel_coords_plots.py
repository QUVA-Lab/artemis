
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from artemis.general.should_be_builtins import izip_equal, bad_value
#
#
# def parallel_coords_plot(field_names, values, color_field = None):
#     """
#     Create a Parallel coordinates plot.
#     Code lifted and modified from http://benalexkeen.com/parallel-coordinates-in-matplotlib/
#
#     :param field_names: A list of (n_fields) field names
#     :param values: A (n_fields, n_samples) array of values.
#     :return:
#     """
#
#     n_fields, n_samples = values.shape
#     df = {name: row for name, row in izip_equal(field_names, values)}
#
#     from matplotlib import ticker
#
#     assert len(field_names)==len(values), 'The number of field names must equal the number of rows in values.'
#
#     # field_names = ['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration']
#     x = [i for i, _ in enumerate(field_names)]
#     # colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']
#
#     # create dict of categories: colours
#     # colours = {df['mpg'].cat.categories[i]: colours[i] for i, _ in enumerate(df['mpg'].cat.categories)}
#
#     # Create (X-1) sublots along x axis
#     fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))
#
#     # Get min, max and range for each column
#     # Normalize the data for each column
#     min_max_range = {}
#     for col in field_names:
#         min_max_range[col] = [np.min(df[col]), np.max(df[col]), np.ptp(df[col])]
#         # df[col] = np.true_divide(df[col] - np.min(df[col]), np.ptp(df[col]))
#         values = (values-np.min(values, axis=1, keepdims=True)) / (np.max(values, axis=1, keepdims=True)-np.min(values, axis=1, keepdims=True))
#
#     # Plot each row
#     for i, ax in enumerate(axes):
#         for idx in range(n_samples):
#
#             # ax.plot(df[])
#
#             # mpg_category = df.loc[idx, 'mpg']
#
#             # ax.plot(x, df.loc[idx, field_names], colours[mpg_category])
#             ax.plot(x, values[:,  idx])
#         ax.set_xlim([x[i], x[i+1]])
#
#     # Set the tick positions and labels on y axis for each plot
#     # Tick positions based on normalised data
#     # Tick labels are based on original data
#     def set_ticks_for_axis(dim, ax, ticks):
#         min_val, max_val, val_range = min_max_range[field_names[dim]]
#         step = val_range / float(ticks-1)
#         tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
#         norm_min = df[field_names[dim]].min()
#         norm_range = np.ptp(df[field_names[dim]])
#         norm_step = norm_range / float(ticks-1)
#         ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
#         ax.yaxis.set_ticks(ticks)
#         ax.set_yticklabels(tick_labels)
#
#     for dim, ax in enumerate(axes):
#         ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
#         set_ticks_for_axis(dim, ax, ticks=6)
#         ax.set_xticklabels([field_names[dim]])
#
#
#     # Move the final axis' ticks to the right-hand side
#     ax = plt.twinx(axes[-1])
#     dim = len(axes)
#     ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
#     set_ticks_for_axis(dim, ax, ticks=6)
#     ax.set_xticklabels([field_names[-2], field_names[-1]])
#
#
#     # Remove space between subplots
#     plt.subplots_adjust(wspace=0)

    # Add legend to plot
    # plt.legend(
    #     [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df['mpg'].cat.categories],
    #     df['mpg'].cat.categories,
    #     bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
#
#
#     # plt.title("Values of car attributes by MPG category")
#
    # plt.show()
from artemis.plotting.pyplot_plus import axhlines

def draw_norm_y_axis(x_position, lims, scale='lin', axis_thickness=2, n_intermediates=3, tickwidth=0.1, axiscolor='k'):
    """
    Draw a y-axis in a Parallel Coordinates plot
    """
    assert scale=='lin', 'For now'
    lower, upper = lims
    line = plt.axvline(x=x_position, ymin=0, ymax=1, linewidth=axis_thickness, color=axiscolor)
    y_axisticks = np.linspace(0, 1, n_intermediates+2)
    y_labels = ['{:.2g}'.format(y*(upper-lower)+lower) for y in y_axisticks]
    h_ticklabels = [plt.text(x=x_position+tickwidth/2., y=y, s=ylab, color='k', bbox=dict(boxstyle="square", fc=(1., 1., 1., 0.5), ec=(0, 0, 0, 0.))) for y, ylab in izip_equal(y_axisticks, y_labels)]
    h_ticks = axhlines(ys = y_axisticks, lims=(x_position-tickwidth/2., x_position+tickwidth/2.), linewidth=axis_thickness, color=axiscolor, zorder=4)
    return line, h_ticks, h_ticklabels


def parallel_coords_plot(field_names, values, scales = {}, ax=None):
    """
    Create a Parallel coordinates plot.

    :param field_names: A list of (n_fields) field names
    :param values: A (n_fields, n_samples) array of values.
    :return: A list of handles to the plot objectss
    """

    assert set(scales.keys()).issubset(field_names), 'All scales must be in field names.'
    assert len(field_names) == len(values)
    if ax is None:
        ax = plt.gca()
    v_min, v_max = np.min(values, axis=1, keepdims=True), np.max(values, axis=1, keepdims=True)

    norm_lines = (values-v_min) / (v_max-v_min)

    cmap = matplotlib.cm.get_cmap('Spectral')
    hs = [plt.plot(line, color=cmap(line[-1]))[0] for i, line in enumerate(norm_lines.T)]

    for i, f in enumerate(field_names):
        draw_norm_y_axis(x_position=i, lims=(v_min[i, 0], v_max[i, 0]), scale = 'lin' if f not in scales else scales[f])

    ax.set_xticks(range(len(field_names)))
    ax.set_xticklabels(field_names)

    ax.tick_params(axis='y', labelleft='off')
    ax.set_yticks([])
    # ax.set_yticklabels([])
    ax.set_xlim(0, len(field_names)-1)

    return hs


def plot_hyperparameter_search_parallel_coords(field_names, x_iters, func_vals, show_iter_first = True, show_score_last = True, iter_name='iter', score_name='score'):
    """
    Visualize the result of a hyperparameter search using a Parallel Coordinates plot
    :param field_names: A (n_hyperparameters) list of names of the hyperparameters
    :param x_iters: A (n_steps, n_hyperparameters) list of hyperparameter values
    :param func_vals: A (n_hyperparameters) list of scores computed for each value
    :param show_iter_first: Insert "iter" (the interation index in the search) as a first column to the plot
    :param show_score_last: Insert "score" as a last column to the plot
    :param iter_name: Name of the "iter" field
    :param score_name: Name of the "score" field.
    :return: A list of plot handels
    """
    field_names = ([iter_name] if show_iter_first else []) + list(field_names) + ([score_name] if show_score_last else [])
    lines = [([i] if show_iter_first else []) + list(params) + ([val] if show_score_last else []) for i, (params, val) in enumerate(izip_equal(x_iters, func_vals))]
    return parallel_coords_plot(field_names=field_names, values=np.array(lines).T)
