
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from artemis.general.mymath import cosine_distance
from artemis.general.should_be_builtins import izip_equal
from artemis.plotting.pyplot_plus import axhlines


def draw_norm_y_axis(x_position, lims, scale='lin', axis_thickness=2, n_intermediates=3, tickwidth=0.1, axiscolor='k', ticklabel_format='{:.3g}', tick_round_grid=40):
    """
    Draw a y-axis in a Parallel Coordinates plot

    :param x_position: Position in x to draw the axis
    :param lims: The (min, max) limit of the y-axis
    :param scale: Not implemented for now, just leave at 'lin'.  (Todo: implement 'log')
    :param axis_thickness: Thickness of the axis line
    :param n_intermediates: Number of ticks to put in between ends of axis
    :param tickwidth: Width of tick lines
    :param axiscolor: Color of axis
    :param ticklabel_format: Format for string ticklabel numbers
    :param tick_round_grid: Round ticks to a grid with this number of points, or None to not do this.  (Causes nicer axis labels)
    :return: The handels for the (<vertical line>, <ticks>, <tick labels>)
    """
    assert scale=='lin', 'For now'
    lower, upper = lims
    vertical_line_handel = plt.axvline(x=x_position, ymin=0, ymax=1, linewidth=axis_thickness, color=axiscolor)
    y_axisticks = np.linspace(0, 1, n_intermediates+2)
    y_trueticks = y_axisticks * (upper - lower) + lower
    if tick_round_grid is not None:
        # spacing = (upper - lower)/tick_round_grid
        spacing = 10**np.round(np.log10((upper-lower)/tick_round_grid))
        y_trueticks = np.round(y_trueticks/spacing)*spacing
        y_axisticks = (y_trueticks - y_trueticks[0])/(y_trueticks[-1] - y_trueticks[0])
    y_labels = [ticklabel_format.format(y) for y in y_trueticks]
    tick_label_handels = [plt.text(x=x_position+tickwidth/2., y=y, s=ylab, color='k', bbox=dict(boxstyle="square", fc=(1., 1., 1., 0.5), ec=(0, 0, 0, 0.))) for y, ylab in izip_equal(y_axisticks, y_labels)]
    tick_handels = axhlines(ys = y_axisticks, lims=(x_position-tickwidth/2., x_position+tickwidth/2.), linewidth=axis_thickness, color=axiscolor, zorder=4)
    return vertical_line_handel, tick_handels, tick_label_handels


def parallel_coords_plot(field_names, values, special_formats = {}, scales = {}, color_index=-1, ax=None, alpha='auto', cmap='Spectral', **plot_kwargs):
    """
    Create a Parallel coordinates plot.  These plots are useful for visualizing high-dimensional data.

    :param field_names: A list of (n_fields) field names
    :param values: A (n_samples, n_fields) array of values.
    :param Dict[int, Dict] special_formats: Optionally a dictionary mapping from sample index to line format.  This can be used to highlight certain lines.
    :param Dict[str, str] scales: (currently not implemented) Identifies the scale ('lin' or 'log') for each field name
    :param color_index: Which column of values to use to colour-code the lines.  Defaults to the last column.
    :param ax: The plot axis (if None, use current axis (gca))
    :param alpha: The alpha (opaqueness) value to use.  If 'auto', this function automatically lowers alpha in regions of dense overlap.
    :param plot_kwargs: Other kwargs to pass to line plots (these will be overridden on a per-plot basis by special_formats, alpha)
    :return: A list of handles to the plot objects
    """
    values = np.array(values, copy=False)
    assert set(scales.keys()).issubset(field_names), 'All scales must be in field names.'
    assert len(field_names) == values.shape[1]
    if ax is None:
        ax = plt.gca()
    v_min, v_max = np.min(values, axis=0), np.max(values, axis=0)

    norm_lines = (values-v_min) / (v_max-v_min)

    cmap = matplotlib.cm.get_cmap(cmap)
    formats = {i: plot_kwargs.copy() for i in range(len(norm_lines))}
    for i, line in enumerate(norm_lines):  # Color lines according to score
        formats[i]['color']=cmap(1-line[color_index])
    if alpha=='auto':
        mean_param = np.mean(norm_lines, axis=0)
        for i, line in enumerate(norm_lines):
            sameness = max(0, cosine_distance(mean_param, line))  # (0 to 1 where 1 means same as the mean)
            alpha = sameness * (1./np.sqrt(values.shape[0])) + (1-sameness)*1.
            formats[i]['alpha'] = alpha
    else:
        for i in range(len(norm_lines)):
            formats[i]['alpha'] = alpha
    for i, form in special_formats.items():  # Add special formats
        formats[i].update(form)
    plot_kwargs.update(dict(alpha=alpha))

    hs = [plt.plot(line, **formats[i])[0] for i, line in enumerate(norm_lines)]

    for i, f in enumerate(field_names):
        draw_norm_y_axis(x_position=i, lims=(v_min[i], v_max[i]), scale = 'lin' if f not in scales else scales[f])

    ax.set_xticks(range(len(field_names)))
    ax.set_xticklabels(field_names)

    ax.tick_params(axis='y', labelleft='off')
    ax.set_yticks([])
    ax.set_xlim(0, len(field_names)-1)

    return hs


def plot_hyperparameter_search_parallel_coords(field_names, param_sequence, func_vals, final_params = None, show_iter_first = True, show_score_last = True, iter_name='iter', score_name='score'):
    """
    Visualize the result of a hyperparameter search using a Parallel Coordinates plot
    :param field_names: A (n_hyperparameters) list of names of the hyperparameters
    :param param_sequence: A (n_steps, n_hyperparameters) list of hyperparameter values
    :param func_vals: A (n_hyperparameters) list of scores computed for each value
    :param final_params: Optionally, provide the final chosen set of hyperparameters.  These will be plotted as a thick
        black dotted line.
    :param show_iter_first: Insert "iter" (the interation index in the search) as a first column to the plot
    :param show_score_last: Insert "score" as a last column to the plot
    :param iter_name: Name of the "iter" field
    :param score_name: Name of the "score" field.
    :return: A list of plot handels
    """
    field_names = ([iter_name] if show_iter_first else []) + list(field_names) + ([score_name] if show_score_last else [])
    lines = [([i+1] if show_iter_first else []) + list(params) + ([val] if show_score_last else []) for i, (params, val) in enumerate(izip_equal(param_sequence, func_vals))]

    if final_params is not None:  # This adds a black dotted line over the final set of hyperparameters
        ix = next(i for i, v in enumerate(param_sequence) if np.array_equal(v, final_params))
        lines.append(([ix] if show_iter_first else [])+list(final_params)+([func_vals[ix] if show_score_last else []]))
        special_formats = {len(lines)-1: dict(linewidth=2, color='k', linestyle='--', alpha=1)}
    else:
        special_formats = {}
    return parallel_coords_plot(field_names=field_names, values=lines, special_formats=special_formats)
