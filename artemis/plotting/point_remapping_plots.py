import numpy as np
from matplotlib import pyplot as plt


def get_2d_point_colours(points):
    points_norm = (points - points.min(axis=0, keepdims=True)) / (points.max(axis=0, keepdims=True) - points.min(axis=0, keepdims=True))
    return [(y, x, 1-x) for x, y in points_norm]


def plot_2D_mapping(old_xy_points, new_xy_points, axes = None, old_title = 'x', new_title = 'f(x)'):
    """
    :param old_xy_points: (N,2) array
    :param new_xy_points: (Nx2) array
    """

    colours = get_2d_point_colours(old_xy_points)

    ax = plt.subplot(1, 2, 1) if axes is None else axes[0]
    ax.scatter(old_xy_points[:, 0], old_xy_points[:, 1], c=colours)
    ax.set_title(old_title)

    ax = plt.subplot(1, 2, 2) if axes is None else axes[1]
    ax.scatter(new_xy_points[:, 0], new_xy_points[:, 1], c=colours)
    ax.set_title(new_title)


if __name__ == '__main__':
    n_x = 40
    n_y = 30

    # Generate a grid of points
    old_xy_points = np.array([v.flatten() for v in np.meshgrid(np.linspace(0, 1, n_y), np.linspace(0, 1, n_x))]).T

    # Apply some transformation
    theta = 5*np.pi/6
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_xy_points = np.tanh(old_xy_points @ transform_matrix)

    plot_2D_mapping(old_xy_points, new_xy_points)
