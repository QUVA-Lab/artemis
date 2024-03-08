from typing import Tuple

import numpy as np

from artemis.general.custom_types import Array


def reframe_from_a_to_b(
        xy_in_a: Array['...,2', float],
        reference_xy_in_b: Tuple[float, float],
        reference_xy_in_a: Tuple[float, float] = (0., 0.),
        scale_in_a_of_b: float = 1.
) -> Array['...,2', float]:
    """ Convert a set of (x,y) coordinates in one reference frame to another.

    :param xy_in_a: (N,2) array of (x,y) coordinates in the reference frame A
    :param reference_xy_in_b: (x,y) coordinates of the origin of reference frame B in reference frame A
    :param reference_xy_in_a: (x,y) coordinates of the origin of reference frame A in reference frame A
    :param scale_in_a_of_b: Scale of reference frame B in reference frame A
    :return: (N,2) array of (x,y) coordinates in the reference frame B
    """
    xy_in_a = np.asarray(xy_in_a)
    assert xy_in_a.shape[-1] == 2
    xy_in_b = (xy_in_a - reference_xy_in_a) * scale_in_a_of_b + reference_xy_in_b
    return xy_in_b


def reframe_from_b_to_a(
        xy_in_b: Array['...,2', float],
        reference_xy_in_a: Tuple[float, float],
        reference_xy_in_b: Tuple[float, float] = (0., 0.),
        scale_in_a_of_b: float = 1.
) -> Array['...,2', float]:
    """ Convert a set of (x,y) coordinates in one reference frame to another.

    :param xy_in_b: (N,2) array of (x,y) coordinates in the reference frame B
    :param reference_xy_in_a: (x,y) coordinates of the origin of reference frame A in reference frame B
    :param reference_xy_in_b: (x,y) coordinates of the origin of reference frame B in reference frame B
    :param scale_in_a_of_b: Scale of reference frame B in reference frame A
    :return: (N,2) array of (x,y) coordinates in the reference frame A
    """
    xy_in_b = np.asarray(xy_in_b)
    assert xy_in_b.shape[-1] == 2
    xy_in_a = (xy_in_b - reference_xy_in_b) / scale_in_a_of_b + reference_xy_in_a
    return xy_in_a


def clip_points_to_limit(points: Array['N,2', float], limit_xxyy: Tuple[float, float, float, float]) -> Array['N,2', float]:
    """
    Clip points to a rectangular region.

    :param points: (N,2) array of (x,y) coordinates
    :param limit_xxyy: (x_min, x_max, y_min, y_max)
    :return: (N,2) array of (x,y) coordinates
    """
    points = np.asarray(points)
    assert points.shape[1] == 2
    x_min, x_max, y_min, y_max = limit_xxyy
    return np.clip(points, (x_min, y_min), (x_max, y_max))
