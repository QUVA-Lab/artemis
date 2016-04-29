__author__ = 'peter'
from matplotlib import pyplot as plt


"""
Matplotlib extras
"""


def axhlines(ys, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axhline(y, **specs) for y in ys]


def axvlines(xs, **specs):
    # Note, more efficient to do single line with nans as breakers but whatever
    return [plt.axvline(x, **specs) for x in xs]
