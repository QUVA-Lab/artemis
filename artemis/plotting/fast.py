try:
    from scipy import weave
except ImportError:
    print("Cannot Import scipy weave.  That's ok for now, you won't be able to use the fastplot function.")

__author__ = 'peter'
import numpy as np
import matplotlib.pyplot as plt
"""
Functions to speed up pylab plotting, which can sometimes be unnecessairily slow.
"""


def fastplot(line_data, xscale = 'linear', yscale = 'linear', resolution = 2000, min_points = 20000):
    """
    Fast plot for those times when you have a lot of data points and it becomes too slow for pylab to handle.
    The plot produced here should look the same (unless you zoom) but display much faster.

    :param line_data: A vector of data
    :param xscale: {'linear', 'log'}
    :param yscale: {'linear', 'log'}
    :param resolution: The number intervals to bin points into
    :param min_points: The minimum number of points required to bother with this approach.
    :return: A plot handle
    """

    assert line_data.ndim == 1

    if len(line_data) < min_points:
        h= plt.plot(line_data)
    else:
        if xscale == 'linear':
            intervals = np.linspace(0, len(line_data), resolution)
        elif xscale == 'log':
            intervals = np.logspace(0, np.log10(len(line_data)), resolution)
        else:
            raise Exception("Can't yet deal with scale %s" % xscale)

        extreme_indices = find_interval_extremes(line_data, edges = intervals[1:])
        h=plt.plot(extreme_indices, line_data[extreme_indices])

    plt.gca().set_xscale(xscale)
    plt.gca().set_yscale(yscale)

    return h


def fastloglog(line_data, **kwargs):
    return fastplot(line_data, xscale='log', yscale = 'symlog', **kwargs)


def find_interval_extremes(array, edges):
    """
    Find the indeces of extreme points within each interval, and on the outsides of the two end edges.
    :param array: A vector
    :param edges: A vector of edges defining the intervals.  It's assumed that -Inf, Inf form the true outer edges.
    :return: A vector of ints indicating the indeces of extreme points.  If a distinct min and max extreme are found
        within every interval, this vector will have length 2*(len(edges)+1).  Otherwise, it will be shorter.
    """

    indices = np.zeros(len(array), dtype = int)-1
    how_many = np.array([0])
    code = """
    float min = INFINITY;
    float max = -INFINITY;
    int argmin;
    int argmax;
    bool foundpoint = false;
    int in_counter = 0;
    int out_counter = 0;
    int edge_counter = 0;
    while(in_counter<Narray[0]){
        float next_edge;
        if (edge_counter == Nedges[0])
            next_edge = INFINITY;
        else
            next_edge = edges[edge_counter];

        if (array[in_counter] < min){
            min = array[in_counter];
            argmin = in_counter;
            foundpoint = true;
        }
        if (array[in_counter] > max){
            max = array[in_counter];
            argmax = in_counter;
            foundpoint = true;
        }
        in_counter++;
        if (in_counter > next_edge){
            if (foundpoint){
                if (argmin < argmax){
                    indices[out_counter] = argmin;
                    indices[out_counter+1] = argmax;
                    out_counter+=2;
                }
                else if (argmax < argmin){
                    indices[out_counter] = argmax;
                    indices[out_counter+1] = argmin;
                    out_counter+=2;
                }
                else {
                    indices[out_counter] = argmax;
                    out_counter++;
                }
                min = INFINITY;
                max = -INFINITY;
                foundpoint = false;
            }
            edge_counter++;
        }
    }
    how_many[0] = out_counter;
    """
    weave.inline(code, ['array', 'edges', 'indices', 'how_many'], compiler = 'gcc')
    result = indices[:how_many[0]]
    return result
