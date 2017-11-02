import logging

from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import memoize, bad_value
import numpy as np

from six.moves import xrange

__author__ = 'peter'

# Note - this module used to be called math, but it somehow results in a numpy import error
# due to some kind of name conflict with another module called math.

sigm = lambda x: 1/(1+np.exp(-x))


def cummean(x, axis = None):
    """
    Cumulative mean along axis
    :param x: An array
    :param axis: The axis
    :return: An array of the same shape
    """
    if axis is None:
        assert isinstance(x, list) or x.ndim == 1, 'You must specify axis for a multi-dimensional array'
        axis = 0
    elif axis < 0:
        axis = x.ndim+axis
    x = np.array(x)
    normalizer = np.arange(1, x.shape[axis]+1).astype(float)[(slice(None), )+(None, )*(x.ndim-axis-1)]
    return np.cumsum(x, axis)/normalizer


def cumvar(x, axis = None, sample = True):
    """
    :return: Cumulative variance along axis
    """
    if axis is None:
        assert isinstance(x, list) or x.ndim == 1, 'You must specify axis for a multi-dimensional array'
        axis = 0
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    ex_2 = cummean(x, axis=axis)**2
    e_x2 = cummean(x**2, axis=axis)
    var = e_x2-ex_2
    if sample and x.shape[axis] > 1:
        var *= x.shape[axis]/float(x.shape[axis]-1)
    return var


@memoize
def binary_permutations(n_bits):
    """
    Given some number of bits, return a shape (2**n_bits, n_bits) boolean array containing every permoutation
    of those bits as a row.
    :param n_bits: An integer number of bits
    :return: A shape (2**n_bits, n_bits) boolean array containing every permoutation
        of those bits as a row.
    """
    return np.right_shift(np.arange(2**n_bits)[:, None], np.arange(n_bits-1, -1, -1)[None, :]) & 1


def softmax(x, axis = None):
    """
    The softmax function takes an ndarray, and returns an ndarray of the same size,
    with the softmax function applied along the given axis.  It should always be the
    case that np.allclose(np.sum(softmax(x, axis), axis)==1)
    """
    if axis is None:
        assert x.ndim==1, "You need to specify the axis for softmax if your data is more thn 1-D"
        axis = 0
    x = x - np.max(x, axis=axis, keepdims=True)  # For numerical stability - has no effect mathematically
    expx = np.exp(x)
    return expx/np.sum(expx, axis=axis, keepdims=True)


def expected_sigm_of_norm(mean, std, method = 'probit'):
    """
    Approximate the expected value of the sigmoid of a normal distribution.

    Thanks go to this guy:
    http://math.stackexchange.com/questions/207861/expected-value-of-applying-the-sigmoid-function-to-a-normal-distribution

    :param mean: Mean of the normal distribution
    :param std: Standard Deviation of the normal distribution
    :return: An approximation to Expectation(sigm(N(mu, sigma**2)))
    """
    from scipy.stats import norm
    if method == 'maclauren-2':
        eu = np.exp(-mean)
        approx_exp = 1/(eu+1) + 0.5*(eu-1)*eu/((eu+1)**3) * std**2
        return np.minimum(np.maximum(approx_exp, 0), 1)

    elif method == 'maclauren-3':
        eu = np.exp(-mean)
        approx_exp = 1/(eu+1) + \
            0.5*(eu-1)*eu/((eu+1)**3) * std**2 + \
            (eu**3-11*eu**2+57*eu-1)/((8*(eu+1))**5) * std**4
        return np.minimum(np.maximum(approx_exp, 0), 1)
    elif method == 'probit':
        return norm.cdf(mean/np.sqrt(2.892 + std**2))
    else:
        raise Exception('Method "%s" not known' % method)


l1_error = lambda x1, x2: np.mean(np.abs(x1-x2), axis = -1)


def normalize(x, axis=None, degree = 2, avoid_nans = False):
    """
    Normalize array x.
    :param x: An array
    :param axis: Which axis to normalize along
    :param degree: Degree of normalization (1 for L1-norm, 2 for L2-norm, etc)
    :param avoid_nans: If, along an axis, there is a norm of zero, then normalize this to a uniform vector (instead of nans).
    :return: An array the same shape as x, normalized along the given axis
    """
    assert degree in (1, 2), "Give me a reason and I'll give you more degrees"

    if degree == 1:
        z = np.sum(np.abs(x.astype(np.float)), axis = axis, keepdims=True)
    else:
        z = np.sum(x**degree, axis = axis, keepdims=True)**(1./degree)
    normed = x/z
    if avoid_nans:
        uniform_vector_value = (1./x.shape[axis])**(1./degree)
        normed[np.isnan(normed)] = uniform_vector_value
    return normed


def mode(x, axis = None, keepdims = False):
    from scipy.stats import mode as sp_mode
    mode_x, _ = sp_mode(x, axis = axis)
    if not keepdims:
        mode_x = np.take(mode_x, 0, axis = axis)
    return mode_x


def cummode(x, weights = None, axis = 1):
    """
    Cumulative mode along an axis.  Ties give priority to the first value to achieve the
    given count.
    """
    import weave  # ONLY WORKS IN PYTHON 2.X !!!

    assert x.ndim == 2 and axis == 1, 'Only implemented for a special case!'
    all_values, element_ids = np.unique(x, return_inverse=True)
    n_unique = len(all_values)
    element_ids = element_ids.reshape(x.shape)
    result = np.zeros(x.shape, dtype = int)
    weighted = weights is not None
    if weighted:
        assert x.shape == weights.shape
    counts = np.zeros(n_unique, dtype = float if weighted else int)
    code = """
    bool weighted = %s;
    int n_samples = Nelement_ids[0];
    int n_events = Nelement_ids[1];
    for (int i=0; i<n_samples; i++){
        float maxcount = 0;
        int maxel = -1;

        for (int k=0; k<n_unique; k++)
            counts[k] = 0;

        for (int j=0; j<n_events; j++){
            int ix = i*n_events+j;
            int k = element_ids[ix];
            counts[k] += weighted ? weights[ix] : 1;
            if (counts[k] > maxcount){
                maxcount = counts[k];
                maxel = k;
            }
            result[ix]=maxel;
        }
    }
    """ % ('true' if weighted else 'false')
    weave.inline(code, ['element_ids', 'result', 'n_unique', 'counts', 'weights'], compiler = 'gcc')
    mode_values = all_values[result]
    return mode_values


def recent_moving_average(x, axis = 0):
    """
    Fast computation of recent moving average, where

        frac = 1/sqrt(t)
        a[t] = (1-frac)*a[t-1] + frac*x[t]
    """

    import weave  # ONLY WORKS IN PYTHON 2.X !!!
    if x.ndim!=2:
        y = recent_moving_average(x.reshape(x.shape[0], x.size//x.shape[0]), axis=0)
        return y.reshape(x.shape)

    assert x.ndim == 2 and axis == 0, 'Only implemented for a special case!'
    result = np.zeros(x.shape)
    code = """
    int n_samples = Nx[0];
    int n_dim = Nx[1];
    for (int i=0; i<n_dim; i++)
        result[i] = x[i];
    int ix=n_dim;
    for (int t=1; t<n_samples; t++){
        float frac = 1./sqrt(t+1);
        for (int i=0; i<n_dim; i++){
            result[ix] = (1-frac)*result[ix-n_dim] + frac*x[ix];
        }
        ix += 1;
    }
    """
    weave.inline(code, ['x', 'result'], compiler = 'gcc')
    return result


def angle_between(a, b, axis=None, in_degrees = False):
    """
    Return the angle between two vectors a and b, in radians.  Raise an exception if one is a zero vector
    :param a: A vector
    :param b: A vector the same size as a
    :return: The angle between these vectors, in radians.

    Credit to Pace: http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    cos_dist = cosine_distance(a, b, axis=axis)
    angle = np.arccos(cos_dist)
    if in_degrees:
        angle = angle * 180/np.pi
    return angle


def cosine_distance(a, b, axis=None):
    """
    Return the cosine distance between two vectors a and b.  Raise an exception if one is a zero vector
    :param a: An array
    :param b: Another array of the same shape
    :return: The cosine distance between a and b, reduced along the given axis.

    Credit to Pace: http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b
    if not a.dtype==float:
        a=a.astype(float)
    if not b.dtype==float:
        b=b.astype(float)
    if axis is None:
        a = a.ravel()
        b = b.ravel()
        axis = 0
    assert a.shape[-1]==b.shape[-1]
    cosine_distance = (a*b).sum(axis=axis)/np.sqrt((a**2).sum(axis=axis) * (b**2).sum(axis=axis))
    # For numerical resons, we might get values outside [-1, 1] here, so we truncate:
    cosine_distance = np.minimum(cosine_distance, 1)
    cosine_distance = np.maximum(cosine_distance, -1)
    return cosine_distance


def degrees_between(a, b):
    return angle_between(a, b, in_degrees=True)


def magnitude_ratio(a, b):
    """
    Return the ratio of the L2-magnitudes of each vector
    :param a: A vector
    :param b: Another vector of the same size
    :return: The ratio |a_mag
    """
    assert a.ndim == 1 and a.shape==b.shape
    a_mag = np.sqrt(np.sum(a**2))
    b_mag = np.sqrt(np.sum(b**2))
    d_magnitude = a_mag/b_mag
    return d_magnitude


def is_parallel(a, b, angular_tolerance = 1e-7):
    """
    Test whether two vectors are parallel to within a given tolerance.
    Throws an exception for zero-vectors.

    :param a: A vector
    :param b: A vector the same size as a
    :param angular_tolerance: The tolerance, in radians.
    :return: A boolean, indicating that the vectors are parallel to within the specified tolerance.
    """
    assert 0 <= angular_tolerance <= 2*np.pi, "It doesn't make sense to specity an angular tolerance outside of [0, 2*pi].  Why are you doing this?"
    angle = angle_between(a, b)
    return angle < angular_tolerance


def align_curves(xs, ys, n_bins='median', xrange = ('min', 'max'), spacing = 'lin'):
    """
    Given multiple curves with different x-coordinates, interpolate so that each has the same x points.

    :param xs: A length-N list of sorted vectors containing the x-coordinates of each curve
    :param ys: A length-N list of vectors containing the corresponding y-coordinates
    :param n_bins: Number of points to make along new x-axis.  'median' to use the median number of points in the curves.
    :param xrange: 2-tuple indicating range of x-axis to span.  'min' indicates "minimum across curves", As with 'max'.
    :param spacing: Either 'lin' or 'log', depenting on whether you want your interpolation points spaced linearly or
        logarithmically.
    :return: (new_xs, new_ys).
        new_xs is a (n_bins, ) curve indicating the new x-locations.
        new_ys is a (N, n_bins)
    """
    assert spacing in ('lin', 'log')
    assert len(xs)==len(ys)
    assert all(len(x)==len(y) for x, y in zip(xs, ys))

    start, stop = xrange
    if start == 'min':
        start = np.min([x[0] for x in xs if len(x)>0])
    if stop == 'max':
        stop = np.max([x[-1] for x in xs if len(x)>0])
    if n_bins == 'median':
        n_bins = int(np.round(np.median([len(x) for x in xs])))

    new_x = np.linspace(start, stop, n_bins) if spacing=='lin' else np.logspace(np.log10(start), np.log10(stop), n_bins)

    new_ys = np.zeros((len(xs), n_bins)) + np.nan

    for x, y, ny in zip(xs, ys, new_ys):
        if len(x)>=2:
            ny[:] = np.interp(x=new_x, xp=x, fp=y, left=np.nan, right=np.nan)

    return new_x, new_ys


def sqrtspace(a, b, n_points):
    """
    :return: Distribute n_points quadratically from point a to point b, inclusive
    """
    return np.linspace(0, 1, n_points)**2*(b-a)+a


def fixed_diff(x, axis=-1, initial_value = 0.):
    """
    Modification of numpy.diff where the first element is compared to the initial value.
    The resulting array has the same shape as x.

    Note that this inverts np.cumsum so that np.cumsum(fixed_diff(x)) == x    (except for numerical errors)

    :param x: An array
    :param axis: Axis along which to diff
    :param initial_value: The initial value agains which to diff the first element along the axis.
    :return: An array of the same shape, representing the difference in x along the axis.
    """

    x = np.array(x, copy=False)

    if axis<0:
        axis = x.ndim+axis

    result = np.empty_like(x)
    initial_indices = (slice(None), )*axis
    result[initial_indices+(slice(1, None), )] = np.diff(x, axis=axis)
    if initial_value == 'first':
        result[initial_indices+(0, )] = 0
    else:
        result[initial_indices+(0, )] = x[initial_indices+(0, )]-initial_value
    return result


def decaying_cumsum(x, memory, axis=-1):

    if axis<0:
        axis = x.ndim+axis
    assert 0 <= memory < 1
    result = np.empty_like(x)
    leading_indices = (slice(None), )*axis
    one_minus_mem = 1-memory
    result[leading_indices+(0, )] = one_minus_mem*x[leading_indices+(0, )]
    for i in xrange(1, x.shape[axis]):
        result[leading_indices+(i, )] = memory*result[leading_indices+(i-1, )] + one_minus_mem*x[leading_indices+(i, )]
    if np.max(np.abs(result)>1e9):
        print('sdfdsf: {}'.format(np.max(np.abs(x))))

    return result


def point_space(start, stop, n_points, spacing):
    if spacing=='lin':
        values = np.linspace(start, stop, n_points)
    elif spacing=='sqrt':
        values = sqrtspace(start, stop, n_points)
    elif spacing=='log':
        values = np.logspace(np.log10(start), np.log10(stop), n_points)
    else:
        raise NotImplementedError(spacing)
    return values


def geosum(rate, t_end, t_start=0):
    """
    Geometric sum of a series from t_start to t_end

    e.g. geosum(0.5, t_end=4, t_start=2) = 0.5**2 + 0.5**3 + 0.5**4 = 0.375
    """
    return np.where(rate==1, np.array(t_end-t_start+1, copy=False).astype(float), np.array(rate**(t_end+1)-rate**t_start)/(rate-1))


def selective_sum(x, ixs):
    """
    :param x: An nd array
    :param ixs: A tuple of length x.ndim indexing each of the dimensions.
    :return: A scalar sum of all array elements selected by any of the dimensions.

    This is best explained by example:

        a = np.array([[ 0,  1,  2,  3],
        ...           [ 4,  5,  6,  7],
        ...           [ 8,  9, 10, 11],
        ...           [12, 13, 14, 15]])

    If we want to add all elements rows 1, and 2, as well as the column, then we go:
        s = selective_sum(a, [(1,3), 2])

    And we can verify that:
        s == 4+5+6+7 + 12+13+14+15 + 2+10 == 88

    If you don't want to select coluns
    """
    assert x.ndim==len(ixs), 'The dimension of x must match the length of ixs'
    al = (slice(None), )
    selection_mask = np.zeros(x.shape, dtype='bool')
    for i, ix in enumerate(ixs):
        selection_mask[al*i+(ix, )+al*(x.ndim-i-1)] = True
    return (x*selection_mask).sum()

    # Note, we'd like to do this more efficiently, but it gets a little complicated.
    # (we have to add the individual indexes, but subtract the double-counted regions, and then subtract the triple-counted
    # regions, and so on....)
    # return sum(x[al*i+(ix, )+al*(x.ndim-i-1)].sum() for i, ix in enumerate(ixs)) - x[ixs].sum()


def conv_fanout(input_len, kernel_len, conv_mode):
    """
    Note: this is horrific and must be simplified.
    :param input_len:
    :param kernel_len:
    :param conv_mode:
    :return:
    """

    if conv_mode=='full':
        return kernel_len*np.ones(input_len)
    else:
        if conv_mode=='half':
            conv_mode='same'
        left_pad = kernel_len // 2 if conv_mode == 'same' else 0 if conv_mode == 'valid' else conv_mode if isinstance(conv_mode, int) else bad_value(conv_mode)
        right_pad = (kernel_len-1) // 2 if conv_mode == 'same' else 0 if conv_mode == 'valid' else conv_mode if isinstance(conv_mode, int) else bad_value(conv_mode)
        full_range = np.arange(left_pad + input_len + right_pad)
        max_fanout = np.minimum(kernel_len, np.maximum(input_len-kernel_len+1+2*left_pad, 1))
        fanout_over_full_range = np.minimum(max_fanout, np.minimum(full_range+1, full_range[::-1]+1))
        fanout = fanout_over_full_range[left_pad:len(full_range)-right_pad]
        return fanout


def conv2_fanout_map(input_shape, kernel_shape, conv_mode):
    size_y, size_x = input_shape
    k_size_y, k_size_x = kernel_shape
    y_fanout = conv_fanout(input_len = size_y, kernel_len=k_size_y, conv_mode=conv_mode)
    x_fanout = conv_fanout(input_len = size_x, kernel_len=k_size_x, conv_mode=conv_mode)
    fanout_map = y_fanout[:, None] * x_fanout
    return fanout_map


def levenshtein_distance(s1, s2):
    """
    The Levenshtein Distance (a type of edit distance) between strings

    Thank you to Salvador Dali here: https://stackoverflow.com/a/32558749/851699
    :param s1: A string
    :param s2: Another String
    :return: An integer distance.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def onehotvector(ix, length):
    """
    Create a one-hot vector of length length with element ix 1 (and the rest 0)
    :param ix: The element to be "hot".  Ot a vector of elememts.
    :param length: The total length of the vector.
    :return: If ix is scalar, a single vector.
        If ix is a vector, a (len(ix), length) array where each row is a one-hot vector for an element of ix.
    """
    if isinstance(ix, int):
        v = np.zeros(length)
        v[ix] = 1
    else:
        v = np.zeros((len(ix), length))
        v[np.arange(len(ix)), ix] = 1
    return v


def proportional_random_assignment(length, split, rng):
    """
    Generate an integer array of the given length, with elements randomly assigned to 0...len(split), with
    frequency of elements with value i proporational to split[i].

    This is useful for splitting training/test sets.  e.g.

        n_samples = 1000
        x = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples)
        subsets = proportional_random_assignment(n_samples, split=0.7, rng=1234)
        x_train = x[subsets==0]
        y_train = y[subsets==0]
        x_test = x[subsets==1]
        y_test = y[subsets==1]

    :param length: The length of the output array
    :param split: Either a list of ratios to assign to each group (must add to <1), or a single float in (0, 1),
        which will indicate that we split into 2 groups.
    :param rng: A random number generator or seed.
    :return: An integer array.
    """
    rng = get_rng(rng)
    if isinstance(split, float):
        split = [split]
    assert 0<=np.sum(split)<=1, "The sum of elements in split: {} must be in [0, 1].  Got {}".format(split, np.sum(split))
    arr = np.zeros(length, dtype=int)
    cut_points = np.concatenate([np.round(np.cumsum(split)*length).astype(int), [length]])
    scrambled_indices = rng.permutation(length)
    for i, (c_start, c_end) in enumerate(zip(cut_points[:-1], cut_points[1:])):
        arr[scrambled_indices[c_start:c_end]] = i+1  # Note we skip zero since arrays already inited to 0
    return arr


def argmaxnd(x):
    ix = np.argmax(x.flatten())
    return np.unravel_index(ix, dims=x.shape)


def clip_to_sum(vec, total):
    new_vec = np.array(vec)  # Yes this is horribly inefficient but I do not care.
    current_total = np.sum(vec)
    while current_total > total:
        i = np.argmax(new_vec)
        new_vec[i] -= 1
        current_total -= 1
    return new_vec
