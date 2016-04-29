from general.should_be_builtins import memoize, bad_value
import numpy as np
from scipy import weave
from scipy.stats import norm, mode as sp_mode
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
        z = np.sum(np.abs(x), axis = axis, keepdims=True)
    else:
        z = np.sum(x**degree, axis = axis, keepdims=True)**(1./degree)
    normed = x/z
    if avoid_nans:
        uniform_vector_value = (1./x.shape[axis])**(1./degree)
        normed[np.isnan(normed)] = uniform_vector_value
    return normed


def mode(x, axis = None, keepdims = False):
    mode_x, _ = sp_mode(x, axis = axis)
    if not keepdims:
        mode_x = np.take(mode_x, 0, axis = axis)
    return mode_x


def cummode(x, weights = None, axis = 1):
    """
    Cumulative mode along an axis.  Ties give priority to the first value to achieve the
    given count.
    """

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


def angle_between(a, b, in_degrees = False):
    """
    Return the angle between two vectors a and b, in radians.  Raise an exception if one is a zero vector
    :param a: A vector
    :param b: A vector the same size as a
    :return: The angle between these vectors, in radians.

    Credit to Pace: http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b
    assert a.ndim == 1 and a.shape==b.shape
    arccos_input = np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
    arccos_input = 1.0 if arccos_input > 1.0 else arccos_input
    arccos_input = -1.0 if arccos_input < -1.0 else arccos_input
    angle = np.arccos(arccos_input)
    if in_degrees:
        angle = angle * 180/np.pi
    return angle


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
