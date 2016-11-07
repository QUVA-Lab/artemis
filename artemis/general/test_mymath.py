from artemis.general.mymath import softmax, cummean, cumvar, sigm, expected_sigm_of_norm, mode, cummode, normalize, is_parallel, \
    align_curves, angle_between
import numpy as np
__author__ = 'peter'


def test_softmax():
    x = np.random.randn(3, 4, 5)

    s = softmax(x, axis=1)
    assert s.shape==(3, 4, 5) and (s>0).all() and (s<1).all() and np.allclose(np.sum(s, axis=1), 1)


def test_cummean():

    arr = np.random.randn(3, 4)
    cum_arr = cummean(arr, axis = 1)
    assert np.allclose(cum_arr[:, 0], arr[:, 0])
    assert np.allclose(cum_arr[:, 1], np.mean(arr[:, :2], axis = 1))
    assert np.allclose(cum_arr[:, 2], np.mean(arr[:, :3], axis = 1))


def test_cumvar():

    arr = np.random.randn(3, 4)
    cum_arr = cumvar(arr, axis = 1, sample = False)
    assert np.allclose(cum_arr[:, 0], 0)
    assert np.allclose(cum_arr[:, 1], np.var(arr[:, :2], axis = 1))
    assert np.allclose(cum_arr[:, 2], np.var(arr[:, :3], axis = 1))


def test_exp_sig_of_norm():

    mean = 1
    std = 0.8
    n_points = 1000
    seed = 1234

    inputs = np.random.RandomState(seed).normal(mean, std, size = n_points)
    vals = sigm(inputs)
    sample_mean = np.mean(vals)

    for method in ('maclauren-2', 'maclauren-3', 'probit'):
        approx_true_mean = expected_sigm_of_norm(mean, std, method = method)
        approx_sample_mean = expected_sigm_of_norm(np.mean(inputs), np.std(inputs), method = method)
        true_error = np.abs(approx_true_mean-sample_mean)/sample_mean
        sample_error = np.abs(approx_sample_mean-sample_mean)/sample_mean
        print 'Error for %s: %.4f True, %.4f Sample.' % (method, true_error, sample_error)
        assert true_error < 0.02, 'Method %s did pretty bad' % (method, )


def test_mode():

    arr = np.random.RandomState(0).randint(low=0, high=2, size=(3, 5, 7))

    m0 = mode(arr, axis = 0)
    assert m0.shape == (5, 7)
    assert np.all(np.sum(m0[None, :, :] == arr, axis = 0) > np.sum(m0[None, :, :] != arr, axis = 0))

    m1 = mode(arr, axis = 1)
    assert m1.shape == (3, 7)
    assert np.all(np.sum(m1[:, None, :] == arr, axis = 1) > np.sum(m1[:, None, :] != arr, axis = 1))

    m2 = mode(arr, axis = 2)
    assert m2.shape == (3, 5)


def test_cummode():

    arr = np.random.RandomState(0).randint(low=0, high=3, size=(5, 7))

    m = cummode(arr, axis = 1)  # (n_samples, n_events)
    assert m.shape == arr.shape

    uniques = np.unique(arr)

    for j in xrange(arr.shape[1]):
        n_elements_of_mode_class = np.sum(arr[:, :j+1] == m[:, j][:, None], axis = 1)  # (n_samples, )
        for k, u in enumerate(uniques):
            n_elements_of_this_class = np.sum(arr[:, :j+1] == u, axis = 1)  # (n_samples, )
            assert np.all(n_elements_of_mode_class >= n_elements_of_this_class)


def test_cummode_weighted():

    arr = np.random.RandomState(0).randint(low=0, high=3, size=(5, 7))
    w = np.random.rand(5, 7)

    m = cummode(arr, weights=w, axis = 1)  # (n_samples, n_events)
    assert m.shape == arr.shape

    uniques = np.unique(arr)

    for j in xrange(arr.shape[1]):
        bool_ixs_of_mode_class = arr[:, :j+1] == m[:, j][:, None]  # (n_samples, j+1)
        weights_of_mode_class = np.sum(w[:, :j+1]*bool_ixs_of_mode_class, axis = 1)  # (n_samples, )

        for k, u in enumerate(uniques):
            bool_ixs_of_this_class = arr[:, :j+1] == u  # (n_samples, j+1)
            weights_of_this_class = np.sum(w[:, :j+1]*bool_ixs_of_this_class, axis = 1)  # (n_samples, )
            assert np.all(weights_of_mode_class >= weights_of_this_class)


def test_normalize():

    # L1 - positive values
    arr = np.random.rand(5, 4)
    norm_arr = normalize(arr, degree=1, axis = 1)
    assert np.allclose(norm_arr.sum(axis=1), 1)

    # L1 - with negative values
    arr = np.random.randn(5, 4)
    norm_arr = normalize(arr, degree=1, axis = 1)
    assert np.allclose(np.abs(norm_arr).sum(axis=1), 1)

    # L2
    arr = np.random.randn(5, 4)
    norm_arr = normalize(arr, degree=2, axis = 1)
    assert np.allclose(np.sqrt((norm_arr**2).sum(axis=1)), 1)

    # L1 - zero row with nan handling
    arr = np.random.rand(5, 4)
    arr[2, :] = 0
    norm_arr = normalize(arr, degree=1, axis = 1)
    assert np.all(np.isnan(norm_arr[2, :]))

    # L1 - zero row with nan handling
    arr = np.random.rand(5, 4)
    arr[2, :] = 0
    norm_arr = normalize(arr, degree=1, axis = 1, avoid_nans=True)
    assert np.allclose(np.abs(norm_arr).sum(axis=1), 1)
    assert np.allclose(norm_arr[2, :], 1./arr.shape[1])

    # L2 - zero row with nan handling
    arr = np.random.rand(5, 4)
    arr[2, :] = 0
    norm_arr = normalize(arr, degree=2, axis = 1, avoid_nans=True)
    assert np.allclose(np.sqrt((norm_arr**2).sum(axis=1)), 1)
    assert np.allclose(norm_arr[2, :], np.sqrt(1./arr.shape[1]))


def test_is_parallel():

    assert is_parallel([1, 2], [2, 4])
    assert not is_parallel([1, 2], [2, 5])
    assert is_parallel([1, 2], [2, 5], angular_tolerance=0.5)
    assert not is_parallel([1, 2], [-2, -4])


def test_align_curves():

    n_curves = 30

    n_points = [np.random.randint(20) for _ in xrange(n_curves)]

    xs = [np.sort(np.random.rand(n)) for n in n_points]
    ys = [np.random.randn(n) for n in n_points]

    new_xs, new_ys = align_curves(xs=xs, ys=ys, n_bins=25, spacing='lin')
    assert new_xs.shape == (25, )
    assert new_ys.shape == (n_curves, 25)

    new_xs, new_ys = align_curves(xs=xs, ys=ys, n_bins=25, spacing='log')
    assert new_xs.shape == (25, )
    assert new_ys.shape == (n_curves, 25)


def test_angle_between():

    a = np.array([[0, 1], [1, 1], [1, 0]])
    b = np.array([[1, 0], [1, 0], [1, 0]])
    angles = angle_between(a, b, in_degrees=True, axis=-1)
    assert np.allclose(angles, [90, 45, 0])
    assert np.allclose(angle_between([2, 1], [1, 0], in_degrees=True), np.arctan(1/2.)*180/np.pi)


if __name__ == '__main__':
    test_angle_between()
    test_align_curves()
    test_is_parallel()
    test_normalize()
    test_cummode_weighted()
    test_cummode()
    test_mode()
    test_exp_sig_of_norm()
    test_cumvar()
    test_cummean()
    test_softmax()
