import numpy as np
from artemis.general.mymath import softmax, sigm

from artemis.ml.datasets.datasets import DataSet, DataCollection


__author__ = 'peter'


def get_synthethic_linear_dataset(noise_level = 0.1, n_input_dims = 20, n_output_dims = 4, n_training_samples = 1000,
        n_test_samples = 200, nonlinearity = None, offset_mag = 0, seed = 8158):
    """
    A Synthethic dataset that can be used for testing generalized linear models.

    :param noise_level:
    :param n_input_dims:
    :param n_output_dims:
    :param n_training_samples:
    :param n_test_samples:
    :param nonlinearity:
    :param seed:
    :return:
    """

    input_singleton = n_input_dims == 0
    if input_singleton:
        n_input_dims = 1

    output_singleton = n_output_dims == 0
    if output_singleton:  # Unfortunately we have to deal with the inconsistencies in numpy's handling of singleton dimensions.
        n_output_dims = 1

    rng = np.random.RandomState(seed)
    w = rng.randn(n_input_dims, n_output_dims) * 1/np.sqrt(n_input_dims)
    input_data = rng.randn(n_training_samples+n_test_samples, n_input_dims)
    target_data = np.dot(input_data, w) + offset_mag * rng.randn(n_output_dims) + noise_level*rng.randn(n_training_samples+n_test_samples, n_output_dims)
    if nonlinearity=='softmax':
        target_data = softmax(target_data, axis=1),
    elif nonlinearity=='sigmoid':
        target_data = sigm(target_data)
    elif nonlinearity=='argmax':
        target_data==np.argmax(target_data, axis=1)
    elif nonlinearity is None:
        target_data = target_data
    else:
        assert callable(nonlinearity), 'Unknown nonlinearity: {}'.format(nonlinearity)
        target_data = nonlinearity(target_data)

    if input_singleton:
        input_data = input_data[:, 0]

    if output_singleton:
        target_data = target_data[:, 0]

    return DataSet(
        training_set = DataCollection(input_data[:n_training_samples], target_data[:n_training_samples]),
        test_set = DataCollection(input_data[n_training_samples:], target_data[n_training_samples:]),
        )
