import numpy as np

from artemis.ml.datasets.datasets import DataCollection, DataSet


__author__ = 'peter'


def get_synthetic_clusters_dataset(n_clusters = 4, n_dims = 20, n_training = 1000, n_test = 200,
        sparsity = 0.5, flip_noise = 0.1, seed = 3425, dtype = 'float32'):
    """
    A dataset consisting of clustered binary data with "bit-flip" noise, and the corresponding cluster labels.
    This should be trivially solvable by any classifier, and serves as a basic test of whether your classifier is
    completely broken or not.

    :param n_clusters:
    :param n_dims:
    :param n_samples_training:
    :param n_samples_test:
    :param sparsity:
    :param flip_noise:
    :param seed:
    :return:
    """

    rng = np.random.RandomState(seed)
    labels = rng.randint(n_clusters, size = n_training+n_test)  # (n_samples, )
    centers = rng.rand(n_clusters, n_dims) < sparsity  # (n_samples, n_dims)
    input_data = centers[labels]
    input_data = np.bitwise_xor(input_data, rng.rand(*input_data.shape) < flip_noise).astype(dtype)

    return DataSet(
        training_set = DataCollection(input_data[:n_training], labels[:n_training]),
        test_set = DataCollection(input_data[n_training:], labels[n_training:]),
        name = 'Synthetic Clusters Dataset'
        )
