from artemis.general.numpy_helpers import get_rng
from artemis.ml.datasets.datasets import DataSet
from artemis.ml.tools.neuralnets import initialize_network_params, forward_pass


def get_synthetic_deep_data(n_samples, layer_sizes, hidden_activations='softplus', output_activation='linear', normalize = True, rng=1234):
    """
    Generate data from a randomly initialized neural network.
    :param n_samples: Number of samples to generate
    :param layer_sizes: Sizes of network layers
    :param hidden_activations: Hidden activation functions
    :param output_activation: Output activation function
    :param normalize: Normalize the output over samples (remove global mean, divide by std)
    :param rng: A random number generator or seed.
    :return: x, y
        x is an (n_samples, layer_sizes[0]) array
        y is a (n_samples, layer_sizes[-1]) array
    """
    rng = get_rng(rng)
    ws = initialize_network_params(layer_sizes=layer_sizes, mag = 'xavier-forward', include_biases=False, rng=rng)
    x = rng.randn(n_samples, layer_sizes[0])
    y = forward_pass(input_data=x, weights=ws, hidden_activations=hidden_activations, output_activation=output_activation)
    if normalize:
        y = (y - y.mean(axis=0))/y.std(axis=0)
    return x, y


def get_synthetic_deep_dataset(n_training, n_test, **kwargs):
    """
    :param n_training: Number of training samples
    :param n_test: Number of test samples
    :param kwargs: See get_synthetic_deep_data above
    :return:
    """
    x, y = get_synthetic_deep_data(**kwargs)
    return DataSet.from_xy(x, y, training_fraction=n_training/float(n_training+n_test))
