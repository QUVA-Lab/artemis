import numbers
import numpy as np
from artemis.general.mymath import softmax
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import bad_value

__author__ = 'peter'


"""
Some helper functions for working with neural networks.

For a more complete library including trainable neural nets, convnets, etc, see Plato.
https://github.com/petered/plato

Or, you know, a "mainstream" library, like Keras: https://keras.io/
"""


def initialize_network_params(layer_sizes, mag='xavier-both', base_dist='normal', last_layer_zero = False, include_biases = True, scale=1., rng=None):
    """
    Initialize parameters for a fully-connected neural network.

    :param layer_sizes: A list of integers indicating layer sizes (including that of the input layer)
    :param mag: The standard deviation, or a string identifying a method for selecting the standard deviation.
        String options can be:
            'xavier-forward' - Best for preserving variance of a linear, tanh, or sigmoidal network across layers.
            'xavier-both': - A compromize between preserving the variance of the forward, backward pass
            'xavier-relu': - Best for preserving variance on the forward pass in a ReLU net.
    :param base_dist: 'normal' or 'uniform', or a function taking (n_in, n_out) and returning a (n_in, n_out) array
    :param include_biases: Also create initial biases.
    :param rng: A random number generator or seed
    :return: A list of 2-tuples of (weight, bias) parameters (if include_biases is True) otherwise a list of weight matrices.

    Note: To get the weights/biases in separate lists, simply go:
        weights, biases = zip(*initialize_network_params(...))

    Note: See http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    For a good explanation of the 'xavier' initialization schemes.
    """
    rng = get_rng(rng)
    ws = [initialize_weight_matrix(n_in, n_out, mag=mag, base_dist=base_dist, scale=scale, rng=rng) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
    if last_layer_zero:
        ws[-1][:] = 0
    if include_biases:
        bs = [np.zeros(n_out) for n_out in layer_sizes[1:]]
        return zip(ws, bs)
    else:
        return ws


def initialize_weight_matrix(n_in, n_out, mag='xavier', base_dist='normal', scale=1., rng=None):
    """
    Initialize a weight matrix
    :param n_in: Number of input units
    :param n_out: Number of output units
    :param mag: The magnitude, or a string identifying how to calculate the magnitude.
        String options can be:
            'xavier-forward' - Best for preserving variance of a linear, tanh, or sigmoidal network across layers.
            'xavier-both': - A compromize between preserving the variance of the forward, backward pass
            'xavier-relu': - Best for preserving variance on the forward pass in a ReLU net.
    :param base_dist: 'normal' or 'uniform', or a function taking (n_in, n_out) and returning a (n_in, n_out) array
    :param rng: Random number generator or seed
    :return: A shape (n_in, n_out) initial weight matrix.
    """
    rng = get_rng(rng)

    w_base = rng.randn(n_in, n_out) if base_dist == 'normal' else \
        (np.rand(n_in, n_out) - 0.5)*np.sqrt(12) if base_dist=='uniform' else \
        bad_value(base_dist)

    mag_number = \
        np.sqrt(2./(n_in+n_out)) if mag in ('xavier', 'xavier-both') else \
        np.sqrt(1./n_in) if mag=='xavier-forward' else \
        np.sqrt(2./n_in) if mag=='xavier-relu' else \
        mag if isinstance(mag, numbers.Real) else \
        bad_value(mag)

    return w_base * (mag_number*scale)


def initialize_conv_kernel(kernel_shape, mag = 'xavier', rng = None):
    rng = get_rng(rng)
    if mag=='xavier':
        n_kern_out, n_kern_in, k_size_y, k_size_x = kernel_shape
        fan_in = k_size_y*k_size_x*n_kern_in
        fan_out = n_kern_out*k_size_y+k_size_x
        mag = np.sqrt(2./(fan_in+fan_out))
    else:
        assert isinstance(mag, (int, float)), mag
    return mag*rng.randn(*kernel_shape)


def activation_function(data, function_name):
    if function_name=='relu':
        return np.maximum(0, data)
    elif function_name=='linear':
        return data
    elif function_name=='softmax':
        return softmax(data, axis=-1)
    elif function_name=='softplus':
        return np.log(np.exp(data)+1)
    elif function_name in ('sigm', 'sigmoid'):
        return 1./(1+np.exp(-data))
    elif function_name == 'tanh':
        return np.tanh(data)
    else:
        raise Exception('No Nonlinearity "{}".  Add it.'.format(function_name))


def forward_pass_activations(input_data, weights, biases = None, hidden_activations='relu', output_activations = 'relu'):
    """
    Return the activations from a forward pass of a ReLU net.
    :param input_data: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activations: Indicates the output layer activation function
    :return: activations: A len(weights)*2+1 array, where:
        activations[::2] are the input and pre-nonlinearities (len(ws)+1)
        activations[1::2] are the post-nonlinerity activations (len(ws))
    """
    activations = [input_data]
    if biases is None:
        biases = [0]*len(weights)
    else:
        assert len(biases)==len(weights)
    x = input_data
    for i, (w, b) in enumerate(zip(weights, biases)):
        u = x.dot(w)+b
        x = activation_function(u, output_activations if i==len(weights)-1 else hidden_activations)
        activations += [u, x]
    return activations


def forward_pass(input_data, weights, biases = None, hidden_activations='relu', output_activation = 'relu'):
    """
    Do a forward pass of an MLP
    :param input_data: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activation: Indicates the output layer activation function
    :return: output: A (input_data.shape[0], weights[-1].shape[1]) array of outputs
    """
    return forward_pass_activations(input_data=input_data, weights=weights, biases=biases, hidden_activations=hidden_activations, output_activations=output_activation)[-1]