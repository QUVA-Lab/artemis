import numbers
import numpy as np
from artemis.general.mymath import softmax
from artemis.general.numpy_helpers import get_rng

__author__ = 'peter'


"""
Some helper functions for working with neural networks.

For a more complete library including trainable neural nets, convnets, etc, see Plato.
https://github.com/petered/plato

Or, you know, a "mainstream" library, like Keras: https://keras.io/
"""


def initialize_network_params(layer_sizes, mag='xavier-both', base_dist='normal', include_biases = True, rng=None):
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
    if base_dist == 'normal':
        noise_gen = lambda n_in_, n_out_: rng.randn(n_in_, n_out_)
    elif base_dist == 'uniform':
        noise_gen = lambda n_in_, n_out_: (rng.rand(n_in_, n_out_)-0.5) * np.sqrt(12)  # For unit variance
    elif hasattr(base_dist, '__call__'):
        noise_gen = base_dist
    else:
        raise Exception("Unknown base distribution: '%s'" % (base_dist, ))

    if isinstance(mag, numbers.Real):
        ws = [noise_gen(n_in, n_out)*mag for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
    elif mag=='xavier-forward':
        ws = [noise_gen(n_in, n_out)*np.sqrt(1./n_in) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
    elif mag=='xavier-both':
        ws = [2*noise_gen(n_in, n_out)*np.sqrt(1./(n_in+n_out)) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
    elif mag=='xavier-relu':
        ws = [noise_gen(n_in, n_out)*np.sqrt(2./n_in) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
    else:
        raise Exception('No method "%s" yet' % (mag, ))
    if include_biases:
        bs = [np.zeros(n_out) for n_out in layer_sizes[1:]]
        return zip(ws, bs)
    else:
        return ws


def activation_function(data, function_name):
    if function_name=='relu':
        return np.maximum(0, data)
    elif function_name=='linear':
        return data
    elif function_name=='softmax':
        return softmax(data, axis=-1)
    elif function_name in ('sigm', 'sigmoid'):
        return 1./(1+np.exp(-data))
    else:
        raise Exception('Add it.')


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