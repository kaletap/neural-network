import numpy as np
from collections import namedtuple
from typing import List, Callable, Iterable

from .losses import Loss


Layer = namedtuple("Layer", "n_neurons activation")


class NeuralNetwork:
    """Feed-forward neural network (also called multi layer perceptron)"""
    def __init__(self, input_size: int, layers: Iterable[Layer], loss: Loss, *, sd=0.1):
        self.input_size = input_size
        self.layers = [Layer(*layer) for layer in layers]
        self.loss = loss
        # Initializing all the matrices with weights
        # as well as the derivatives
        self.weights = []
        self.bias = []
        self.d_weights = []
        self.d_bias = []
        in_size = input_size
        for layer in self.layers:
            w_shape = [layer.n_neurons, in_size]
            b_shape = [layer.n_neurons, 1]
            w = np.random.normal(0, sd, size=w_shape)
            d_w = np.zeros(w_shape)
            b = np.zeros(b_shape)
            d_b = np.zeros(b_shape)
            self.weights.append(w)
            self.bias.append(b)
            self.d_weights.append(d_w)
            self.d_bias.append(d_b)
            # Updating number of input size of the next layer (to current vector size)
            in_size = layer.n_neurons
        self.states = []

    def forward(self, x: np.ndarray):
        self.states = []
        for w in self.weights:
            x = w @ x
            self.states.append(x)
        return x

    def backward(self, y_predicted: np.ndarray, y_true: np.ndarray):
        """
        The link below provides a set of equations for one hidden layer classification perceptron.
        Here, we have to generalize it to multiple layers and arbitrary activation and loss functions.
        https://www.coursera.org/learn/neural-networks-deep-learning/lecture/6dDj7/backpropagation-intuition-optional
        Notation in the code is similar to the one used in the link above by Andrew Ng.
        """
        # For simplicity
        loss = self.loss
        layers = self.layers
        weights = self.weights
        bias = self.bias
        d_weights = self.d_weights
        d_bias = self.d_bias
        states = self.states
        n_layers = len(layers)

        assert len(states) == len(layers) == len(weights) == len(bias) == n_layers
        # Last element in states is also the output of the network
        assert (states[-1] == y_predicted).all()

        m = y_predicted.shape[-1]  # number of observations (could be 1)
        # Calculate d_z (will need d_a in the middle), d_w, d_b
        d_a = loss.backward(y_predicted, y_true)  # 1 x m
        layers[n_layers - 1].activation.backward(states[n_layers - 1])
        d_z = d_a * layers[n_layers - 1].activation.backward(states[n_layers - 1])
        for i in range(n_layers - 1, -1, -1):
            a = states[i]
            d_z = weights[i].transpose()
        raise NotImplementedError("Backpropagation has to be finished")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

