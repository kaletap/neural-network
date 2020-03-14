import numpy as np
from collections import namedtuple
from typing import Iterable
from tqdm import trange

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
        self.n_layers = len(self.layers)
        in_size = input_size
        for layer in self.layers:
            w_shape = [layer.n_neurons, in_size]
            b_shape = [layer.n_neurons, 1]
            w = np.random.normal(0, sd, size=w_shape)
            d_w = np.zeros(w_shape)
            b = np.zeros(b_shape)
            d_b = np.zeros(b_shape)
            # Initializing d_weights and d_bias
            self.weights.append(w)
            self.bias.append(b)
            self.d_weights.append(d_w)
            self.d_bias.append(d_b)
            # Updating number of input size of the next layer (to current vector size)
            in_size = layer.n_neurons
        self.z_states = []
        self.a_states = []

    def forward(self, a: np.ndarray):
        self.z_states = []
        self.a_states = []
        self.a_states.append(a)
        for w, layer in zip(self.weights, self.layers):
            z = w @ a
            self.z_states.append(z)
            a = layer.activation(z)
            self.a_states.append(a)
        return a

    def backward(self, y_true: np.ndarray):
        """
        The link below provides a set of equations for one hidden layer classification perceptron.
        Here, we have to generalize it to multiple layers and arbitrary activation and loss functions.
        https://www.coursera.org/learn/neural-networks-deep-learning/lecture/6dDj7/backpropagation-intuition-optional
        Notation in the code is similar to the one used in the link above by Andrew Ng.
        Also, check out this video by Andrej Karpathy: https://www.youtube.com/watch?v=i94OvYb6noo
        """
        # For simplicity
        loss = self.loss
        layers = self.layers
        weights = self.weights
        bias = self.bias
        d_weights = self.d_weights
        d_bias = self.d_bias
        z_states = self.z_states
        a_states = self.a_states
        n_layers = self.n_layers

        assert len(z_states) == len(a_states) - 1 == len(layers) == len(weights) == len(bias) == n_layers
        # Last element in states is also the output of the network
        y_predicted = a_states[-1]
        m = y_predicted.shape[-1]

        # Backpropagation
        # Calculate d_z (will need d_a in the middle), d_w, d_b
        d_a = (1 / m) * loss.backward(y_predicted, y_true)  # 1 x m (in general it should be jacobian, but we use the
        # fact that it's a diagonal matrix anyway)
        last_activation = layers[n_layers - 1].activation
        a = a_states[n_layers]  # a_states includes additionally input x
        d_z = d_a * last_activation.backward(a)  # 1 x m
        for i in range(n_layers - 1, -1, -1):
            a = a_states[i]
            activation = layers[i].activation
            # Compute d_w
            d_w = d_z @ a.transpose()
            d_weights[i] = d_w
            if i > 0:
                # Compute d_z with respect to the previous hidden state
                z = z_states[i - 1]
                d_z = (weights[i].transpose() @ d_z) * activation.backward(z)

    def fit(self, x, y, *, n_iter: int = 100, lr: float = 0.01):
        for i in trange(n_iter):
            output = self(x)
            self.backward(y)
            for w, d_w in zip(self.weights, self.d_weights):
                w -= lr * d_w
        current_loss = np.mean(self.loss(output, y))
        print(f"Training completed. Loss: {current_loss}")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
