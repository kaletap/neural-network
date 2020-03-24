import numpy as np
from collections import namedtuple
from typing import Iterable
from tqdm import trange

from .losses import Loss


Layer = namedtuple("Layer", "n_neurons activation")


class NeuralNetwork:
    """Feed-forward neural network (also called multi layer perceptron)"""
    def __init__(self, input_size: int, layers: Iterable[Layer], loss: Loss, *, sd=1e-3):
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
        # Last element in states is also the output of the network (y_predicted)
        a = a_states[-1]
        m = a.shape[-1]

        # Backpropagation
        # Calculate d_z (will need d_a in the middle), d_w, d_b
        d_a = (1 / m) * loss.backward(a, y_true)  # 1 x m (if `a` is one-dimensional)
        activation = layers[n_layers - 1].activation
        z = z_states[n_layers - 1]
        d_z = d_a * activation.backward(z)  # 1 x m (in general we would have to compute a jacobian of `a` with
        # respect to `z`, but we use the fact that it's a diagonal matrix anyway)
        for i in range(n_layers - 1, -1, -1):
            a = a_states[i]  # a_states includes additionally input x (it has size one bigger than z_states)
            # Compute d_w
            d_w = d_z @ a.transpose()
            d_weights[i] = d_w
            if i > 0:
                # Compute d_z with respect to the previous hidden state
                activation = layers[i - 1].activation
                z = z_states[i - 1]
                d_z = (weights[i].transpose() @ d_z) * activation.backward(z)  # note: partial derivative of `z` with
                # respect to previous hidden state `a` does not depend on `a` (it's a linear transformation)

    def fit(self, x, y, *, n_iter: int = 50_000, lr: float = 1e-4, verbose=True) -> float:
        for i in trange(n_iter) if verbose else range(n_iter):
            self.forward(x)  # forward pass to save intermediate states
            self.backward(y)  # backward pass
            for w, d_w in zip(self.weights, self.d_weights):
                w -= lr * d_w
        output = self(x)
        current_loss = np.mean(self.loss(output, y))
        if verbose:
            print(f"Training completed. Loss: {current_loss}")
        return current_loss

    def fit_predict(self, x, y, *, n_iter: int = 50_000, lr: float = 1e-4):
        self.fit(x, y, n_iter=n_iter, lr=lr)
        return self(x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # TODO: hacky, change it with proper bias implementation
    def predict(self, x: np.ndarray):
        n = x.shape[1]
        a = np.vstack([x, np.ones([1, n])])
        return self.forward(a)
