import numpy as np
from collections import namedtuple
from typing import List, Callable, Iterable


Layer = namedtuple("Layer", "n_neurons activation")


class NeuralNetwork:
    """Feed-forward neural network (also called multi layer perceptron)"""
    def __init__(self, input_size: int, layers: Iterable[Layer], *, sd=0.1):
        self.input_size = input_size
        self.layers = layers
        # Initializing all the matrices with weights
        # as well as the derivatives
        self.weights = []
        self.bias = []
        self.d_weights = []
        self.d_bias = []
        in_size = input_size
        for layer in self.layers:
            w = np.random.normal(0, sd, size=[in_size, layer.n_neurons])
            d_w = np.zeros([in_size, layer.n_neurons])
            b = np.zeros([layer.n_neurons, 1])
            d_b = np.zeros([layer.n_neurons, 1])
            self.weights.append(w)
            self.bias.append(b)
            self.d_weights.append(d_w)
            self.d_bias.append(d_b)
        self.states = []

    def forward(self, x: np.ndarray):
        self.states = []
        for w in self.weights:
            x = w @ x
            self.states.append(x)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


