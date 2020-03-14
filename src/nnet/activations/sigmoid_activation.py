import numpy as np

from .base_activation import Activation


class Sigmoid(Activation):
    def forward(self, x: np.array) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array):
        # Based on a property of a sigmoid function
        return self(x) * (1 - self(x))
