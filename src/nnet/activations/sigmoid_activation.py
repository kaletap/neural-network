import numpy as np

from .base_activation import Activation


class Sigmoid(Activation):
    def forward(self, x: np.array) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array) -> np.ndarray:
        # Based on a property of a sigmoid function
        s = self(x)
        return s * (1 - s)
