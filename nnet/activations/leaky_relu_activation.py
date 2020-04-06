import numpy as np

from .base_activation import Activation


class LeakyRelu(Activation):
    def __init__(self, a=0.01):
        self.a = a

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, self.a * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, self.a)
