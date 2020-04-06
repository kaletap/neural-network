import numpy as np

from .base_activation import Activation


class Relu(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, 0)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)
