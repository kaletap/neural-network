import numpy as np

from .base_activation import Activation


class Identity(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
