import numpy as np

from .base_activation import Activation


class Identity(Activation):
    def forward(self, x: np.array) -> np.ndarray:
        return x

    def backward(self, x: np.array):
        pass
