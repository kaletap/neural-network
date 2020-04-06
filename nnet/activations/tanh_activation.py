import numpy as np

from .base_activation import Activation


class TanH(Activation):
    def __init__(self, a=0.01):
        self.a = a

    def forward(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x)
        e_minus_x = np.exp(-x)
        return (e_x - e_minus_x) / (e_x + e_minus_x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - self(x)**2
