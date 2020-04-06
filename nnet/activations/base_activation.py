import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Base class for an activation function"""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Activation (point-wise) function"""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the activation function.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
