import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Abstract base class for a loss function of a network.
    """
    @abstractmethod
    def forward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Returns a Value of a loss function."""
        pass

    @abstractmethod
    def backward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Derivative of a loss. We have to use it since we do not use automatic differentiation."""
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
