import numpy as np

from .base_loss import Loss


class QuadraticLoss(Loss):
    """
    Loss for simple regression: squared error.
    For `n` observations it will produce n errors.
    """

    def forward(self, y_predicted: np.ndarray, y_true: np.ndarray):
        assert y_true.shape[0] == y_predicted.shape[0] == 1
        assert y_true.shape[0] == y_predicted.shape[0]
        return (y_true - y_predicted)**2

    def backward(self, y_predicted: np.ndarray, y_true: np.ndarray):
        assert y_true.shape[0] == y_predicted.shape[0] == 1
        assert y_true.shape[0] == y_predicted.shape[0]
        pass
