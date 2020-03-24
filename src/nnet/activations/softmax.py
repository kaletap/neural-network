import numpy as np

from .base_activation import Activation


class Softmax(Activation):
    """
    In this framework softmax is implemented as Activation. However, it is not activation in the strict sense,
    because it is not a point-wise function. In general, computing backward would require computing jacobian
    matrix, however for multinomial loss the computations simplify a lot. That's why backward computation
    with respect to Softmax input is done in MultinomialLoss rather than in Softmax.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / np.sum(e_x, axis=0)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        MultinomialLoss already computes gradient with respect to input of softmax,
        that's why we return ones, which don't change anything in d_z computation in neural_network.py.
        This code assumes that the loss of NeuralNetwork automatically computes gradients with respect to the input
        of softmax.
        """
        return np.ones_like(x)
