import numpy as np

from .base_loss import Loss


class MultinomialLoss(Loss):
    """
    Loss for multinomial classification
    Note: Due to computational and numerical reasons we combine Multinomial Loss with last (softmax)
    layer of a neural network. Thus, we always assume softmax layer before multinomial loss and
    multinomial loss after softmax.
    """
    def forward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        :param y_predicted: vector of predicted probabilities of each class
        :param y_true: one-hot encoded vector of a true class
        :return: vector of losses of each observation of shape 1xm
            (where m==y_predicted.shape[1] is a number of observations)
        """
        mask = y_true > 0
        probas = y_predicted.T[mask.T]  # hack: transposing to get values in correct order
        assert probas.size == y_predicted.shape[1] == y_true.shape[1]
        return -np.log(probas).reshape(1, -1)

    def backward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes derivative of a multinomial loss with respect to the softmax input (`a`).
        The code in a neural_network.py assumes that derivative of softmax doesn't change anything.
        For an in-depth analysis of softmax function and it's derivatives look here:
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        """
        assert y_true.shape[0] == y_predicted.shape[0]
        assert y_true.shape[1] == y_predicted.shape[1]
        grad = y_predicted.copy()  # and subtract 1 from true class for each observation
        grad[y_true == 1] -= 1
        return grad
