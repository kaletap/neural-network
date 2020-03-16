import numpy as np

from .base_loss import Loss


class BinomialLoss(Loss):
    """
    Loss for binary classification (negative binomial likelihood). We model the probability of `y`=1 and since
    `y` is binary it has to have Bernoulli distribution. We than maximize it's (log) likelihood and thus minimize
    negative log likelihood assuming all observations are independent.
    Also known as cross-entropy between empirical and binomial distribution, or Binary Cross Entropy (torch).
    The output of the network should be one dimensional (not two) and have values between 0 and 1.
    For `n` observations it will produce n error values.
    Note: Due to computational and numerical reasons it is better to combine Binomial Loss with last (sigmoid)
    layer of a neural network.
    """

    def forward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        assert y_true.shape[0] == y_predicted.shape[0] == 1
        # Warning: in this way we compute log of something that can be zero and do unnecessary computations
        return -np.where(y_true == 1, np.log(y_predicted), np.log(1 - y_predicted))

    def backward(self, y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        assert y_true.shape[0] == y_predicted.shape[0] == 1
        assert y_true.shape[0] == y_predicted.shape[0]
        return np.where(y_true == 1, -np.log(1 / y_predicted), 1 / (1 - y_predicted))
