import numpy as np

from src.losses import QuadraticLoss


def test_quadratic_loss():
    n = 20
    y_true = np.ones([1, n])
    y_predicted = np.zeros([1, n])
    assert (QuadraticLoss()(y_predicted, y_true) == np.ones([1, n])).all()
