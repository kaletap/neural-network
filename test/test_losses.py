import numpy as np

from nnet.losses import QuadraticLoss


def test_quadratic_loss():
    n = 20
    y_true = np.ones([1, n])
    y_predicted = np.zeros([1, n])
    assert (QuadraticLoss()(y_predicted, y_true) == np.ones([1, n])).all()


def test_quadratic_loss_backward():
    n = 20
    y_true = np.ones([1, n])
    y_predicted = np.zeros([1, n])
    loss = QuadraticLoss()
    assert loss.backward(y_predicted, y_true).shape == (1, n)
