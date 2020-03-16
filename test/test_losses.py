import numpy as np

from nnet.losses import QuadraticLoss, BinomialLoss


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


def test_binomial_loss():
    n = 20
    # First case
    y_true = np.ones([1, n])
    y_predicted = np.zeros([1, n]) + 0.4
    loss = BinomialLoss()(y_predicted, y_true)
    assert (loss == -np.log(0.4)).all()
    assert loss.shape == (1, n)

    # Second case
    y_true = np.zeros([1, n])
    y_predicted = np.zeros([1, n]) + 0.4
    loss = BinomialLoss()(y_predicted, y_true)
    assert (loss == -np.log(0.6)).all()
    assert loss.shape == (1, n)
