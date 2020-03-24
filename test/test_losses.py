import numpy as np

from nnet.losses import QuadraticLoss, BinomialLoss, MultinomialLoss


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


def test_multinomial_loss():
    m = 4
    k = 3
    y_true = np.array([[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]])
    y_predicted = np.array([[0.7, 0.7, 0.7, 0.7], [0.1, 0.2, 0.1, 0.1], [0.2, 0.1, 0.2, 0.2]])
    loss = MultinomialLoss()
    loss_value = loss(y_predicted, y_true)
    assert loss_value.shape == (1, m)
    print(loss_value)
    assert (loss_value == np.array([[-np.log(0.7), -np.log(0.7), -np.log(0.2), -np.log(0.7)]])).all()
    grad = loss.backward(y_predicted, y_true)  # grad for each sample
    assert grad.shape == (k, m)
