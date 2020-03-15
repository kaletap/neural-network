import numpy as np

from nnet import NeuralNetwork
from nnet.activations import Sigmoid, Identity
from nnet.losses import QuadraticLoss


def test_forward():
    n = 100
    p = 4
    x = np.ones([p, n])
    net = NeuralNetwork(p, [(10, Sigmoid()), (11, Sigmoid()), (1, Identity())], QuadraticLoss())
    output = net(x)
    assert output.shape == (1, n)


def test_backward():
    n = 100
    p = 4
    x = np.ones([p, n])
    y = np.zeros([1, n])
    net = NeuralNetwork(p, [(10, Sigmoid()), (11, Sigmoid()), (1, Identity())], QuadraticLoss())
    net(x)
    net.backward(y)
    assert True
