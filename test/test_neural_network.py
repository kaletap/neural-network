import numpy as np

from src.neural_network import NeuralNetwork, Layer
from src.activations import Sigmoid, Identity
from src.losses import QuadraticLoss


def test_forward():
    n = 100
    p = 4
    x = np.ones([p, n])
    nnet = NeuralNetwork(p, [(10, Sigmoid), (11, Sigmoid), (1, Identity)], QuadraticLoss())
    output = nnet(x)
    assert output.shape == (1, n)
