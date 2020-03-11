import numpy as np

from src.activations import Identity, Sigmoid


def test_sigmoid():
    p = 19
    n = 1
    x = np.ones([p, 1])
    sigmoid = Sigmoid()
    assert sigmoid.backward(x).shape == (p, 1)
