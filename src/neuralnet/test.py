# testing file
import pytest
import numpy as np
from neuralnet.loss_func import  *
from neuralnet.activation_func import *
import numpy as np

# Agent tests
# This checks mostly the simple auxiliary methods as the main algorithm is too complex to unit test and was purely
# tested by hand



@pytest.mark.skip()
def test_soft_max_activation():
    x = np.ones(10)
    x_loss = softmax_prime(x)
    print(x_loss)
    assert x_loss.size == 100
    x = np.ones(5)
    x_loss = softmax_prime(x)
    print(x_loss)
    assert x_loss.size == 25

@pytest.mark.skip()
def test_cross_entropy_loss():
    y = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    y_pred = softmax(np.array([[1, 1, 2, 3, 12, 4, 1, 1, 1, 1]]))
    loss = cross_entropy(y, y_pred)
    print(loss)
    y_pred = softmax(np.array([[1, 1, 2, 3, 8, 4, 6, 1, 6, 1]]))
    loss2 = cross_entropy(y, y_pred)
    print(loss2)
    assert loss < loss2
    loss_deriv = cross_entropy_prime(y, y_pred)
    print(loss_deriv)


def test_sigmoid_activation():
    x = np.ones(30)
    n = np.diag(x).shape
    print(n)
    x_loss = sigmoid_prime(x)
    print(x_loss.shape)
    print(x_loss)
    assert n == x_loss.shape








