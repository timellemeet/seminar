# testing file
import pytest
import numpy as np
from neuralnet.loss_func import  *
from neuralnet.activation_func import *


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


def test_cross_entropy_loss():
    y = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    y_pred = softmax(np.array([[1, 1, 2, 3, 12, 4, 1, 1, 1, 1]]))
    loss = cross_entropy(y, y_pred)
    print(loss)
    y_pred2 = softmax(np.array([[1, 1, 2, 3, 8, 4, 6, 1, 6, 1]]))
    loss2 = cross_entropy(y, y_pred2)
    print(loss2)
    assert loss < loss2
    loss_deriv = cross_entropy_prime(y, y_pred)
    loss_deriv2 = cross_entropy_prime(y, y_pred2)
    print(loss_deriv)
    print(loss_deriv2)

@pytest.mark.skip()
def test_sigmoid_activation():
    x = np.ones(30)
    n = np.diag(x).shape
    print(n)
    x_loss = sigmoid_prime(x)
    print(x_loss.shape)
    print(x_loss)
    assert n == x_loss.shape


def test_reloid_activation():

    x = [-2,-1,-0.5,0,0.5,1,2]
    x_reloid = reloid(x,alpha=0.6)
    assert x_reloid == pytest.approx([0,0,0.05,0.3,0.55,1,2])
    x_reloid_prime = reloid_prime(x,alpha = 0.6)
    assert x_reloid_prime == pytest.approx([0,0,0,0.5,0.5,0.5,1,1])





