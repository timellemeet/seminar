import numpy as np


# activation functions and its derivative
def tanh(x):
    """
    @param x input vector z=w'x+b
    @return: vector of tanh(x)
    """
    return np.tanh(x);


def tanh_prime(x):
    return 1-np.tanh(x)**2;


def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return (x > 0)*1

def sigmoid(x):
    """
    @param x input vector z=w'x+b
    @return activation vector sigm(x)
    """
    return 1/(1+np.exp(-x))


# faulty output
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def softmax(x):
    e_x = np.exp(x)
    return e_x/np.sum(e_x)


# faulty output
def softmax_prime(x):
    kron_delta = np.eye(x.shape[0])
    return softmax(x)*kron_delta - softmax(x)*softmax(x).T








