import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x);


def tanh_prime(x):
    return 1-np.tanh(x)**2;


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    sigm = sigmoid(x)*(1-sigmoid(x))
    return np.diagflat(sigm)


def softmax(x):
    e_x = np.exp(x)
    return e_x/np.sum(e_x)

def softmax_prime(x):
    kron_delta = np.eye(x.shape[0])
    return softmax(x)*kron_delta - softmax(x)*softmax(x).T








