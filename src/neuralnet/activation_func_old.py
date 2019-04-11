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


def reloid(x, alpha):
    for idx, element in enumerate(x):
        if element < -alpha:
            x[idx] = 0
        if abs(element) < alpha:
            x[idx] = 0.5*(element+alpha)
    return x


def reloid_prime(x, alpha):
    new_x = np.zeros(len(x))
    for idx, element in enumerate(x):
        if abs(element) < alpha:
            new_x[idx] = 0.5
        if abs(element) > alpha:
            new_x[idx] = 1

    return new_x



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


class ActivationFunction:
    def __init__(self, act_func, func_prime, alpha=None):
        self.func = act_func
        self.func_prime = func_prime
        self.alpha = alpha
        self.parametric = False
        if act_func == softmax:
            self.parametric = True

    def forward(self, x):
        if self.parametric:
            return self.func(x, self.alpha)
        else:
            return self.func(x)

    def backward(self, x):
        if self.parametric:
            return self.func_prime(x, self.alpha)
        else:
            return self.func_prime(x)





