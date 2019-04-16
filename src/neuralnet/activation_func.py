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
    # return x * (x > 0)
    return np.where(x > 0, x, 0)


def relu_prime(x):
    # return (x > 0)*1
    return np.where(x > 0, 1, 0)


def reloid(x, alpha):
    half_reloid = np.where(x > alpha, x, 0.5*(x+alpha))
    return np.where(half_reloid > 0, half_reloid, 0)


def reloid_prime(x, alpha):
    half_reloid_prime = np.where(x > alpha, 1, 1/2)
    return np.where(x > 0, half_reloid_prime, 0)


def leaky_relu(x, alpha):
    return np.where(x > 0, x, alpha*x)


def leaky_relu_prime(x, alpha):
    return np.where(x > 0, 1, alpha)


def param_relu(x, theta):
    return np.where(x > 0, x, theta*x)


def param_relu_prime(x, theta):
    return np.where(x > 0, 1, theta)



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
        if act_func == reloid or act_func == leaky_relu:
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

    def forward_param(self, x, theta):
        return self.func(x, self.alpha, theta)


    def backward_param(self, x, theta):
        if self.parametric:
            return self.func_prime(x, self.alpha)
        else:
            return self.func_prime(x)





