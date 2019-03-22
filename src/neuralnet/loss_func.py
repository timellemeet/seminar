
import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    return -np.log(np.sum(y_true * y_pred))


def cross_entropy_prime(y_true, y_pred):
    return -y_true*y_pred


