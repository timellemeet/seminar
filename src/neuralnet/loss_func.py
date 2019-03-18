
import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    loss_vector = np.zeros(m)
    for i in range(m):
        if y_true[0][i] == 1:
            loss_vector[i] = -np.log(y_pred[0][i])
        else:
            loss_vector[i] = -np.log(1-y_pred[0][i])
    return np.sum(loss_vector)/m


def cross_entropy_prime(y_true, y_pred):
    m = y_true.shape[1]
    loss_derivative = np.zeros([m, 1])
    for i in range(m):
        if y_true[0][i] == 1:
            loss_derivative[i] = -1/y_pred[0][i]
        else:
            loss_derivative[i] = 1/(1-y_pred[0][i])
    return loss_derivative


