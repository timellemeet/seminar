
import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


def cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = 0
    for i in range(m):
        if y_true[0][i] == 1:
            loss=-np.log(y_pred[0][i])
    return loss


def cross_entropy_prime(y_true, y_pred):
    return -y_true*y_pred


