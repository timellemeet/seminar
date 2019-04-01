import numpy as np
import os
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
from Layer import *
from network import Network
from activation_func import tanh, tanh_prime, sigmoid, sigmoid_prime, softmax, softmax_prime, relu, relu_prime
from loss_func import mse, mse_prime, cross_entropy, cross_entropy_prime
from data_func import vectorize_labels, k_fold, import_data
from performance_func import plot_error, plot_confusion_matrix
from network_queue import Queue
np.random.seed(10)
# import data
training_size = 3000
normalize = True
training, labels, test, original_test_labels, test_labels = import_data(size=training_size, normalize=normalize)

# intializing queue
queue = Queue(training, labels, test, test_labels, original_test_labels)

# specify input and output sizes
features = 784
output_classes = 10

# hyper parameters
learning_rate = 5e-3
hidden_layers = [30]
max_epochs = 5
batch_size = 1
weight_decay = 0.01
momentum = True

architectures = [[30],[100]]
for layers in architectures:
    queue.add(netparams={
        "hidden_layers": layers,
        "features": features,
        "output_classes": output_classes,
        "activation": relu,
        "activation_prime": relu_prime,
        "loss_activation": softmax,
        "loss_activation_prime": softmax_prime,
        "loss": cross_entropy,
        "loss_prime": cross_entropy_prime
    },
    folds=5,
    params={"epochs": 5,
            "learning_rate": 5e-3,
            "batch_size": 1,
            "momentum": False,
            "weight_decay": 0.01})

results, accuracies = queue.execute()

# # prepare data for training by selecting validation set
# fold_train_data, fold_train_labels, fold_val_data, fold_val_labels = k_fold(training, labels, k=5, n=5)
#
# # train the model on training data and labels using specific hyper-parameters
# errors, val_errors, val_accs = net.fit(fold_train_data, fold_train_labels, fold_val_data, fold_val_labels,
#                                        max_epochs, learning_rate, batch_size, momentum, weight_decay)

# print the accuracy
# print("The test accuracy of the network is: {}".format(
#     net.accuracy(x=test, y_true=original_test_labels, errors=errors, val_errors=val_errors)))

# # plot and print performance measures
# plot_confusion_matrix(y_pred, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]),
#                       normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()
plot_error(accuracies[:5], accuracies[5:10])

