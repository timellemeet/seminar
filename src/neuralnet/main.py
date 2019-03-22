import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from neuralnet.loss_layer import LossLayer
from neuralnet.network import Network
from neuralnet.fc_layer import FCLayer
from neuralnet.activation_layer import ActivationLayer
from neuralnet.activation_func import tanh, tanh_prime, sigmoid, sigmoid_prime, softmax, softmax_prime, relu, relu_prime
from neuralnet.loss_func import mse, mse_prime, cross_entropy, cross_entropy_prime
from neuralnet.data_func import vectorize_labels, k_fold
from neuralnet.confusion_matrix import plot_confusion_matrix

# import data
# change this to your local repo location and file names
# change to small data set for testing, entire data set for measuring
os.chdir("..")
training = np.genfromtxt('data/images_small.csv', delimiter=',')
labels = vectorize_labels(np.genfromtxt('data/labels_small.csv', delimiter=','))
test = np.genfromtxt('data/images_test.csv', delimiter=',')
original_test_labels = np.genfromtxt('data/labels_test.csv', delimiter=',')
test_labels = vectorize_labels(original_test_labels)
# normalize data
training /= 255
test /= 255

# parameters
features = 784
output_classes = 10


def train_and_test(hidden_layers, activation, activation_prime, loss_activation, loss_activation_prime,
                   loss, loss_prime, epochs=20, learning_rate=0.01, test_size=1000):
    fold_train_data, fold_train_labels, fold_val_data, fold_val_labels = k_fold(training, labels, 5, 5)

    # initialize network instance
    net = Network()
    # fill it with several layers
    net.add(FCLayer(features, hidden_layers[0]))
    net.add(ActivationLayer(activation, activation_prime))
    for i in range(1, len(hidden_layers)):
        net.add(FCLayer(hidden_layers[i - 1], hidden_layers[i]))
        net.add(ActivationLayer(activation, activation_prime))
    net.add(FCLayer(hidden_layers[-1], output_classes))
    net.add(LossLayer(loss_activation, loss_activation_prime, loss, loss_prime))

    # train the model on training data and labels using specific hyper-parameters
    errors, val_errors = net.fit(fold_train_data, fold_train_labels, fold_val_data, fold_val_labels,
                                 epochs=epochs, learning_rate=learning_rate)

    # test
    out = net.predict(test[:test_size])
    # extract specific predicted number from output neuron probabilities
    y_pred = np.zeros(test_size)
    for i in range(test_size):
        y_pred[i] = np.argmax(out[i:i + 1])

    return accuracy_score(y_pred, original_test_labels[:test_size]), errors, val_errors, net.layers[0].weights


# # plot and print performance measures
# plot_confusion_matrix(y_pred, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]), normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()
# print("The accuracy of the model is {}".format(accuracy_score(y_hat, original_test_labels[:test_size])))

accuracy, errors, val_errors, weights = train_and_test([30], relu, relu_prime, softmax, softmax_prime, cross_entropy,
                                                       cross_entropy_prime,
                                                       epochs=20, learning_rate=0.01, test_size=10000)
print("The test accuracy of the model is {}".format(accuracy))

def plot_error(error, val_error):
    fig, ax1 = plt.subplots()
    ax1.plot(error, 'r', label="training loss ({:.6f})".format(error[-1]))
    ax1.plot(val_error, 'b--', label="validation loss ({:.6f})".format(val_error[-1]))
    ax1.grid(True)
    ax1.set_xlabel('iteration')
    ax1.legend(loc="best", fontsize=9)
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')
    plt.show()
plot_error(errors, val_errors)
