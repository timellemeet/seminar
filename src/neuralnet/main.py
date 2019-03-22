import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from neuralnet.Layer import *
from neuralnet.network import Network
from neuralnet.activation_func import tanh, tanh_prime, sigmoid, sigmoid_prime, softmax, softmax_prime, relu, relu_prime
from neuralnet.loss_func import mse, mse_prime, cross_entropy, cross_entropy_prime
from neuralnet.data_func import vectorize_labels, k_fold
from neuralnet.confusion_matrix import plot_confusion_matrix

# import data
# change this to your local repo location and file names
# change to small data set for testing, entire data set for measuring
os.chdir("..")
training = np.genfromtxt('data/mini.csv', delimiter=',')[1:-4]
labels = vectorize_labels(np.genfromtxt('data/mini_label.csv', delimiter=',')[1:-4])
test = np.genfromtxt('data/images_test.csv', delimiter=',')
original_test_labels = np.genfromtxt('data/labels_test.csv', delimiter=',')
test_labels = vectorize_labels(original_test_labels)

# normalize data
training /= 255
test /= 255

#functions
activation = relu
activation_prime = relu_prime
loss = cross_entropy
loss_prime = cross_entropy_prime
loss_activation = softmax
loss_activation_prime = softmax_prime

#test size
test_size = 100

#specify input and output parameters
features = 784
output_classes = 10

#hyper parameters
learning_rate = 1e-2
hidden_layers = [300]
epochs = 10


#set up the network with specified layers, loss, and activation
net = Network()
net.setup_net(hidden_layers, activation, features, output_classes,
                    activation_prime,
                    loss_activation, loss_activation_prime,
                    loss, loss_prime,
                    FCLayer, ActivationLayer, LossLayer)

#prepare data for training
fold_train_data, fold_train_labels, fold_val_data, fold_val_labels = k_fold(training, labels, 5, 5)

# train the model on training data and labels using specific hyper-parameters
errors, val_errors = net.fit(fold_train_data, fold_train_labels, fold_val_data, fold_val_labels,
                             epochs=epochs, learning_rate=learning_rate)

#print the accuracy
print("The test accuracy of the network is: {}"
      .format(net.accuracy(x=test[:test_size], y_true=original_test_labels[:test_size], errors=errors, val_errors=val_errors)))

# # plot and print performance measures
# plot_confusion_matrix(y_pred, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]), normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()
# print("The accuracy of the model is {}".format(accuracy_score(y_hat, original_test_labels[:test_size])))

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
