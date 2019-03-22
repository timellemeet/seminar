
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
from neuralnet.data_func import vectorize_labels
from neuralnet.confusion_matrix import plot_confusion_matrix

# import data
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
hidden_neurons = 30
output_classes = 10

# initialize network instance
net = Network()
# fill it with several layers
net.add(FCLayer(features, hidden_neurons))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(hidden_neurons, output_classes))
net.add(LossLayer(softmax, softmax_prime, cross_entropy, cross_entropy_prime))

# train the model on training data and labels using specific hyper-parameters
errors = net.fit(training, labels, epochs=20, learning_rate=0.01)

# test
test_size = 1000
out = net.predict(test[:test_size])
# extract specific predicted number from output neuron probabilities
y_pred = np.zeros(test_size)
for i in range(test_size):
    y_pred[i] = np.argmax(out[i:i+1])

# plot and print performance measures
plot_confusion_matrix(y_pred, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]), normalize=True,
                      title='Normalized confusion matrix')
plt.show()
print("The accuracy of the model is {}".format(accuracy_score(y_hat, original_test_labels[:test_size])))
