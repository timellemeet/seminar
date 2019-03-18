
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from neuralnet.network import Network
from neuralnet.fc_layer import FCLayer
from neuralnet.activation_layer import ActivationLayer
from neuralnet.activation_func import tanh, tanh_prime, sigmoid, sigmoid_prime, softmax, softmax_prime
from neuralnet.loss_func import mse, mse_prime, cross_entropy, cross_entropy_prime
from neuralnet.data_func import vectorize_labels
from neuralnet.confusion_matrix import plot_confusion_matrix

# training data
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

os.chdir("..")
# import data
training = np.genfromtxt('data/images_small.csv', delimiter=',')
labels = vectorize_labels(np.genfromtxt('data/labels_small.csv', delimiter=','))
test = np.genfromtxt('data/images_test.csv', delimiter=',')
original_test_labels = np.genfromtxt('data/labels_test.csv', delimiter=',')
test_labels = vectorize_labels(original_test_labels)

print(training.shape)
print(training[0:1].shape)
print(len(training))
print(labels.shape)
print(labels[0:1])
print(labels[1:2])
# parameters
features = 784
hidden_neurons = 30
output_classes = 10

# network
net = Network()
net.add(FCLayer(features, hidden_neurons))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(hidden_neurons, output_classes))
net.add(ActivationLayer(softmax, softmax_prime))

# train
net.use(cross_entropy, cross_entropy_prime)
net.fit(training, labels, epochs=50, learning_rate=0.1)

# test
test_size = 10
out = net.predict(test[:test_size])
y_hat = np.zeros(test_size)
for i in range(test_size):
    y_hat[i] = np.argmax(out[i:i+1])
print(out)
print(y_hat.shape)

plot_confusion_matrix(y_hat, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]), normalize=True,
                      title='Normalized confusion matrix')
plt.show()
print("The accuracy of the model is {}".format(accuracy_score(y_hat, original_test_labels[:test_size])))
