import numpy as np
from numpy import genfromtxt
from neuralnet.NeuralNetwork import NeuralNetwork
import os
import matplotlib.pyplot as plt

print (os.getcwd())
os.chdir("..")
print (os.getcwd())

def vectorize_labels(labels):
    vector_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        vector_labels[i, int(labels[i])] = 1
    return vector_labels


#import data
training = np.genfromtxt('data/images_small.csv', delimiter=',')
labels = vectorize_labels(np.genfromtxt('data/labels_small.csv', delimiter=','))
test = np.genfromtxt('data/images_test.csv', delimiter=',')
test_labels = vectorize_labels(np.genfromtxt('data/labels_test.csv'))

print(training.shape)
print(training)
print(labels.shape)
print(labels)
print(test.shape)
print(test_labels.shape)

nn = NeuralNetwork(no_of_in_nodes=np.size(training,1),
                                   no_of_out_nodes=10,
                                   no_of_hidden_nodes=30,
                                   learning_rate=0.1)
print(nn.weights_in_hidden.shape)

iterations = 6000
errors = np.zeros(iterations)
for i in range(iterations):
    nn.train(training[i], labels[i])
    errors[i] = np.dot(np.transpose(nn.current_error), nn.current_error)
print(errors)
plt.plot(errors)
plt.show()

y_hat = nn.run(test[0])
test_error = y_hat.T - test_labels[0]
print(y_hat)
print(test_error)
print(y_hat.shape)
print(test_labels[0].shape)
# for i in range(test.size(0)):
#     y_hat = nn.run(test[i])
