import numpy as np
from numpy import genfromtxt
from neuralnet.NeuralNetwork import NeuralNetwork
import os

print (os.getcwd())
os.chdir("..")
print (os.getcwd())

def vectorize_labels(labels):
    vector_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        vector_labels[i, int(labels[i])-1] = 1
    return vector_labels


#import data
training = np.genfromtxt('data/images_small.csv')
labels = vectorize_labels(np.genfromtxt('data/labels_small.csv'))
test = np.genfromtxt('data/images_test.csv')
test_labels = vectorize_labels(np.genfromtxt('data/labels_test.csv'))



nn = NeuralNetwork(no_of_in_nodes=784,
                                   no_of_out_nodes=10,
                                   no_of_hidden_nodes=30,
                                   learning_rate=0.1)
print(nn.weights_in_hidden)