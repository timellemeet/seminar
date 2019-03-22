import numpy as np
from neuralnet.Layer import Layer
from neuralnet.activation_func import softmax
from neuralnet.loss_func import cross_entropy


# inherit from base class Layer
class LossLayer(Layer):
    def __init__(self, activation, activation_prime, loss, loss_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.loss = loss
        self.loss_prime = loss_prime
        self.combined_back_prop = False
        if self.activation == softmax and self.loss == cross_entropy:
            self.combined_back_prop = True

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def delta(self, y, y_hat):
        if self.combined_back_prop:
            return y_hat - y
        else:
            return self.loss_prime(y, y_hat) * self.activation_prime(y_hat)


