import numpy as np
from activation_func import softmax, ActivationFunction
from loss_func import cross_entropy
# Base class
class Layer:
    def __init__(self):
        self.input = None
        # self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate, batch_size=None):
        raise NotImplementedError

    def update(self, learning_rate, momentum, weight_decay):
        pass

#inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        #He et al. 2015 for weight initialization https://arxiv.org/pdf/1502.01852.pdf
        self.weights = np.random.randn(input_size, output_size)*np.sqrt(2/output_size)
        self.bias = np.zeros([1, output_size])
        self.output_gradient = np.zeros(self.bias.shape[1]).reshape(1,-1)
        self.weights_gradient = np.zeros(self.weights.shape)

        #momentum
        self.prev_output_gradient = np.zeros(self.bias.shape[1]).reshape(1,-1)
        self.prev_weights_gradient = np.zeros(self.weights.shape)

    # returns output for a given input
    # also stores input and output for further reference in backprop
    def forward_propagation(self, input_data):
        self.input = input_data
        output = np.dot(self.input, self.weights) + self.bias
        return output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate, batch_size=None):
        input_error = np.dot(self.weights, output_error.T).T
        self.output_gradient += output_error/batch_size
        dot = np.dot(self.input.T, output_error)
        self.weights_gradient += dot / batch_size
        # dBias = output_error

        # update parameters
        # v_weights = (learning_rate * weights_error+0.9*self.v_weights_prev)
        # v_bias = (learning_rate * output_error + 0.9*self.v_bias_prev)
        # self.weights -= learning_rate * weights_error
        # self.bias -= learning_rate * output_error
        # self.v_weights_prev = v_weights
        # self.v_bias_prev = v_bias
        return input_error

    def update(self, learning_rate, momentum, weight_decay):
        if momentum:
            v_weights = (learning_rate * self.weights_gradient + .9*self.prev_weights_gradient)
            v_bias = (learning_rate * self.output_gradient + .9*self.prev_output_gradient)
            self.weights = (1-learning_rate*weight_decay)*self.weights - v_weights
            self.bias = (1-learning_rate*weight_decay)*self.bias - v_bias

            #momentum
            self.prev_weights_gradient = v_weights
            self.prev_output_gradient = v_bias
            #

            self.weights_gradient = np.zeros(self.weights_gradient.shape)
            self.output_gradient = np.zeros(self.output_gradient.shape)
        else:
            self.weights = (1-learning_rate*weight_decay)*self.weights - learning_rate * self.weights_gradient
            self.bias = (1-learning_rate*weight_decay)*self.bias - learning_rate * self.output_gradient

            self.weights_gradient = np.zeros(self.weights_gradient.shape)
            self.output_gradient = np.zeros(self.output_gradient.shape)




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
        output = self.activation(self.input)
        return output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def delta(self, y, y_hat):
        if self.combined_back_prop:
            return y_hat - y
        else:
            return self.loss_prime(y, y_hat) * self.activation_prime(y_hat)


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation

    # returns the activated input
    # also stores both input and output for backprop
    def forward_propagation(self, input_data):
        self.input = input_data
        output = self.activation.forward(self.input)
        return output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate, batch_size=None):
        return self.activation.backward(self.input) * output_error

