import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes))

    def train(self):
        def train(self, input_vector, target_vector):
            # input_vector and target_vector can be tuple, list or ndarray

            input_vector = np.array(input_vector, ndmin=2).T
            target_vector = np.array(target_vector, ndmin=2).T

            output_vector1 = np.dot(self.weights_in_hidden, input_vector)
            output_vector_hidden = activation_function(output_vector1)

            output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
            output_vector_network = activation_function(output_vector2)

            output_errors = target_vector - output_vector_network
            # update the weights:
            tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
            tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
            self.weights_hidden_out += tmp
            # calculate hidden errors:
            hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
            # update the weights:
            tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)

    def run(self, input_vector):
        """
        running the network with an input vector input_vector.
        input_vector can be tuple, list or ndarray
        """

        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


if __name__ == "__main__":
    simple_network = NeuralNetwork(no_of_in_nodes=784,
                                   no_of_out_nodes=10,
                                   no_of_hidden_nodes=30,
                                   learning_rate=0.1)
    print(simple_network.weights_in_hidden)
    print(simple_network.weights_hidden_out)
