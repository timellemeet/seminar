import math


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i:i+1]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        errors = []

        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j:j+1]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                # the last layer is the loss layer
                err += self.layers[-1].loss(y_train[j:j+1], output)

                # backward propagation
                # start with last layer since it requires backprop through both the loss and activation function
                error = self.layers[-1].delta(y_train[j:j+1], output)
                # backprop through all subsequent layers, while also updating parameters
                for layer in reversed(self.layers[:-1]):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            # save to epoch error list
            errors.append(err)
        return errors
