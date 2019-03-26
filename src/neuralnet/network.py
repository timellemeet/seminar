import math
import numpy as np
from sklearn.metrics import accuracy_score

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    def setup_net(self,hidden_layers, activation,
                  features, output_classes,
                  activation_prime,
                  loss_activation, loss_activation_prime,
                  loss, loss_prime,
                  FCLayer, ActivationLayer, LossLayer):
        # fill it with several layers
        self.add(FCLayer(features, hidden_layers[0]))
        self.add(ActivationLayer(activation, activation_prime))

        for i in range(1, len(hidden_layers)):
            self.add(FCLayer(hidden_layers[i - 1], hidden_layers[i]))
            self.add(ActivationLayer(activation, activation_prime))

        self.add(FCLayer(hidden_layers[-1], output_classes))
        self.add(LossLayer(loss_activation, loss_activation_prime, loss, loss_prime))

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

    def accuracy(self, x, errors, y_true, val_errors):
        out = self.predict(x)
        size = len(x)
        # extract specific predicted number from output neuron probabilities
        y_pred = np.zeros(len(x))
        for i in range(size):
            y_pred[i] = np.argmax(out[i:i + 1])

        return accuracy_score(y_pred, y_true)

    # train the network
    def fit(self, x_train, y_train, x_val, y_val, epochs, learning_rate, batch_size):
        errors = []
        val_errors = []

        # sample dimension first
        samples = len(x_train)
        if samples%batch_size !=0:
            raise Exception("Make sure samples ({}) % batch_size ({}) is 0, now {}".format(samples, batch_size, samples%batch_size))

        # training loop
        for i in range(epochs):
            err = 0
            seed = np.arange(samples)
            np.random.shuffle(seed)
            x_shuffle = x_train[seed]
            y_shuffle = y_train[seed]


            for k in range(1, samples//batch_size):
                start_slice = (k-1)*batch_size
                end_slice = k*batch_size
                x_batch = x_shuffle[start_slice:end_slice]
                y_batch = y_shuffle[start_slice:end_slice]
                batch_error = np.zeros([1,10])

                for j in range(batch_size):
                    # forward propagation
                    output = x_batch[j:j+1]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # compute loss (for display purpose only)
                    # the last layer is the loss layer
                    err += self.layers[-1].loss(y_batch[j:j+1], output)

                    # backward propagation
                    # start with last layer since it requires backprop through both the loss and activation function
                    error = self.layers[-1].delta(y_batch[j:j+1], output)
                    batch_error += error

                batch_error /= batch_size


                # backprop through all subsequent layers, while also updating parameters
                for layer in reversed(self.layers[:-1]):
                    batch_error = layer.backward_propagation(batch_error, learning_rate)

            # calculate average error on all samples
            err /= samples

            # validate and save to epoch error lists
            val_error = self.validate(x_val, y_val, self.layers[-1].loss)
            val_errors.append(val_error)
            errors.append(err)
            print('epoch %d/%d   training error=%f  validation error=%f' % (i+1, epochs, err, val_error))
        return errors, val_errors

    def validate(self, x_validation, y_validation, lossfunc):
        loss = 0
        result = self.predict(x_validation)
        validation_size = len(x_validation)
        y_pred = np.zeros(validation_size)
        y_actual = np.zeros(validation_size)
        for i in range(validation_size):
            y_pred[i] = np.argmax(result[i:i + 1])
            y_actual[i] = np.argmax(y_validation[i:i+1])
            loss += lossfunc(y_validation[i:i+1], result[i:i+1])
        return loss / validation_size
        # return 1-accuracy_score(y_pred, y_actual)