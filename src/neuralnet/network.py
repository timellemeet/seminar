import math
import numpy as np
import time
import datetime
from Layer import *
from sklearn.metrics import accuracy_score
from loss_func import cross_entropy
import matplotlib.pyplot as plt

class Network:
    def __init__(self, hidden_layers,
                  features, output_classes,
                  activation, activation_prime,
                  loss_activation, loss_activation_prime,
                  loss, loss_prime):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        # fill it with several layers
        self.add(FCLayer(features, hidden_layers[0]))
        self.add(ActivationLayer(activation, activation_prime))

        for i in range(1, len(hidden_layers)):
            self.add(FCLayer(hidden_layers[i - 1], hidden_layers[i]))
            self.add(ActivationLayer(activation, activation_prime))

        self.add(FCLayer(hidden_layers[-1], output_classes))
        self.add(LossLayer(loss_activation, loss_activation_prime, loss, loss_prime))

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

    def accuracy(self, x, y_true):
        out = self.predict(x)
        size = len(x)
        # extract specific predicted number from output neuron probabilities
        y_pred = np.zeros(len(x))
        for i in range(size):
            y_pred[i] = np.argmax(out[i:i + 1])

        return accuracy_score(y_pred, y_true)

    def top_losses(self,x,y_true, tops):
        y_pred = self.predict(x)
        losses = []
        for i in range(len(y_pred)):
            losses.append(cross_entropy(y_true[i], y_pred[i]))
        top = np.argsort(np.array(losses))[-tops:]
        for i in top:
            print("loss: {}, true: {}, predicted: {}".format(losses[i], np.argmax(y_true[i,:]), np.argmax(y_pred[i]) ))
            plt.imshow(x[i].reshape(28, 28), cmap='gray')
            plt.show()

    def update_parameters(self, learning_rate, momentum, weight_decay):
        for layer in reversed(self.layers[:-1]):
            layer.update(learning_rate, momentum, weight_decay)

    # train the network
    def fit(self, x_train, y_train, x_val, y_val, epochs, learning_rate, batch_size, momentum, weight_decay):
        errors = []
        val_errors = []
        val_accs = []
        epoch_times = []
        
        # sample dimension first
        samples = len(x_train)
        if samples%batch_size !=0:
            raise Exception("Make sure samples ({}) % batch_size ({}) is 0, now {}".format(samples, batch_size, samples%batch_size))
        
        previous_epoch_time = 0
        # training loop
        for i in range(epochs):
            start_time = time.time()
            seed = np.arange(samples)
            np.random.shuffle(seed)
            x_shuffle = x_train[seed]
            y_shuffle = y_train[seed]

            # train one epoch
            err = self.train_epoch(i, x_shuffle, y_shuffle, samples, learning_rate, batch_size, momentum, weight_decay)
            
            # validate and save to epoch error lists
            val_error, val_acc = self.validate(x_val, y_val, self.layers[-1].loss)
            val_errors.append(val_error)
            val_accs.append(val_acc)
            errors.append(err)
            epoch_time = time.time()-start_time
            epoch_times.append(epoch_time)
            epoch_time = np.mean(epoch_times)
            time_pred_error = epoch_time-previous_epoch_time
            eta = str(datetime.timedelta(seconds=round((epoch_time)*(epochs-(i+1)))))
            print('epoch %d/%d   training error=%f  validation error=%f validation accuracy=%f ETA=%s tpe=%f'  % (i+1, epochs, err, val_error, val_acc, eta, time_pred_error))
            previous_epoch_time = epoch_time
        
        print('Average epoch computational time: ',np.mean(epoch_times))
        
        return [errors, val_errors, val_accs]

    def train_epoch(self, i, x_shuffle, y_shuffle, samples, learning_rate, batch_size, momentum, weight_decay):
        err = 0
        for k in range(1, samples // batch_size):
            start_slice = (k - 1) * batch_size
            end_slice = k * batch_size
            x_batch = x_shuffle[start_slice:end_slice]
            y_batch = y_shuffle[start_slice:end_slice]

            # std_x = np.std(x_batch, axis=1).reshape(-1,1)
            # mu_x = np.mean(x_batch, axis=1).reshape(-1,1)
            #
            # x_batch = (x_batch-mu_x)/std_x

            for j in range(batch_size):
                # forward propagation
                output = x_batch[j:j + 1]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                # the last layer is the loss layer
                err += self.layers[-1].loss(y_batch[j:j + 1], output)

                # backward propagation
                # start with last layer since it requires backprop through both the loss and activation function
                error = self.layers[-1].delta(y_batch[j:j + 1], output)
                output_error = error

                # backprop through all subsequent layers, while also updating parameters
                for layer in reversed(self.layers[:-1]):
                    output_error = layer.backward_propagation(output_error, learning_rate, batch_size)

            self.update_parameters(learning_rate, momentum=momentum, weight_decay=weight_decay)

        # calculate average error on all samples
        err /= samples
        return err

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

        return loss / validation_size, accuracy_score(y_pred, y_actual)

    def save_parameters(self, name):
        path = '../networks/' +name
        np.savez(path,*[layer.weights for layer in self.layers if type(layer) == FCLayer], *[layer.bias for layer in self.layers if type(layer) == FCLayer])

    def load_parameters(self,file):
        parameters = np.load(file)

        i = 0
        for layer in self.layers:
            if type(layer) == FCLayer:
                s = parameters['arr_'+str(i)].shape
                if layer.weights.shape == parameters['arr_'+str(i)].shape:
                    layer.weights = parameters['arr_'+str(i)]
                    layer.bias = parameters['arr_'+ str(i+len(self.layers)//2)]
                    i+=1
                else:
                    raise Exception("Shape error, amount of neurons and layers in network unequal to network trying to load")