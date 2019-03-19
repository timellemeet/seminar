import numpy as np

class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))

class Softmax:
    @staticmethod
    def activation(z):
        pass

    @staticmethod
    def prime(z):
        pass

class CrossEntrophy:
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        pass

    @staticmethod
    def prime(y_true, y_pred):
        pass

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        pass

class Network:
    def __init__(self, inputs, layers, hiddenneuron, output, activations):
        self.w = {}
        self.b = {}
        self.sigma = {}

        #first layer
        self.w[1] = np.random.rand(inputs,hiddenneuron)
        self.b[1] = np.zeros(hiddenneuron)
        self.sigma[2] = activations[0]

        #other layers
        for i in range(1, layers):
            self.w[i+1] = np.random.rand(hiddenneuron,hiddenneuron)
            self.b[i+1] = np.zeros(hiddenneuron)
            self.sigma[i+2] = activations[0]

        #output layer
        self.w[layers+1] = np.random.rand(hiddenneuron, output)
        self.b[layers+1] = np.zeros(output)
        self.sigma[layers+2] = activations[1]

    def feedforward(self,x):
        #w'a +b
        a = {}

        #f(a)
        h = {1: x}

        for i in range(1, len(self.w)+1):
            # current layer is i
            # activation layer is i+1
            a[i + 1] = np.dot(h[i], self.w[i]) + self.b[i]
            h[i + 1] = self.sigma[i + 1].activation(a[i + 1])
        return a, h

    def back_prop(self, a, h, y_true):
        pass

    def fit(self, x, y_true, loss, epochs, batch_size, lr = 1e-03):
        pass

    def predict(self, x):
        _,a = self.feedforward(x)
        return a[-1]




def main():
    # img = np.genfromtxt("C:\\Users\\niels\\gitlab\\seminar\\src\\data\\images.csv", delimiter=',')
    # labels = np.genfromtxt('"C:\\Users\\niels\\gitlab\\seminar\\src\\data\\labels.csv"', delimiter=',')
    init = np.random.rand(3)
    print("input: {}".format(init))
    np.random.seed(1)
    nn = Network(3,1,2,5, (Relu, Sigmoid))
    print(nn.feedforward(init))
main()



