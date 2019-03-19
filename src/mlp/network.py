import numpy as np
from random import shuffle

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

class MSE:
    def __init__(self, activation_fn=None):
        """
        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))

class Linear:
    @staticmethod
    def activation(z):
        return z

    @staticmethod
    def prime(z):
        return 0

class Softmax:
    @staticmethod
    def activation(z: np.array):
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def prime(z: np.array):
        a = Softmax.activation(z)
        J = np.zeros([len(z),len(z)])
        for i in range(len(z)):
            for j in range(len(z)):
                if i == j:
                    J[i][j] = a[i]*(1-a[j])
                else:
                    J[i][j] = -a[j]*a[i]
        return J


class CrossEntropy:
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
        """

        :param inputs:
        :param layers:
        :param hiddenneuron:
        :param output:
        :param activations:
        """
        self.w = {}
        self.b = {}
        self.sigma = {}
        self.depth = layers
        self.loss = None #specify in fit function
        self.lr = 0

        #first layer
        self.w[1] = np.ones([inputs,hiddenneuron])*.05
        self.b[1] = np.zeros(hiddenneuron)
        self.sigma[2] = activations[0]

        #other layers
        for i in range(1, layers):
            self.w[i+1] = np.ones([hiddenneuron,hiddenneuron])*.05
            self.b[i+1] = np.zeros(hiddenneuron)
            self.sigma[i+2] = activations[0]

        #output layer
        self.w[layers+1] = np.ones([hiddenneuron, output])*.05
        self.b[layers+1] = np.zeros(output)
        self.sigma[layers+2] = activations[1]

    def feedforward(self,x):
        """

        :param x:
        :return:
        """
        #w'a +b
        a = {}

        #f(a) --> last entry == prediction
        h = {1: x}

        for i in range(1, len(self.w)+1):
            # current layer is i
            # activation layer is i+1
            a[i + 1] = np.dot(h[i], self.w[i]) + self.b[i]
            h[i + 1] = self.sigma[i + 1].activation(a[i + 1])
        return a, h

    def back_prop(self, a, h, y_true):
        """

        :param a:
        :param h:
        :param y_true:
        """

        delta = self.loss.delta(y_true, h[self.depth+2])
        delta_next = 0
        #function (6)
        dw = np.dot(h[self.depth+1].T, delta)

        #backprop the other layers
        for i in range(self.depth+2, 2, -1):
            delta_next = self.sigma[i].prime(a[i]) * np.dot(self.w[i-1], delta)
            dw_next = np.dot(h[i-1].T, delta)
            self.weight_update(i-1, dw, delta)
            delta = delta_next
            dw = dw_next


    def weight_update(self, index, dw, delta):
        """

        :param index: layer index
        :param dw:  partial a^(d) / partial w^(d) * delta
        :param delta: partial y_hat / partial a^(d) * partial C / partial y_hat
        """

        #not sure yet about update rule for bias term
        self.w[index] -= self.lr * dw
        self.b[index] -= self.lr * np.mean(delta,0)




    def fit(self, x, y_true, loss, epochs, batch_size, lr = 1e-03):
        """

        :param x:
        :param y_true:
        :param loss:
        :param epochs:
        :param batch_size:
        :param lr:
        """
        self.lr = lr
        self.loss = loss(Sigmoid)

        for i in range(epochs):
            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                a, h = self.feedforward(x[k:l])
                self.back_prop(a, h, y_true[k:l])

            if (i + 1) % 10 == 0:
                _, h = self.feedforward(x)
                print("Loss: {}".format(self.loss.loss(y_true[i], h[self.depth + 2])))

    def predict(self, x):
        _,h = self.feedforward(x)
        return h[self.depth + 2]

def test_softmax():
    print(Softmax.activation(np.array([1,2,3])))
    print(Softmax.prime(np.array([1,2,3])))

test_softmax()

def vectorize(x):
    result = np.zeros([1,10])
    for i in np.nditer(x):
        t = np.zeros([1,10])
        t[0][int(i)] = 1
        result = np.append(result, t, axis=0)
    return result[1:]

def main():
    img = np.genfromtxt("C:\\Users\\niels\\gitlab\\seminar\\src\\data\\mini.csv", delimiter=',')
    img = img[1:-1]
    labels = np.genfromtxt('C:\\Users\\niels\\gitlab\\seminar\\src\\data\\mini_label.csv', delimiter=',')
    labels = labels[1:-1]
    y_true = vectorize(labels)

    nn = Network(784,1,34,10, (Relu, Sigmoid))
    nn.fit(img,y_true, loss=MSE, epochs=100,batch_size=10)
    for i in range(len(img)-990):
        print("Prediction: {}, actual: {} \n".format(nn.predict(img[i]), y_true[i]))
main()



