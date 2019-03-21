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
        self.loss = MSE(activations[1]) #specify in fit function
        self.lr = 0

        #first layer
        self.w[1] = np.ones([inputs,hiddenneuron])*.05
        self.b[1] = np.zeros([hiddenneuron,1])
        self.sigma[2] = activations[0]

        #other layers
        for i in range(1, layers):
            self.w[i+1] = np.ones([hiddenneuron,hiddenneuron])*.05
            self.b[i+1] = np.zeros([hiddenneuron,1])
            self.sigma[i+2] = activations[0]

        #output layer
        self.w[layers+1] = np.ones([hiddenneuron, output])*.05
        self.b[layers+1] = np.zeros([output,1])
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
        _,batchsize = x.shape
        ones = np.ones([1,batchsize])

        for i in range(1, self.depth+2):
            # current layer is i
            # activation layer is i+1
            # a[i + 1] = np.dot(h[i], self.w[i]) + self.b[i]
            temp = np.matmul(self.b[i], ones)
            # print("h matrix: {} \nw matrix: {} \nb vector: {} \nb matrix: {}"
            #       .format(h[i].shape, self.w[i].shape, self.b[i].shape, temp.shape))
            a[i+1] = np.dot(self.w[i].T, h[i]) + temp
            h[i+1] = self.sigma[i+1].activation(a[i+1])
            # print(a[i+1].shape)
            # print(h[i+1].shape)
            # h[i + 1] = self.sigma[i + 1].activation(a[i + 1])

        return a, h

    def back_prop(self, a, h, y_true):
        """

        :param a:
        :param h:
        :param y_true:
        """

        delta = self.loss.delta(y_true=y_true, y_pred=h[self.depth+2])
        delta_next = 0
        #function (6)
        # print("h matrix: {} \n delta shape: {}"
        #       .format(h[self.depth+1].shape, delta.shape))
        dw = np.dot(h[self.depth+1], delta.T)

        #backprop the other layers
        for i in range(self.depth+1, 2, -1):
            # print("a matrix: {} \nw matrix: {} \n delta: {}"
            #       .format(a[i].shape, self.w[i].shape, delta.shape))
            delta_next = self.sigma[i].prime(a[i]) * np.dot(self.w[i], delta)
            dw_next = np.dot(h[i-1], delta.T)
            self.weight_update(i, dw, delta)
            delta = delta_next
            dw = dw_next


    def weight_update(self, index, dw, delta):
        """

        :param index: layer index
        :param dw:  partial a^(d) / partial w^(d) * delta
        :param delta: partial y_hat / partial a^(d) * partial C / partial y_hat
        """
        # print("delta: {} \n b: {}".format(delta.shape, self.b[index].shape))
        #not sure yet about update rule for bias term
        self.w[index] -= self.lr * dw
        self.b[index] -= self.lr * np.mean(delta)




    def fit(self, x, y_true, loss, epochs, batch_size, lr = 1e-06):
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
                a, h = self.feedforward(x[:,k:l])
                self.back_prop(a, h, y_true[:,k:l])

            if (i + 1) % 10 == 0:
                print("Epoch: {}".format(i))
                _, h = self.feedforward(x)
                print("Loss: {}".format(self.loss.loss(y_true, h[self.depth + 2])))

    def predict(self, x):
        x = np.reshape(x, (-1,1))
        _,h = self.feedforward(x)
        return h[self.depth + 2]

def test_softmax():
    print(Softmax.activation(np.array([1,2,3])))
    print(Softmax.prime(np.array([1,2,3])))

def vectorize(x):
    result = np.zeros([1,10])
    for i in np.nditer(x):
        t = np.zeros([1,10])
        t[0][int(i)] = 1
        result = np.append(result, t, axis=0)
    return result[1:]

def main():
    img = np.genfromtxt("C:\\Users\\niels\\gitlab\\seminar\\src\\data\\images.csv", delimiter=',')
    img = img[1:-1].T
    labels = np.genfromtxt('C:\\Users\\niels\\gitlab\\seminar\\src\\data\\labels.csv', delimiter=',')
    labels = labels[1:-1]
    y_true = vectorize(labels).T

    nn = Network(784,2,300,10, (Relu, Sigmoid))
    # a, h = nn.feedforward(img)
    # nn.back_prop(a, h, y_true)
    nn.fit(img,y_true, loss=MSE, epochs=100,batch_size=30)
    for i in range(len(img.T)-990):
        print("Prediction: {}, actual: {} \n".format(nn.predict(img[:,i]), labels[i]))

main()



