import numpy as np
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self, neurons = 64, inputs = 784, output = 10):
        # 10  = neurons (row)
        # 784 = inputs (column)
        # random.rand = 0 -> 1, so if we -0.5 = -0.5 -> 0.5
        self.W1 = np.random.rand(neurons, inputs) - 0.5
        # 10 = neurons (row)
        # 1 = column
        self.b1 = np.random.rand(neurons, 1) - 0.5

        self.W2 = np.random.rand(output, neurons) - 0.5
        self.b2 = np.random.rand(output, 1) - 0.5

    def forward(self, X):
        # Layer 1
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.ReLU(self.Z1)
        # Layer 2
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.Softmax(self.Z2)

    def backward(self, X, Y):
        # examples
        m = Y.size
        # Zeros matrix w/ ones
        one_hot_Y = self.one_hot(Y)
        # Layer 2
        self.dZ2 = self.A2 - one_hot_Y
        self.dW2 = (1 / m) * self.dZ2.dot(self.A1.T) 
        self.db2 = (1 / m) * np.sum(self.dZ2, axis = 1, keepdims = True)
        # Layer 1
        self.dZ1 = self.W2.T.dot(self.dZ2) * self.deriv_ReLU(self.Z1)
        self.dW1 = (1 / m) * self.dZ1.dot(X.T)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis = 1, keepdims = True)

    def update_params(self, alpha = 0.01):
        self.W1 -= alpha * self.dW1
        self.b1 -= alpha * self.db1
        self.W2 -= alpha * self.dW2
        self.b2 -= alpha * self.db2

    def gradient_descent(self, X, Y, iterations):
        for i in range(iterations+1):
            self.forward(X)
            self.backward(X, Y)
            self.update_params()
            if i % 10 == 0:
                print(f"Iterations: {i} / {iterations}")
                print(f"Accuracy: {self.get_accuracy(Y)}")
                print(f"Loss: {self.Loss(Y)}")

    def Loss(self, Y):
        m = Y.size
        target = self.A2[Y, np.arange(m)]
        return np.mean(-np.log(target + 1e-8))

    def get_accuracy(self, Y):
        return np.sum(np.argmax(self.A2, 0) == Y) / Y.size

    def one_hot(self, Y):
        # Y.size = m = number of examples
        # Y.max() = 9 because number ranging from 0 - 9
        # Y.max() + 1 because 0 - 9 is 10 digits
        # [m, 10]
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        # np.arange(Y.size) which is m creates an array 0 -> m
        # [0->m, Y], Y being the current column in that row
        # it then adds a 1 to that [row, column]
        one_hot_Y[np.arange(Y.size), Y] = 1
        # Each row is the example, but we want the column to be the example
        # So we Transpose it
        return one_hot_Y.T

    # Activation functions
    def ReLU(self, Z):
        return np.maximum(0, Z)
    # Derivative of ReLU
    def deriv_ReLU(self, Z):
        return Z > 0
    
    def Softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis = 0, keepdims = True)

    def save_params(self, filename="model.npz"):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_params(self, filename="model.npz"):
        data = np.load(filename)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

    def get_predictions(self):
        return np.argmax(self.A2, axis=0)

    def predict(self, X):
        self.forward(X)
        return self.get_predictions()

    def test_accuracy(self, X, Y):
        self.forward(X)
        print(f"Test accuracy: {self.get_accuracy(Y) * 100}%")

    def test_prediction(self, index, X, Y):
        current_image = X[:, index, None]        # shape (784, 1) — one example
        prediction = self.predict(current_image)[0]
        label = Y[index]
        print(f"Prediction: {prediction}, Label: {label}")

        img = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(img, interpolation='nearest')
        plt.title(f"Prediction: {prediction} | Label: {label}")
        plt.show()