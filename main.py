import pandas as pd
import numpy as np
import os

from Neurons import Neuron

train_data = np.array(pd.read_csv("mnist_train.csv"))
m, n = train_data.shape

np.random.shuffle(train_data)

# Instead of each row being an example
# Each column is an example
data_dev = train_data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = train_data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.

test_data = np.array(pd.read_csv("mnist_test.csv"))
m_test, n_test = test_data.shape
test_data = test_data.T
Y_test = test_data[0]
X_test = test_data[1:n_test] / 255.

neuron = Neuron()

if os.path.exists("model.npz"):
    neuron.load_params("model.npz")
    print("Loaded existing weights")
else:
    print("No saved weights found - training from scratch")
    neuron.gradient_descent(X_train, Y_train, 15000)
    neuron.save_params("model.npz")

neuron.test_accuracy(X_test, Y_test)

# for i in range(m_test):
#     neuron.test_prediction(i, X_test, Y_test)