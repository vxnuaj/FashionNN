'''

This model will be able to classify images in the Fashion MNIST dataset.

To start, it will contain 2 layers, 1 hidden and 1 output. The hidden layer will have 200 neurons, the output will have 10.

'''

import numpy as np
import pandas as pd
import pickle

def save_model(w1, b1, w2, b2, filename):
    with open(filename, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def init_params():
    w1 = np.random.rand(32, 784) - .5 # DIMS: (200, 784) | (neurons, input features)
    b1 = np.zeros((32, 1)) - .5 # DIMS: (200, 1) | (neurons, bias per neuron)
    w2 = np.random.rand(10, 32) - .5 # DIMS: (10, 200) | (neurons, input features)
    b2 = np.zeros((10, 1)) - .5 # DIMS: (10, 1) | (neurons, bias per neuron)
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    #z = z - np.max(z, axis = 0)
    return np.exp(z) / sum(np.exp(z))

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1 # DIMS: (200, 60000) | (neurons, total samples)
    a1 = ReLU(z1) # DIMS: (200, 60000) | (neurons, total samples)
    z2 = np.dot(w2, a1) + b2 # DIMS: (10, 60000) | (neurons, total samples)
    a2 = softmax(z2) # DIMS: (10, 60000) | (neurons, total samples)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((np.max(y) + 1, y.size)) # DIMS: (10, 60000)
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y

#loss, where m is equal to our total samples of 60000
def loss(y, a, m):
    eps = 1e-10
    l = - np.sum(y * np.log(a + eps)) * 1 / m
    return l

def predictions(a2):
    pred = np.argmax(a2, axis = 0)
    return pred #Returns index of the highest probability in activations, a2 (where dims are 10, 60000)

def accuracy(pred, y):
    acc = np.sum(pred == y) / y.size
    return acc

def deriv_ReLU(z):
    return z > 0

def backward(x, y, z1, a1, a2, w2, m):
    dz2 = a2 - y # DIMS: (10, 60000)
    dw2 = dz2.dot(a1.T) * 1 / m # DIMS: (10, 200)
    db2 = np.sum(dz2) * 1 / m # DIMS: (10, 1)
    dz1 = (w2.T.dot(dz2) * deriv_ReLU(z1)) # DIMS: (200, 60000)
    dw1 = dz1.dot(x.T) * 1 / m # DIMS: (200, 784)
    db1 = np.sum(dz1) * 1 / m # DIMS(200, 1)
    return dz2, dw2, db2, dz1, dw1, db1

def update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha):
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    return w2, b2, w1, b1

def gradient_descent(x, y, y_orig, w1, b1, w2, b2, alpha, epochs):
    for epoch in range(epochs):
        z1, a1, _, a2 = forward(x, w1, b1, w2, b2)

        l = loss(y, a2, m)
        pred = predictions(a2)
        acc = accuracy(pred, y_orig)

        _, dw2, db2, _, dw1, db1 = backward(x, y,z1, a1, a2, w2, m)
        w2, b2, w1, b1 = update(w2, b2, w1, b1, dw2, db2, dw1, db1, alpha)

        print(f"Epoch: {epoch} | Loss: {l} | Accuracy: {acc}")
    
    return w1, b1, w2, b2

def model(x, y, alpha, epochs, filename):
    one_hot_y = one_hot(y)

    try:
        w1, b1, w2, b2 = load_model(filename)
        print(f"Model found! Initializing {filename}!")
    except FileNotFoundError:
        print("Model not found! Initializing new parameters! ")
        w1, b1, w2, b2 = init_params()

    w1, b1, w2, b2 = gradient_descent(x, one_hot_y, y, w1, b1, w2, b2, alpha, epochs)

    save_model(w1, b1, w2, b2, filename)
    return

if __name__ == "__main__":

    data = pd.read_csv("data/fashion-mnist_train.csv")
    data = np.array(data)
    np.random.shuffle(data)
    X_train = data[:, 1:785].T / 255 #DIMS: (784, 60000) | (features per sample, sample size)
    Y_train = data[:, 0].reshape(-1, 60000) #DIMS (1, 60000) | (num_labels/samples, num_labels_per_sample)
    n, m = X_train.shape

    model(X_train, Y_train, .1, 1000, 'models/fashionNN.pkl')

    ''''
    Reflection

    - Need to learn about softmax and ReLU implementations, rather than only sigmoid
    - FIO how to use categorical cross entropy
    - Need to learn the derivations or get an intuition of them for Softmax and ReLU implementations alongside cat. cross entropy.
    
    '''