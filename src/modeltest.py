import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from model import load_model, forward, predictions, accuracy


def test(filename, x, y):
    try:
        w1, b1, w2, b2 = load_model(filename)
        print('Model loaded!')

    except FileNotFoundError:
        sys.exit("File not found! Go train a model!")

    _, _, _, a2 = forward(x, w1, b1, w2, b2)
    pred = predictions(a2)
    acc = accuracy(pred, y)

    print(f"Accuracy {acc}")
    return

if __name__ == "__main__":

    test_data = pd.read_csv("data/fashion-mnist_test.csv")
    test_data = np.array(test_data) 
    Y_test = test_data[:, 0].reshape(-1, 10000) # DIMS (1, 10000)
    X_test = test_data[:, 1:785].T / 255 # DIMS (784, 10000)

    n, m = X_test.shape

    filename = 'models/fashionNN.pkl'

    test(filename, X_test, Y_test)