import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/fashion-mnist_train.csv') #
data = np.array(data) # DIMS: (60000, 785) | (samples, label + pixel features)

Y_train = data[:, 0].reshape(60000, -1)
X_train = data[:, 1:785] # DIMS: (60000, 784) | (samples, pixel features)

plt.imshow(X_train[80, :].reshape(28, 28))
plt.show()

