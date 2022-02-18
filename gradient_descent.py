'''
GRADIENT DESCENT

@author: anaustinbeing
'''

import numpy as np
import matplotlib.pyplot as plt

eta = 0.1  # learning rate
n_iterations = 1000
m = 100

# sample input data points
X = 4 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

X_with_bias = np.c_[np.ones((100, 1)), X]


def find_theta(X, y):
    theta = np.random.randn(2, 1)  # random initialization

    for iteration in range(n_iterations):
        '''
        gradients = 2/m * XT.(X.theta-y)
        '''
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - eta * gradients
    return theta


def predict(X, theta):
    return X.dot(theta)


theta = find_theta(X_with_bias, y)
y_predict = predict(X_with_bias, theta)

plt.plot(X, y, "b.")
plt.plot(X, y_predict, "r-")
plt.show()
