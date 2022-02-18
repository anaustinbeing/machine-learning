'''
CLOSED-FORM LINEAR REGRESSION (a.k.a normal equation):
Closed form finds the theta best which will help to predict solution
having the lowest global mean square error (mse) value.
This method is suitable if:
1. The model to be used is linear regression
2. The number of features is less
3. The training sample is less (fewer than 20,000)

@author anaustinbeing
'''

import numpy as np
import matplotlib.pyplot as plt

X = 4 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

X_with_bias = np.c_[np.ones((100, 1)), X]    # Adding bias term (a column of ones)

def find_theta_best(x, y):
    '''
    (XT.X)-1 XTY
    '''
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

def predict(theta, x):
    return x.dot(theta)

theta_best = find_theta_best(X_with_bias, y)

y_pred = predict(theta_best, X_with_bias)

plt.plot(X, y, "b.")
plt.plot(X, y_pred, "r-")
plt.show()
