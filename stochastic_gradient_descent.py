'''
STOCHASTIC GRADIENT DESCENT

@author: anaustinbeing
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# sample input data points
X = 4 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
# print(sgd_reg.intercept_, sgd_reg.coef_)
y_predict = sgd_reg.predict(X)

plt.plot(X, y, 'b.')
plt.plot(X, y_predict, 'r-')
plt.show()