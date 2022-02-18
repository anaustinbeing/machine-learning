'''
LINEAR REGRESSION

@author: anaustinbeing
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# sample input data points
X = 4 * np.random.rand(100, 1)
y = 5 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
y_pred = lin_reg.predict(X)

plt.plot(X, y, 'b.')
plt.plot(X, y_pred, 'r-')
plt.show()
