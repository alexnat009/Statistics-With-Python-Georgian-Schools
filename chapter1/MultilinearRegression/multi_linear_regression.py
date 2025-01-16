import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn

np.random.seed(0)

X1 = 4 * np.random.rand(150, 1)
X2 = 4 * np.random.rand(150, 1)
y = 4 * X1 + 2 * X2 - 6 + np.random.randn(150, 1)

X_b = np.c_[np.ones((150, 1)), X1, X2]
X_s = np.c_[X1, X2]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"Theta best:\n{theta_best}")

reg = LinearRegression().fit(X_s, y)
intercept = reg.intercept_
coefficients = reg.coef_.flatten()
print("Intercept:", intercept)
print("Coefficients:", coefficients)

X_new = np.array([[1, 0], [0, 0], [0, 1]])

X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]

y_predict = X_new_b.dot(theta_best)

print(f"Predicted values:\n{y_predict}")

points = np.array([[1, 0, -2.18159436],
				   [0, 0, -6.07869782],
				   [0, 1, -3.99960443]])

theta_0, theta_1, theta_2 = theta_best.flatten()  # theta_best contains intercept and coefficients for X1, X2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, y, color='b', label='Data Points')

x1_range = np.linspace(0, 4, 10)
x2_range = np.linspace(0, 4, 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

Y_grid = theta_0 + theta_1 * X1_grid + theta_2 * X2_grid

ax.plot_surface(X1_grid, X2_grid, Y_grid, color='r', alpha=0.3)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='g', s=100, label='Predicted values')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')

ax.legend()

plt.show()
y_pred = X_b.dot(theta_best)
