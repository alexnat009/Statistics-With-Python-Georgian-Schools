import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn.linear_model import Ridge, LinearRegression

np.random.seed(0)
X1 = 4 * np.random.rand(150, 1)
X2 = X1 + 0.01 * np.random.rand(150, 1)
y = 4 * X1 + 4 * X2 - 6 + np.random.randn(150, 1)
# calculate correlation coefficient

# Standard Linear Regression
X_b = np.c_[np.ones((150, 1)), X1, X2]
X_s = np.c_[X1, X2]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"Theta best (Linear Regression):\n{theta_best}")
# linear regression sklearn
reg = LinearRegression().fit(X_s, y)
intercept = reg.intercept_
coefficients = reg.coef_.flatten()

print("Intercept(Linear regression sklearn):", intercept)
print("Coefficients(Linear regression sklearn):", coefficients)

# ridge regression
y = 4 * X1 + 4 * X2 - 6 + np.random.randn(150, 1)
X_combined = np.c_[X1, X2]
X_standardized = np.c_[X1 - np.mean(X1), X2 - np.mean(X2)]
alpha = 1
I = np.eye(X_standardized.shape[1])
theta = np.linalg.inv(X_standardized.T.dot(X_standardized) + alpha * I).dot(X_standardized.T).dot(y)
y_pred = X_combined.dot(theta)
intercept = np.mean(y - y_pred)
print("Intercept(Ridge regression):", intercept)
print("Coefficients(Ridge regression):", theta.flatten())
# Ridge regression sklearn
ridge_reg = Ridge(alpha=1, fit_intercept=True, solver="cholesky")
ridge_reg.fit(X_combined, y)
ridge_intercept = ridge_reg.intercept_[0]
ridge_coefficients = ridge_reg.coef_.flatten()

print(f"Intercept(Ridge regression sklearn): {ridge_intercept}")
print(f"Coefficients(Ridge regression sklearn): {ridge_coefficients}")

# visualisation
x1_range = np.linspace(0, 4, 10)
x2_range = np.linspace(0, 4, 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

Y_grid_standard = (
        theta_best[0] + theta_best[1] * X1_grid + theta_best[2] * X2_grid
)
Y_grid_ridge = (
        ridge_intercept + ridge_coefficients[0] * X1_grid + ridge_coefficients[1] * X2_grid
)

fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X1, X2, y, color='blue', label='Data Points')
ax1.plot_surface(X1_grid, X2_grid, Y_grid_standard, color='red', alpha=0.5)

ax1.set_title("Standard Linear Regression")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("y")
ax1.set_xlim([0, 4])
ax1.set_ylim([0, 4])
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X1, X2, y, color='blue', label='Data Points')
ax2.plot_surface(X1_grid, X2_grid, Y_grid_ridge, color='green', alpha=0.5)

ax2.set_title("Ridge Regression")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.set_zlabel("y")
ax2.set_xlim([0, 4])
ax2.set_ylim([0, 4])

ax2.legend()

plt.tight_layout()
plt.show()
