import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X = 4 * np.random.rand(150, 1)
y = 4 * X - 6 + np.random.randn(150, 1)

#
print("X values:\n", X)
X_b = np.c_[np.ones((150, 1)), X]
print("X_b values (with bias term):\n", X_b)

# Calculating theta
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta (best fit parameters):\n", theta_best)

# Predicting new values
X_new = np.array([[0], [4]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print("Predicted y values:\n", y_predict)

# Using SVD
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print("Theta (best fit parameters using SVD):\n", theta_best_svd)

# Using scikit-learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Intercept (scikit-learn):\n", lin_reg.intercept_)
print("Coefficients (scikit-learn):\n", lin_reg.coef_)
print("Predicted y values (scikit-learn):\n", lin_reg.predict(X_new))
print("R^2 score (scikit-learn):\n", lin_reg.score(X, y))


def plot_points(X, y, save=False, filepath=None, regression_line=None, X_new=None):
	if regression_line is not None and X_new is not None:
		plt.plot(X_new, regression_line, 'g-')
	plt.scatter(X, y, s=7, c='grey')
	plt.xlabel('X')
	plt.ylabel('y')
	if save:
		plt.savefig(filepath)
	plt.show()


plot_points(X, y, regression_line=None, X_new=X_new, filepath='figures/random_data.png')
plot_points(X, y, regression_line=y_predict, X_new=X_new, filepath='figures/data_with_regression.png')
