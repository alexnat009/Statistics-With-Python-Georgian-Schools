import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 6
X = 2 * np.round(np.random.rand(n, 1), 2)
y = -1 + 3 * X + np.round(np.random.randn(n, 1), 2)

sorted_indices = np.argsort(X, axis=0)
X = X[sorted_indices].squeeze(axis=2)
y = y[sorted_indices].squeeze(axis=2)

X_b = np.c_[np.ones((n, 1)), X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best2 = np.array([[-2.84587357], [4.66777374]])
theta_best3 = np.array([[2.84587357], [0.66777374]])
theta_best4 = np.array([[3], [0]])  # Yellow line y = 3

X_new = np.array([[np.min(X) - 0.1], [np.max(X) + 0.1]])
X_new_b = np.c_[np.ones((2, 1)), X_new]

y_predict = X_new_b.dot(theta_best)
y_predict2 = X_new_b.dot(theta_best2)
y_predict3 = X_new_b.dot(theta_best3)
y_predict4 = X_new_b.dot(theta_best4)


def linear_equation(m, c):
	return f"y = {round(m, 2)}x{' - ' if c < 0 else ' + '}{round(abs(c), 2)}"


lines = [
	(theta_best, "r-", "red", linear_equation(theta_best[1][0], theta_best[0][0])),
	(theta_best2, "g-", "green", linear_equation(theta_best2[1][0], theta_best2[0][0])),
	(theta_best3, "b-", "blue", linear_equation(theta_best3[1][0], theta_best3[0][0])),
	(theta_best4, "y-", "y", linear_equation(theta_best4[1][0], theta_best4[0][0]))
]


def plot_lines_and_distances(X, y, lines, save=False, filepath=None):
	for idx, (theta, line_style, color, label) in enumerate(lines):
		plt.figure(idx)
		plt.scatter(X, y)
		for theta_all, line_style_all, color_all, label_all in lines:
			y_line = X_new_b.dot(theta_all)
			plt.plot(X_new, y_line, line_style_all, label=label_all)
		y_points = X_b.dot(theta)
		for i in range(n):
			plt.text(X[i] - 0.04, y[i] - 0.2, f"($x_{i}$, $y_{i}$)")
			y_pred = np.array([[1, X[i][0]]]).dot(theta)
			plt.scatter(X[i], y_pred, color=color, zorder=5)
			plt.plot([X[i], X[i]], [y[i], y_points[i]], "k:")
			plt.text(X[i] + 0.02, (y[i] + y_pred.flatten()) / 2, f"$({np.round(y_pred, 2).flatten()[0]} - y_{i})^2$",
					 color=color)
		plt.legend(loc="upper left")
		plt.xlabel("$x$")
		plt.ylabel("$y$", rotation=0)
		plt.title(f"Distances for {color} line")
		if save:
			plt.savefig(f"{filepath}distanceY{color}.png")
		plt.show()


# plot_lines_and_distances(X, y, lines)


def plot_lines(X, y, lines, save=False, filepath=None):
	for num_lines in range(1, len(lines) + 1):
		plt.figure()
		plt.scatter(X, y)
		for i in range(num_lines):
			theta, line_style, color, label = lines[i]
			y_line = X_new_b.dot(theta)
			plt.plot(X_new, y_line, line_style, label=label)
		if num_lines != 0:
			plt.legend(loc="upper left")
		plt.xlabel("$x$")
		plt.ylabel("$y$", rotation=0)
		if save:
			plt.savefig(f"{filepath}line{num_lines + 1}.png")
		plt.show()


# plot_lines(X, y, lines)
