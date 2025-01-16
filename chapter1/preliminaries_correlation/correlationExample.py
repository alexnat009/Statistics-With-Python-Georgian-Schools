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

df = pd.DataFrame(np.c_[X1, X2, y])


def plot_correlation_matrix(X, save=False, filepath=None):
	corr_pd = X.corr()
	seaborn.heatmap(corr_pd, annot=True)
	if save:
		plt.savefig(filepath)
	plt.show()


def calculate_corr(X1, X2):
	Cov_X1X2 = np.mean(X1 * X2) - np.mean(X1) * np.mean(X2)
	Var_X1 = np.mean(X1 ** 2) - np.mean(X1) ** 2
	Var_X2 = np.mean(X2 ** 2) - np.mean(X2) ** 2
	corr = Cov_X1X2 / (math.sqrt(Var_X1) * math.sqrt(Var_X2))
	return corr


calculate_corr(X1, X2)
plot_correlation_matrix(df, save=True, filepath='figures/corrMatrix.png')
