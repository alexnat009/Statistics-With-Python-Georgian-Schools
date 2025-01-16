from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import optuna
import numpy as np
import pandas as pd


def optimize_lof(df, column, normalize=False, dropna=True):
	n_neighbors_range = range(10, np.sqrt(len(df)).astype(int), 2)
	contamination_range = np.linspace(0.01, 0.2, 20)

	lof_df = df[column].copy()
	lof_df = lof_df.dropna() if dropna else lof_df.fillna(0)
	if normalize:
		lof_df[column] = np.log1p(lof_df[column])
	result = []
	for n_neighbors in n_neighbors_range:
		for contamination in contamination_range:
			lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
			labels = lof.fit_predict(lof_df)
			outlier_indices = lof_df[labels == -1].index

			result.append({
				'n_neighbors': n_neighbors,
				'contamination': contamination,
				'outliers': lof_df.loc[outlier_indices]
			})
	return result


def visualize_pca(pcaDf, outliers=None, all_outliers=None, title="PCA Projection", dimensions=2, save=False,
				  filename=None):
	if dimensions not in [2, 3]:
		raise ValueError("Only 2D and 3D visualizations are supported.")

	if isinstance(pcaDf, np.ndarray):
		pca_columns = [f"Principal Component {i + 1}" for i in range(pcaDf.shape[1])]
		pcaDf = pd.DataFrame(data=pcaDf, columns=pca_columns)

	fig = plt.figure(figsize=(10, 8))

	if dimensions == 2:
		plt.scatter(
			pcaDf["Principal Component 1"],
			pcaDf["Principal Component 2"],
			color="blue",
			s=10,
			label="Data Points"
		)

		if outliers is not None:
			outliers_pca = pcaDf.loc[pcaDf.index.intersection(outliers.index)]

			plt.scatter(
				outliers_pca["Principal Component 1"],
				outliers_pca["Principal Component 2"],
				color="red",
				s=50,
				label="LOF Outliers",
				marker="p"
			)

		if all_outliers is not None:
			colors = ["green", "orange", "purple", "brown", "pink"]
			for i, iteration_outliers in enumerate(all_outliers):
				plt.scatter(
					iteration_outliers[:, 0],
					iteration_outliers[:, 1],
					color=colors[i % len(colors)],
					s=50,
					label=f"Outliers Iteration {i + 1}",
					marker='x'
				)

		plt.title(title)
		plt.xlabel("Principal Component 1")
		plt.ylabel("Principal Component 2")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		if save:
			plt.savefig(filename)
		plt.show()

	elif dimensions == 3:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(
			pcaDf["Principal Component 1"],
			pcaDf["Principal Component 2"],
			pcaDf["Principal Component 3"],
			c='blue',
			s=10,
			label="Data Points"
		)

		if outliers is not None:
			outliers_pca = pcaDf.loc[pcaDf.index.intersection(outliers.index)]
			ax.scatter(
				outliers_pca["Principal Component 1"],
				outliers_pca["Principal Component 2"],
				outliers_pca["Principal Component 3"],
				c='red',
				s=50,
				label="LOF Outliers",
				marker='p'
			)

		if all_outliers is not None:
			colors = ["green", "orange", "purple", "brown", "pink"]
			for i, iteration_outliers in enumerate(all_outliers):
				ax.scatter(
					iteration_outliers[:, 0],
					iteration_outliers[:, 1],
					iteration_outliers[:, 2],
					color=colors[i % len(colors)],
					s=50,
					label=f"Outliers Iteration {i + 1}",
					marker='x'
				)

		ax.set_title(title)
		ax.set_xlabel("Principal Component 1")
		ax.set_ylabel("Principal Component 2")
		ax.set_zlabel("Principal Component 3")
		ax.legend()
		if save:
			plt.savefig(filename)
		plt.show()


def gridSearchFor2DPCA():
	pca_2d = PCA(n_components=2)
	df_pca_2d = pca_2d.fit_transform(dfNumericalColumns.fillna(0))

	n_neighbors = [5, 10, 15, 20, 25, 30, 35, 40, 45]

	# Create a 3x3 subplot figure
	fig, axes = plt.subplots(3, 3, figsize=(15, 15))
	fig.suptitle("PCA 2D Projection with LOF Outliers (Varying n_neighbors)", fontsize=16)

	# Apply LOF (Local Outlier Factor) to the 2D PCA data
	for i, n in enumerate(n_neighbors):
		# Apply LOF with the current n_neighbors
		lof_2d = LocalOutlierFactor(n_neighbors=n, contamination='auto')
		lof_2d_labels = lof_2d.fit_predict(df_pca_2d)
		outliers_2d = df_pca_2d[lof_2d_labels == -1]

		# Get the current subplot
		ax = axes[i // 3, i % 3]

		# Scatter plot of the data points and outliers
		ax.scatter(
			df_pca_2d[:, 0],
			df_pca_2d[:, 1],
			color="blue", s=10, label="Data points"
		)
		ax.scatter(
			outliers_2d[:, 0],
			outliers_2d[:, 1],
			color="red", s=50, label="Outliers"
		)
		ax.set_title(f"n_neighbors = {n}")
		ax.set_xlabel("Principal Component 1")
		ax.set_ylabel("Principal Component 2")
		ax.grid(True)
		print(f'LOF with n_neighbors = {n} detected {len(outliers_2d)} outliers')

	# Adjust layout and show the plot
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()
