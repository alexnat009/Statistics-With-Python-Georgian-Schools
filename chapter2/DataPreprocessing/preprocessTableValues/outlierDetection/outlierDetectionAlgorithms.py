from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from scipy.stats import zscore


# 1. Z-Score Method
def zscoreTest(df, column, normalize=False, threshold=3):
	zscoreDf = df.copy()
	if normalize:
		zscoreDf[column] = np.log1p(zscoreDf[column])
	zscores = zscore(zscoreDf[column], nan_policy='omit')
	return zscoreDf[np.abs(zscores) > threshold]


# 2. IQR Method
def iqr_test(df, column):
	iqr_df = df.copy()
	Q1 = iqr_df[column].quantile(0.25)
	Q3 = iqr_df[column].quantile(0.75)
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	outliers = iqr_df[(iqr_df[column] < lower_bound) | (iqr_df[column] > upper_bound)]
	return outliers.dropna()


# 3. Isolation Forest Method
def isolation_forest_test(df, column, n_estimators=100, contamination=0.01, random_state=42, normalize=False,
						  dropna=True):
	iso_df = df[column].copy()
	iso_df = iso_df.dropna() if dropna else iso_df.fillna(0)
	if normalize:
		iso_df[column] = np.log1p(iso_df[column])
	iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
	iso_df['Anomaly_Score'] = iso_forest.fit_predict(iso_df)
	outlier_indices = iso_df[iso_df['Anomaly_Score'] == -1].index
	return df.loc[outlier_indices]


# 4. DBSCAN Method
def dbscan_test(df, column, eps=0.5, min_samples=5, dropna=True):
	dbscan_df = df[column].copy()
	dbscan_df = dbscan_df.dropna() if dropna else dbscan_df.fillna(0)
	dbscan = DBSCAN(eps=eps, min_samples=min_samples)
	labels = dbscan.fit_predict(dbscan_df)
	outlier_indices = dbscan_df[labels == -1].index
	return df.loc[outlier_indices]


# 5. LOF Method
def lof_test(df, column, n_neighbors=20, contamination=0.1, normalize=False, dropna=True):
	lof_df = df[column].copy()
	lof_df = lof_df.dropna() if dropna else lof_df.fillna(0)
	if normalize:
		lof_df[column] = np.log1p(lof_df[column])
	lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
	labels = lof.fit_predict(lof_df)
	outlier_indices = lof_df[labels == -1].index
	return df.loc[outlier_indices]


# 6. Ensemble Outlier Detection
def ensemble_outlier_detection(df, column, methods, dropna=True, threshold=None):
	if threshold is None:
		threshold = len(methods) - 2
	scores_df = df[column].copy()
	scores_df = scores_df.dropna() if dropna else scores_df.fillna(0)
	scores_df['Outlier_Score'] = 0  # Initialize combined score

	# Loop through each method
	for method_name, method_func in methods.items():
		# Get outlier indices from the method
		outlier_indices = method_func(df, column)

		# Mark outliers in the combined score
		scores_df.loc[outlier_indices.index, 'Outlier_Score'] += 1

	# Determine final outliers based on a score threshold
	final_outliers_idx = scores_df[scores_df['Outlier_Score'] > threshold].index
	final_outliers = df.loc[final_outliers_idx]

	return final_outliers


def iterative_lof_removal(data, n_neighbors=45, max_iterations=5, contamination='auto'):
	remaining_data = data.copy()
	all_outliers = []
	outlier_counts = {}

	for iteration in range(max_iterations):
		lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
		lof_labels = lof.fit_predict(remaining_data)
		outlier_indices = np.where(lof_labels == -1)[0]

		if len(outlier_indices) == 0:  # Stop if no outliers are found
			print(f"No more outliers detected at iteration {iteration + 1}.")
			break

		# Store outliers and their counts
		outliers = remaining_data[outlier_indices]
		all_outliers.append(outliers)
		outlier_counts[f"Iteration {iteration + 1}"] = len(outlier_indices)

		# Remove outliers
		remaining_data = np.delete(remaining_data, outlier_indices, axis=0)

	return remaining_data, all_outliers, outlier_counts


