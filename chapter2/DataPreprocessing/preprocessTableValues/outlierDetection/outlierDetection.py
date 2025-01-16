from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from chapter2.DataPreprocessing.preprocessTableValues.outlierDetection.outlierDetectionAlgorithms import iqr_test, \
	isolation_forest_test, \
	dbscan_test, lof_test, \
	ensemble_outlier_detection, iterative_lof_removal
from chapter2.DataPreprocessing.preprocessTableValues.outlierDetection.parameterTuningForOutlierSearch import \
	visualize_pca


# withDifferentBuildings = False
# name = "withDifferentBuildings" if withDifferentBuildings else "withoutDifferentBuildings"

# df = pd.read_pickle(
# 	f'../../../Data/modified/preprocesses/revenueFormat/preprocess3_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl')
# df = pd.read_pickle(
# 	f'../../../Data/modified/preprocesses/revenueFormat/tmptmp.pkl')


# cols = [
# 	'მოსწავლეთა რაოდენობა',
# 	'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)',
# 	'ეზოს ფართობი (კვ.მ)',
# 	'სართულების რაოდენობა',
# 	'საპირფარეშო ოთახების რაოდენობა',
# 	'საკლასო ოთახების რაოდენობა',
# ]
#
# dfNumericalColumns = df[cols].copy()


def run_algorithms(df, cols):
	# Define outlier detection methods
	methods = {
		'IQR': lambda df, col: iqr_test(df, col),
		'Isolation Forest': lambda df, col: isolation_forest_test(df, col, n_estimators=136,
																  contamination=0.5, random_state=42,
																  normalize=True),
		'DBSCAN': lambda df, col: dbscan_test(df, col, eps=100, min_samples=2),
		'LOF': lambda df, col: lof_test(df, col, n_neighbors=45, contamination=0.5, normalize=True)
	}

	# 1. Z-Score Method
	# zscores = zscoreTest(dfNumericalColumns, cols, normalize=True, threshold=3)

	# 2. IQR Method
	iqr_outliers = iqr_test(df, cols)
	# 3. Isolation Forest
	iso_forest_outliers = isolation_forest_test(df, cols, n_estimators=136,
												contamination='auto', random_state=42, normalize=True,
												dropna=True)
	# 4. DBSCAN
	dbscan_outliers = dbscan_test(df, cols, eps=500, min_samples=2, dropna=True)
	# 5. LOF Parameters
	lof_outliers = lof_test(df, cols, n_neighbors=45, contamination='auto', normalize=True,
							dropna=False)
	# 6. Ensemble Outlier Detection
	ensemble_outliers = ensemble_outlier_detection(df, cols, methods, dropna=False)
	return iqr_outliers, iso_forest_outliers, dbscan_outliers, lof_outliers, ensemble_outliers


def perform_pca(data, n_components):
	tmp = data.copy()
	scaled_data = StandardScaler().fit_transform(
		tmp.fillna(0)
	)

	# Apply PCA
	pca = PCA(n_components=n_components)
	transformed_data = pca.fit_transform(scaled_data)

	# Create a DataFrame for transformed data
	columns = [f"Principal Component {i + 1}" for i in range(n_components)]
	transformed_df = pd.DataFrame(data=transformed_data, columns=columns, index=data.index)

	# Calculate explained variance
	explained_variance_ratio = pca.explained_variance_ratio_.sum() * 100

	return transformed_df, explained_variance_ratio, pca, transformed_data


# pca_2d_df, explained_variance_ratio_2d, pca_2d, transformed_data_2d = perform_pca(dfNumericalColumns, n_components=2)
# pca_3d_df, explained_variance_ratio_3d, pca_3d, transformed_data_3d = perform_pca(dfNumericalColumns, n_components=3)
# print(f'Explained variance ratio (2D): {explained_variance_ratio_2d:.2f}%')
# print(f'Explained variance ratio (3D): {explained_variance_ratio_3d:.2f}%')
#
# remaining_data_2d, all_outliers_2d, outlier_counts_2d = iterative_lof_removal(
# 	transformed_data_2d, n_neighbors=45, max_iterations=4, contamination=0.04
# )
#
# remaining_data_3d, all_outliers_3d, outlier_counts_3d = iterative_lof_removal(
# 	transformed_data_3d, n_neighbors=45, max_iterations=4, contamination=0.04
# )


def map_outliers_to_original_data(original_data, transformed_data, all_outliers):
	mapped_outliers = []

	for iteration, outliers in enumerate(all_outliers):
		# Find indices of outliers in the transformed data
		outlier_indices = [
			np.where((transformed_data == outlier).all(axis=1))[0][0]
			for outlier in outliers
		]
		# Map back to the original data
		mapped_outliers.append(original_data.iloc[outlier_indices])

	# Combine all mapped outliers into a single DataFrame
	all_mapped_outliers = pd.concat(mapped_outliers)

	return all_mapped_outliers


# all_mapped_outliers_2d = map_outliers_to_original_data(dfNumericalColumns, transformed_data_2d, all_outliers_2d)
# all_mapped_outliers_3d = map_outliers_to_original_data(dfNumericalColumns, transformed_data_3d, all_outliers_3d)
#
# visualize_pca(transformed_data_2d, outliers=None, all_outliers=None, title="PCA Projection 2D",
# 			  dimensions=2, save=False,
# 			  filename=f'../../../chapter2/graphs/PCA_2D_points_{name}.png')
# visualize_pca(transformed_data_3d, outliers=None, all_outliers=None, title="PCA Projection 3D",
# 			  dimensions=3, save=False,
# 			  filename=f'../../../chapter2/graphs/PCA_3D_points_{name}.png')
#
# unionPCA_LOF = pd.concat([all_mapped_outliers_3d, lof_outliers])
#
# unique_union_indices = unionPCA_LOF.index.unique()
# df = df.drop(unique_union_indices)


# df.to_pickle(
# 	f"../../../Data/modified/preprocesses/dropOutliers/preprocess4_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl")
# df.to_excel(
# 	f"../../../Data/modified/preprocesses/dropOutliers/preprocess4_infrastructure_condition_studentNumber_counts_revenue_join_{name}.xlsx",
# 	index=False)

def outlierDetection_main(df, output_path, withDifferentBuildings, save=False):
	cols = [
		'მოსწავლეთა რაოდენობა',
		'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)',
		'ეზოს ფართობი (კვ.მ)',
		'სართულების რაოდენობა',
		'საპირფარეშო ოთახების რაოდენობა',
		'საკლასო ოთახების რაოდენობა',
	]

	dfNumericalColumns = df[cols].copy()

	pca_2d_df, explained_variance_ratio_2d, pca_2d, transformed_data_2d = perform_pca(dfNumericalColumns,
																					  n_components=2)
	pca_3d_df, explained_variance_ratio_3d, pca_3d, transformed_data_3d = perform_pca(dfNumericalColumns,
																					  n_components=3)
	print(f'Explained variance ratio (2D): {explained_variance_ratio_2d:.2f}%')
	print(f'Explained variance ratio (3D): {explained_variance_ratio_3d:.2f}%')

	remaining_data_2d, all_outliers_2d, outlier_counts_2d = iterative_lof_removal(
		transformed_data_2d, n_neighbors=45, max_iterations=4, contamination=0.04
	)

	remaining_data_3d, all_outliers_3d, outlier_counts_3d = iterative_lof_removal(
		transformed_data_3d, n_neighbors=45, max_iterations=4, contamination=0.04
	)

	all_mapped_outliers_2d = map_outliers_to_original_data(dfNumericalColumns, transformed_data_2d, all_outliers_2d)
	all_mapped_outliers_3d = map_outliers_to_original_data(dfNumericalColumns, transformed_data_3d, all_outliers_3d)

	_, _, _, lof_outliers, _ = run_algorithms(dfNumericalColumns, cols)
	visualize_pca(transformed_data_2d, outliers=lof_outliers, all_outliers=all_outliers_2d, title="PCA Projection 2D",
				  dimensions=2, save=False,
				  filename=f'../../../chapter2/graphs/PCA_2D_points_{withDifferentBuildings}.png')
	visualize_pca(transformed_data_3d, outliers=lof_outliers, all_outliers=all_outliers_3d, title="PCA Projection 3D",
				  dimensions=3, save=False,
				  filename=f'../../../chapter2/graphs/PCA_3D_points_{withDifferentBuildings}.png')
	unionPCA_LOF = pd.concat([all_mapped_outliers_3d, lof_outliers])

	unique_union_indices = unionPCA_LOF.index.unique()
	df = df.drop(unique_union_indices)
	if save:
		df.to_pickle(output_path)
		df.to_excel(output_path.replace('.pkl', '.xlsx'), index=False)
	return df
