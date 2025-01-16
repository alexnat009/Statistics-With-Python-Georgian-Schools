import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

withDifferentBuildings = True
name = 'withDifferentBuildings' if withDifferentBuildings else 'withoutDifferentBuildings'
df = pd.read_pickle(f'../../Data/modified/finalDatabase/{name}/finalDatabase_{name}.pkl')

cols_Admin = [
	'დირექცია, ადმინისტრაციულ-ტექნიკური-პერსონალის შრომის ანაზღაურება',
	'მ.შ. დირექცია, ადმინისტრაციულ-ტექნიკური-პერსონალის, ინკლიზიური განათლების მხარდამჭერი სპეციალისტები თანამდებობრივი სარგო (ხელფასი)',
	'მ.შ. დირექცია, ადმინისტრაციულ-ტექნიკური-პერსონალის პრემია'
]
cols_Spec_Ed = [
	'სპეც მასწავლებელთა შრომის ანაზღაურება',
	'მ.შ. სპეც მასწავლებლის შრომის ანზღაურება (საათობრივი დატვირთვა და სხვა დანამატები, სქემის ფარგლებში გათვალისწინებული დანამატების გარდა)',
	'მ.შ. სპეც მასწავლებლის სქემის ფარგლებში გათვალისწინებული დანამატების ოდენობა',
	'მ.შ. სპეც მასწავლებელთა პრემია',
]


# Abstracted regression function
def perform_regression(df, feature_col, target_cols, model, test_size=0.2, random_state=42, print_prediction=False,
					   save_predictions=False, print_scores=False):
	# Filter valid rows (non-zero feature values and no missing targets)
	filtered_df = df.loc[
		df[feature_col].ne(0) & df[target_cols].notna().all(axis=1),
		[feature_col] + target_cols
	]

	# Define features (X) and targets (Y)
	X = filtered_df[feature_col].to_numpy().reshape(-1, 1)
	Y = filtered_df[target_cols].to_numpy()

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
	# Fit the model
	model.fit(X_train, y_train)

	# Predict on the test set
	y_pred = model.predict(X_test)

	# Scoring
	scores = {
		"MSE": mean_squared_error(y_test, y_pred),
		"R2": r2_score(y_test, y_pred)
	}
	if print_scores:
		print("Scores:", scores)

	if print_prediction:
		nan_rows = df.loc[df[filtered_df.columns].isna().any(axis=1)]

		if not nan_rows.empty:
			nan_X = nan_rows[feature_col].to_numpy().reshape(-1, 1)

			nan_predictions = model.predict(nan_X).round()
			print(f'Features:{feature_col}, Predicted Target: {target_cols}')
			for feature, target in zip(nan_X, nan_predictions):
				print(f"Feature: {feature}, Predicted Target: {target}")
			if save_predictions:
				df.loc[nan_rows.index, target_cols] = nan_predictions

	return {
		"model": model,
		"X_train": X_train,
		"y_train": y_train,
		"X_test": X_test,
		"y_test": y_test,
		"y_pred": y_pred,
		"scores": scores
	}


# Example usage with administrative data


ridge_reg = Ridge(alpha=1, fit_intercept=True, solver="cholesky")

# Perform regression for Admin columns
admin_results = perform_regression(
	df,
	feature_col=cols_Admin[0],
	target_cols=cols_Admin[1:],
	model=ridge_reg,
	print_prediction=True,
	print_scores=False,
	save_predictions=True,
)

# Perform regression for Special Education columns
spec_results = perform_regression(
	df,
	feature_col=cols_Spec_Ed[0],
	target_cols=cols_Spec_Ed[1:],
	model=ridge_reg,
	print_prediction=True,
	print_scores=False,
	save_predictions=True

)

# df.to_pickle(f'../../Data/modified/finalDatabase/{name}/finalDatabase_withRegressionValues_{name}.pkl')
# df.to_excel(f'../../Data/modified/finalDatabase/{name}/finalDatabase_withRegressionValues_{name}.xlsx', index=False)

# Grid search for best parameters
# # Finding best score/parameters
# param_grid = {
# 	"alpha": [0.1, 1, 10, 100, 1000, 10000],  # Regularization strength
# 	"solver": ["auto", "svd", "cholesky", "lsqr", "saga"],  # Solvers to try
# }
#
# grid_search = GridSearchCV(
# 	estimator=ridge_reg,
# 	param_grid=param_grid,
# 	scoring="r2",  # Use negative MSE for regression
# 	cv=5,  # 5-fold cross-validation
# 	verbose=1,
# 	n_jobs=-1,  # Use all available processors
# )
#
# # Perform regression for Admin columns
# print('Administrative')
# grid_admin_results = perform_regression(
# 	df,
# 	feature_col=cols_Admin[0],
# 	target_cols=cols_Admin[1:],
# 	model=grid_search,
# 	print_prediction=True,
# 	print_scores=True
# )
#
# # Best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Mean cross-validated Score:", grid_search.best_score_, '\n')
#
# # Perform regression for Special Education columns
# print('Special ed')
# grid_spec_results = perform_regression(
# 	df,
# 	feature_col=cols_Spec_Ed[0],
# 	target_cols=cols_Spec_Ed[1:],
# 	model=grid_search,
# 	print_scores=True,
# 	print_prediction=True
# )
#
# # Best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score (Negative MSE):", grid_search.best_score_)
