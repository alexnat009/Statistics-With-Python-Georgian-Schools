import numpy as np
from matplotlib import pyplot as plt


def find_optimal_alpha(alphas, model, X_train, y_train, X_test, y_test, plot=False, save=False, filepath=None):
	model_scores = []

	for alpha in alphas:
		model.set_params(alpha=alpha)
		model.fit(X_train, y_train)
		model_scores.append(model.score(X_test, y_test))

	# Plot the results

	if plot:
		plt.figure(figsize=(10, 6))
		plt.plot(alphas, model_scores, label="Ridge Test Score")
		plt.xscale("log")
		plt.xlabel("Alpha (log scale)")
		plt.ylabel("R^2 Score")
		plt.title("Ridge Regression Test Scores over Alpha")
		plt.axvline(x=alphas[np.argmax(model_scores)], color='red', linestyle='--', label="Optimal Alpha")
		plt.legend()
		plt.grid(True, which="both", linestyle="--", linewidth=0.5)
		if save:
			plt.savefig(filepath)
		plt.show()

	# Output the optimal alpha and its corresponding score
	optimal_alpha = alphas[np.argmax(model_scores)]
	optimal_score = max(model_scores)
	return optimal_alpha, optimal_score
