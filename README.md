# Monte Carlo Conformal Information-theoretic (MCCI) Feature Selection

## Summary

MCCI is a feature selection method combining Monte Carlo conformal prediction and mutual information.

It computes mutual information between features and the target, then uses Monte Carlo sampling to estimate a conformal threshold and feature importance scores.
This method calibrates the selection process using a specified significance level (alpha) and aims to identify the most relevant features across multiple random samples.

The Monte Carlo conformal prediction implementation has been inspired by [Google DeepMind's approach](https://github.com/google-deepmind/uncertain_ground_truth/blob/main/monte_carlo.py).

MCCI approaches the selective power of Boruta and RFECV, with a computation time close to a simple Lasso or Elastic Net. Exact comparison figures will soon be uploaded.

## Usage

### Feature selection
```
from mcci_selector import MCCISelector

selector = MCCISelector(alpha=0.2, num_samples=100, random_state=RANDOM_SEED)

n_features_to_select = 10
X_selected = selector.fit_transform(X, y, n_features_to_select)
```

### Explainability

##### Feature importance
```
from mcci_explainability import MCCIExplainability

explainer = MCCIExplainability(selector)

importance_df = explainer.get_feature_importance()
print("Feature Importance:")
print(importance_df)

# or plot
explainer.plot_feature_importance(top_n=15)
```

##### Threshold
```
threshold = explainer.get_threshold()
print("Calibrated Threshold:", threshold)
```

##### Feature stability
```
stability_df = explainer.get_feature_stability(X, y, n_iterations=20, subsample_ratio=0.8)
print("Feature Stability:")
print(stability_df)

# or plot
explainer.plot_feature_stability(stability_df, top_n=15)
```

##### Local feature importance
```
# select an instance for local interpretation
instance_index = 0
X_instance = X[instance_index]

local_importance = explainer.local_feature_importance(X_instance)
print("Local Feature Importance:")
print(local_importance)

# or plot
explainer.plot_local_feature_importance(local_importance, top_n=15)
```


