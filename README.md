# Monte Carlo Conformal Information-theoretic (MCCI) Feature Selection

### Summary

MCCI is a novel feature selection method combining Monte Carlo conformal prediction and mutual information.

It computes mutual information between features and the target, then uses Monte Carlo sampling to estimate a conformal threshold and feature importance scores.
This method calibrates the selection process using a specified significance level (alpha) and aims to identify the most relevant features across multiple random samples.

It approaches the selective power of Boruta and RFECV, with a computation time close to a simple Lasso or Elastic Net. Exact comparison figures will soon be uploaded.

### Ackowledgements

The Monte Carlo conformal prediction implementation was heavily inspired by [DeepMind's approach](https://github.com/google-deepmind/uncertain_ground_truth/blob/main/monte_carlo.py).
