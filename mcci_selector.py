import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
import numpy as np

class MCCISelector(BaseEstimator, TransformerMixin):
    """
    Monte Carlo Conformal Information-theoretic (MCCI) Feature Selection.

    This class implements a novel feature selection method that combines
    conformal prediction with information-theoretic measures (mutual information). It uses Monte Carlo
    sampling to estimate feature importance and select the most relevant features.

    Parameters
    ----------
    alpha : float, optional (default=0.1)
        The significance level for the conformal prediction.
    num_samples : int, optional (default=100)
        The number of Monte Carlo samples to use for importance estimation.
    random_state : int, optional (default=None)
        The seed for the random number generator. If None, a random seed will be used.

    Attributes
    ----------
    rng : jax.random.PRNGKey
        The random number generator key.
    n_features : int
        The number of features in the input data.
    conformity_scores : jax.numpy.ndarray
        The mutual information scores between features and target.
    smooth_labels : jax.numpy.ndarray
        The smoothed class probabilities.
    X_fitted : numpy.ndarray
        The input data used for fitting.
    feature_names : list
        The names of the features.
    threshold : float
        The calibrated threshold for feature selection.
    importance_scores : jax.numpy.ndarray
        The computed importance scores for each feature.

    Notes
    -----
    This implementation uses JAX for efficient computation and JIT compilation.
    """

    def __init__(self, alpha: float = 0.2, num_samples: int = 100, random_state: int = None):
        self.alpha = alpha
        self.num_samples = num_samples
        self.random_state = random_state if random_state is not None else 0
        self.rng = random.PRNGKey(self.random_state)

    @partial(jit, static_argnums=(0, 2))
    def sample_mc_labels(self, smooth_labels: jnp.ndarray, num_examples: int, key: jnp.ndarray) -> jnp.ndarray:
        """
        Sample Monte Carlo labels based on smoothed class probabilities.

        Parameters
        ----------
        smooth_labels : jax.numpy.ndarray
            The smoothed class probabilities.
        num_examples : int
            The number of examples to sample.
        key : jax.random.PRNGKey
            The random number generator key.

        Returns
        -------
        jax.numpy.ndarray
            The sampled Monte Carlo labels.
        """
        return random.categorical(key, logits=jnp.log(smooth_labels + 1e-8), 
                                  shape=(self.num_samples, num_examples))

    @partial(jit, static_argnums=(0,))
    def mc_conformal_quantile(self, scores: jnp.ndarray, num_examples: int) -> float:
        """
        Compute the Monte Carlo conformal quantile.

        Parameters
        ----------
        scores : jax.numpy.ndarray
            The conformity scores.
        num_examples : int
            The number of examples.

        Returns
        -------
        float
            The computed conformal quantile.
        """
        quantile = (jnp.floor(self.alpha * self.num_samples * (num_examples + 1)) - self.num_samples + 1) / (num_examples * self.num_samples)
        return jnp.quantile(scores, quantile, method='midpoint')

    @partial(jit, static_argnums=(0,))
    def calibrate_mc_threshold(self, conformity_scores: jnp.ndarray, smooth_labels: jnp.ndarray, key: jnp.ndarray) -> float:
        """
        Calibrate the Monte Carlo threshold for feature selection.

        Parameters
        ----------
        conformity_scores : jax.numpy.ndarray
            The conformity scores for each feature.
        smooth_labels : jax.numpy.ndarray
            The smoothed class probabilities.
        key : jax.random.PRNGKey
            The random number generator key.

        Returns
        -------
        float
            The calibrated threshold.
        """
        num_examples = conformity_scores.shape[1]
        mc_labels = self.sample_mc_labels(smooth_labels, num_examples, key)
        mc_conformity_scores = jnp.repeat(conformity_scores, self.num_samples, axis=0)
        mc_labels_reshaped = mc_labels.reshape(-1, 1)
        true_mc_conformity_scores = mc_conformity_scores[jnp.arange(mc_labels_reshaped.shape[0]), mc_labels_reshaped.squeeze()]
        return self.mc_conformal_quantile(true_mc_conformity_scores, num_examples * self.num_samples)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCCISelector':
        """
        Fit the MCCI selector to the input data.

        Parameters
        ----------
        X : numpy.ndarray
            The input features, shape (n_samples, n_features).
        y : numpy.ndarray
            The target values, shape (n_samples,).

        Returns
        -------
        self : MCCISelector
            The fitted selector.

        Notes
        -----
        This method computes the mutual information between features and target,
        and prepares the necessary attributes for feature selection.
        """
        self.n_features = X.shape[1]
        self.conformity_scores = jnp.array(mutual_info_classif(X, y, random_state=self.random_state)).reshape(1, -1)
        unique_labels, counts = np.unique(y, return_counts=True)
        self.smooth_labels = jnp.array(counts / len(y))
        self.X_fitted = X
        self.feature_names = [f'Feature {i}' for i in range(self.n_features)]
        return self

    @partial(jit, static_argnums=(0,))
    def _compute_importance_scores(self, conformity_scores: jnp.ndarray, threshold: float, key: jnp.ndarray):
        """
        Compute importance scores for features using Monte Carlo sampling.

        Parameters
        ----------
        conformity_scores : jax.numpy.ndarray
            The conformity scores for each feature.
        threshold : float
            The threshold for feature selection.
        key : jax.random.PRNGKey
            The random number generator key.

        Returns
        -------
        jax.numpy.ndarray
            The computed importance scores for each feature.
        """
        keys = random.split(key, self.num_samples)

        @jit
        def single_sample(key):
            sampled_scores = conformity_scores + random.normal(key, conformity_scores.shape) * 0.1
            return (sampled_scores > threshold).astype(float)

        importance_scores = vmap(single_sample)(keys)
        return jnp.mean(importance_scores, axis=0)

    def transform(self, X: np.ndarray, n_features_to_select: int) -> np.ndarray:
        """
        Transform the input data by selecting the most important features.

        Parameters
        ----------
        X : numpy.ndarray
            The input features, shape (n_samples, n_features).
        n_features_to_select : int
            The number of features to select.

        Returns
        -------
        numpy.ndarray
            The transformed input data with selected features.

        Notes
        -----
        This method computes the importance scores for features and selects
        the top `n_features_to_select` features based on these scores.
        """
        self.rng, subkey = random.split(self.rng)
        self.threshold = self.calibrate_mc_threshold(self.conformity_scores, self.smooth_labels, subkey)
        self.rng, subkey = random.split(self.rng)
        self.importance_scores = self._compute_importance_scores(self.conformity_scores, self.threshold, subkey)
        selected_features = jnp.argsort(self.importance_scores)[-n_features_to_select:]
        return X[:, selected_features.tolist()]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, n_features_to_select: int) -> np.ndarray:
        """
        Fit the selector to the data and then transform it.

        This is a convenience method that calls fit and transform sequentially.

        Parameters
        ----------
        X : numpy.ndarray
            The input features, shape (n_samples, n_features).
        y : numpy.ndarray
            The target values, shape (n_samples,).
        n_features_to_select : int
            The number of features to select.

        Returns
        -------
        numpy.ndarray
            The transformed input data with selected features.
        """
        return self.fit(X, y).transform(X, n_features_to_select)