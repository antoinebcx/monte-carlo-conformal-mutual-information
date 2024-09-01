import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from mcci_selector import MCCISelector
from jax import random

class MCCIExplainability:
    """
    Explainability class for the Monte Carlo Conformal Information-theoretic (MCCI) Feature Selector.

    This class provides methods to interpret and visualize the results of the MCCI feature selection process,
    including feature importance, stability analysis, and local feature importance.

    Parameters
    ----------
    mcci_selector : MCCISelector
        An instance of the MCCISelector class that has been fitted to the data.

    Attributes
    ----------
    selector : MCCISelector
        The MCCI selector instance used for explainability.

    Notes
    -----
    This class assumes that the MCCISelector has been properly fitted to the data before use.
    """

    def __init__(self, mcci_selector: MCCISelector):
        self.selector = mcci_selector

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get a DataFrame of feature names and their importance scores.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'Feature' and 'Importance', sorted by importance in descending order.

        Notes
        -----
        The importance scores are derived from the MCCI selector's computed importance scores.
        """
        return pd.DataFrame({
            'Feature': self.selector.feature_names,
            'Importance': self.selector.importance_scores.flatten()
        }).sort_values('Importance', ascending=False)

    def get_selected_features(self, n_features_to_select: int) -> list:
        """
        Get the names of the top N selected features.

        Parameters
        ----------
        n_features_to_select : int
            The number of top features to select.

        Returns
        -------
        list
            A list of feature names, ordered by importance (most important first).
        """
        flattened_importance_scores = np.ravel(self.selector.importance_scores)
        selected_indices = np.argsort(flattened_importance_scores)[-n_features_to_select:]
        return [self.selector.feature_names[i] for i in selected_indices]

    def get_threshold(self) -> float:
        """
        Get the calibrated threshold used for feature selection.

        Returns
        -------
        float
            The threshold value used in the MCCI selector.
        """
        return self.selector.threshold

    def plot_feature_importance(self, top_n: int = 10):
        """
        Plot a bar chart of the top N most important features.

        Parameters
        ----------
        top_n : int, optional (default=10)
            The number of top features to display in the plot.

        Notes
        -----
        This method uses seaborn to create a bar plot and displays it using matplotlib.
        """
        importance_df = self.get_feature_importance().head(top_n)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importance')
        plt.show()

    def get_feature_stability(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 10, subsample_ratio: float = 0.8):
        """
        Calculate feature stability across multiple subsamples of the data.

        Parameters
        ----------
        X : np.ndarray
            The input features, shape (n_samples, n_features).
        y : np.ndarray
            The target values, shape (n_samples,).
        n_iterations : int, optional (default=10)
            The number of subsampling iterations to perform.
        subsample_ratio : float, optional (default=0.8)
            The ratio of samples to use in each subsample.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'Feature' and 'Stability', sorted by stability in descending order.

        Notes
        -----
        Stability is calculated by running the MCCI selector on multiple subsamples of the data
        and averaging the resulting importance scores.
        """
        n_samples, n_features = X.shape
        subsample_size = int(n_samples * subsample_ratio)
        stability_matrix = np.zeros((n_iterations, n_features))

        for i in range(n_iterations):
            subsample_indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_subsample, y_subsample = X[subsample_indices], y[subsample_indices]

            selector = MCCISelector(alpha=self.selector.alpha, num_samples=self.selector.num_samples, random_state=self.selector.random_state)
            selector.fit(X_subsample, y_subsample)
            _ = selector.transform(X_subsample, self.selector.n_features)

            stability_matrix[i] = selector.importance_scores.flatten()

        stability_scores = np.mean(stability_matrix, axis=0)
        return pd.DataFrame({
            'Feature': self.selector.feature_names,
            'Stability': stability_scores
        }).sort_values('Stability', ascending=False)

    def plot_feature_stability(self, stability_df: pd.DataFrame, top_n: int = 10):
        """
        Plot a bar chart of the stability of the top N features.

        Parameters
        ----------
        stability_df : pd.DataFrame
            A DataFrame containing feature stability scores, as returned by get_feature_stability().
        top_n : int, optional (default=10)
            The number of top features to display in the plot.

        Notes
        -----
        This method uses seaborn to create a bar plot and displays it using matplotlib.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Stability', y='Feature', data=stability_df.head(top_n))
        plt.title(f'Top {top_n} Feature Stability')
        plt.show()

    def local_feature_importance(self, X_instance, n_samples=1000, kernel_width=0.75):
        """
        Compute local feature importance for a specific instance.

        Parameters
        ----------
        X_instance : array-like
            The instance for which to compute local feature importance, shape (n_features,).
        n_samples : int, optional (default=1000)
            The number of perturbed samples to generate for local importance estimation.
        kernel_width : float, optional (default=0.75)
            The width of the kernel used in the local importance calculation.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'Feature' and 'Local_Importance', sorted by local importance in descending order.

        Notes
        -----
        This method uses a local surrogate model to estimate feature importance in the vicinity of the given instance.
        It generates perturbed samples around the instance and computes weighted importance scores.

        Raises
        ------
        ValueError
            If the selector hasn't been fitted before calling this method.
        """
        if not hasattr(self.selector, 'X_fitted'):
            raise ValueError("The selector hasn't been fitted yet. Call 'fit' before using this method.")

        X_instance = np.array(X_instance).reshape(1, -1)

        perturbed_samples = np.random.normal(
            loc=X_instance, 
            scale=np.std(self.selector.X_fitted, axis=0) * 0.1, 
            size=(n_samples, self.selector.n_features)
        )

        distances = cdist(X_instance, perturbed_samples, metric='euclidean').flatten()
        weights = np.exp(-(distances ** 2) / kernel_width ** 2)

        local_importance_scores = []
        for i in range(n_samples):
            self.selector.rng, subkey = random.split(self.selector.rng)
            score = self.selector._compute_importance_scores(
                self.selector.conformity_scores, 
                self.selector.threshold, 
                subkey
            )
            local_importance_scores.append(score)

        local_importance_scores = np.array(local_importance_scores)

        weighted_scores = np.average(local_importance_scores, axis=0, weights=weights)

        weighted_scores = weighted_scores.flatten()

        return pd.DataFrame({
            'Feature': self.selector.feature_names,
            'Local_Importance': weighted_scores
        }).sort_values('Local_Importance', ascending=False)

    def plot_local_feature_importance(self, local_importance_df, top_n=10):
        """
        Plot a bar chart of the top N locally important features.

        Parameters
        ----------
        local_importance_df : pd.DataFrame
            A DataFrame containing local feature importance scores, as returned by local_feature_importance().
        top_n : int, optional (default=10)
            The number of top features to display in the plot.

        Notes
        -----
        This method uses seaborn to create a bar plot and displays it using matplotlib.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Local_Importance', y='Feature', data=local_importance_df.head(top_n))
        plt.title(f'Top {top_n} Locally Important Features')
        plt.xlabel('Local Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()