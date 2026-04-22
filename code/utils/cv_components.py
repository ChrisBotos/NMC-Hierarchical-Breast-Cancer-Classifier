"""Reusable sklearn-compatible transformers for the nested CV pipeline.

Provides feature selection transformers that can be embedded inside
sklearn Pipelines and tuned via GridSearchCV, and a NearestCentroid
subclass with predict_proba for AUROC computation.
"""

import logging
import warnings

import numpy as np
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid

log = logging.getLogger(__name__)


class KruskalWallisSelector(BaseEstimator, TransformerMixin):
    """Selects the top-k features ranked by Kruskal-Wallis H-statistic.

    For each feature, the KW test compares distributions across all
    classes. Features with the highest H-statistics are retained.

    Args:
        k (int): Number of top features to select.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        """Compute KW H-statistics and identify the top-k feature indices.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Class labels of shape (n_samples,).

        Returns:
            KruskalWallisSelector: The fitted selector.

        Raises:
            ValueError: If k exceeds the number of features.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_features = X.shape[1]
        if self.k > n_features:
            raise ValueError(
                f"k={self.k} exceeds the number of features ({n_features})."
            )

        unique_classes = np.unique(y)
        scores = np.empty(n_features)

        for i in range(n_features):
            groups = [X[y == cls, i] for cls in unique_classes]
            stat, _ = kruskal(*groups)
            scores[i] = stat

        self.scores_ = scores
        self.ranking_ = np.argsort(scores)[::-1]
        self.indices_ = self.ranking_[: self.k]
        return self

    def transform(self, X):
        """Reduce X to the selected top-k features.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Reduced matrix of shape (n_samples, k).
        """
        X = np.asarray(X, dtype=float)
        return X[:, self.indices_]


class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """Feature selector using Elastic Net logistic regression coefficients.

    Fits a multinomial logistic regression with elasticnet penalty,
    then selects the top_k features ranked by the sum of absolute
    coefficients across all classes.

    Args:
        C (float): Inverse regularisation strength.
        l1_ratio (float): L1 vs L2 mixing parameter (0=Ridge, 1=Lasso).
        top_k (int): Number of top features to retain.
        max_iter (int): Maximum SAGA solver iterations.
        random_state (int): Random seed for the solver.
    """

    def __init__(self, C=1.0, l1_ratio=0.5, top_k=10, max_iter=10000,
                 random_state=42):
        self.C = C
        self.l1_ratio = l1_ratio
        self.top_k = top_k
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit Elastic Net and identify top_k features by coefficient magnitude.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Class labels of shape (n_samples,).

        Returns:
            ElasticNetSelector: The fitted selector.

        Raises:
            ValueError: If top_k exceeds the number of features.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_features = X.shape[1]
        if self.top_k > n_features:
            raise ValueError(
                f"top_k={self.top_k} exceeds the number of features "
                f"({n_features})."
            )

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            multi_class="multinomial",
            C=self.C,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=1,
        )

        # Surface convergence warnings via the logging system.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    log.warning(
                        "SAGA did not converge (C=%.4g, l1_ratio=%.2f, "
                        "max_iter=%d).",
                        self.C,
                        self.l1_ratio,
                        self.max_iter,
                    )

        self.model_ = model
        self.coef_ = model.coef_

        # Rank features by summed absolute coefficients across all classes.
        importance = np.abs(self.coef_).sum(axis=0)
        self.importances_ = importance
        self.ranking_ = np.argsort(importance)[::-1]
        self.indices_ = self.ranking_[: self.top_k]

        return self

    def transform(self, X):
        """Reduce X to the selected top_k features.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Reduced matrix of shape (n_samples, top_k).
        """
        X = np.asarray(X, dtype=float)
        return X[:, self.indices_]


class NearestCentroidWithProba(NearestCentroid):
    """NearestCentroid extended with predict_proba via softmax of distances.

    Computes pairwise Euclidean distances from each sample to every class
    centroid, then converts to probabilities using a softmax over negative
    distances. This enables AUROC computation for NMC pipelines.
    """

    def predict_proba(self, X):
        """Estimate class probabilities via softmax of negative distances.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        distances = pairwise_distances(X, self.centroids_)
        # Softmax with numerical stability (subtract row max).
        neg_dist = -distances
        neg_dist -= neg_dist.max(axis=1, keepdims=True)
        exp_vals = np.exp(neg_dist)
        proba = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return proba
