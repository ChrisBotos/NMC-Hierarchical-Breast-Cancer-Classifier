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
            try:
                stat, _ = kruskal(*groups)
            except ValueError:
                # All values identical across groups; no discriminative power.
                stat = 0.0
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

    def __init__(self, C=1.0, l1_ratio=0.5, top_k=10, max_iter=10000, random_state=42):
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
                f"top_k={self.top_k} exceeds the number of features " f"({n_features})."
            )

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
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

    When class_weight='balanced', a log-weight bias is added to the negative
    distances before softmax, equivalent to adjusting class priors to inverse
    class frequency. This compensates for class imbalance without tuning.

    Args:
        metric (str): Distance metric for NearestCentroid.
        shrink_threshold (float or None): Shrinkage threshold.
        class_weight (str or None): If 'balanced', applies inverse class
            frequency log-weight bias to distances. If None, no bias
            (identical to original behavior).
    """

    def __init__(self, metric="euclidean", shrink_threshold=None, class_weight=None):
        super().__init__(metric=metric, shrink_threshold=shrink_threshold)
        self.class_weight = class_weight

    def fit(self, X, y):
        """Fit NearestCentroid and compute class weight bias if requested.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Class labels of shape (n_samples,).

        Returns:
            NearestCentroidWithProba: The fitted estimator.
        """
        super().fit(X, y)

        if self.class_weight == "balanced":
            # Compute log(n_total / (n_classes * n_per_class)) for each class.
            n_total = len(y)
            n_classes = len(self.classes_)
            log_weights = np.array(
                [np.log(n_total / (n_classes * np.sum(y == c))) for c in self.classes_]
            )
            self.log_weights_ = log_weights
        else:
            self.log_weights_ = None

        return self

    def predict_proba(self, X):
        """Estimate class probabilities via softmax of negative distances.

        When class_weight='balanced', a log-weight bias is added before
        softmax, shifting probabilities toward underrepresented classes.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        distances = pairwise_distances(X, self.centroids_)
        # Softmax with numerical stability (subtract row max).
        neg_dist = -distances

        # Apply class weight bias if fitted with class_weight='balanced'.
        if self.log_weights_ is not None:
            neg_dist = neg_dist + self.log_weights_

        neg_dist -= neg_dist.max(axis=1, keepdims=True)
        exp_vals = np.exp(neg_dist)
        proba = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict class labels using weighted probabilities.

        Overrides the parent predict() to ensure predictions are consistent
        with predict_proba() when class weights are applied.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        if self.log_weights_ is not None:
            # Use weighted probabilities for prediction.
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        # No weights: use parent's predict (identical to argmax of unweighted proba).
        return super().predict(X)
