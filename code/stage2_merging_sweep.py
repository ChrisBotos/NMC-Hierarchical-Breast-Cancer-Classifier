"""Stage 2 (HR+ vs TN) KW+NMC across different
fixed merging thresholds r in {0.7, 0.8, 0.9, no_merging}.

Tests whether the merge threshold r=0.8 (chosen with HER2+ in the picture)
is actually optimal for Stage 2. 10 repeats of stratified 5-fold outer CV,
inner 5-fold CV to tune k.

Usage:
    python3 code/stage2_merging_sweep.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.cv_components import NearestCentroidWithProba

# Resolved from this file so the sweep runs from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Robust KW selector (handles constant-within-class features in small folds).
# ---------------------------------------------------------------------------


class RobustKWSelector(BaseEstimator, TransformerMixin):
    """KW feature selector that assigns H=0 to degenerate features.

    Args:
        k (int): Number of top features to select.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        """Compute KW H-statistics and identify top-k features.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Class labels.

        Returns:
            RobustKWSelector: The fitted selector.
        """
        from scipy.stats import kruskal as kw_test

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]
        effective_k = min(self.k, n_features)

        unique_classes = np.unique(y)
        scores = np.zeros(n_features)
        for i in range(n_features):
            groups = [X[y == cls, i] for cls in unique_classes]
            try:
                stat, _ = kw_test(*groups)
                scores[i] = stat
            except ValueError:
                scores[i] = 0.0

        self.scores_ = scores
        self.ranking_ = np.argsort(scores)[::-1]
        self.indices_ = self.ranking_[:effective_k]
        return self

    def transform(self, X):
        """Reduce X to selected features.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Reduced matrix.
        """
        return np.asarray(X, dtype=float)[:, self.indices_]


# ---------------------------------------------------------------------------
# Region merging (from preprocessing_phase.py, without rich progress bars).
# ---------------------------------------------------------------------------


def merge_regions(cn_df, sample_cols, threshold):
    """Merge adjacent correlated regions into consensus segments.

    Args:
        cn_df (pd.DataFrame): Copy-number data sorted by Chromosome and Start.
        sample_cols (list[str]): Sample column names.
        threshold (float): Pearson r threshold for merging.

    Returns:
        dict: Merge map - segment index (str) to list of raw row indices.
    """
    data = cn_df[sample_cols].values.astype(float)
    chroms = cn_df["Chromosome"].values
    chrom_order = sorted(cn_df["Chromosome"].unique())

    all_chains = []
    for chrom in chrom_order:
        chrom_idx = np.where(chroms == chrom)[0]
        if len(chrom_idx) == 0:
            continue

        current_chain = [int(chrom_idx[0])]
        for k in range(1, len(chrom_idx)):
            i_prev = chrom_idx[k - 1]
            i_curr = chrom_idx[k]
            r = np.corrcoef(data[i_prev], data[i_curr])[0, 1]
            if np.isnan(r):
                r = 0.0
            if r > threshold:
                current_chain.append(int(i_curr))
            else:
                all_chains.append(current_chain)
                current_chain = [int(i_curr)]
        all_chains.append(current_chain)

    return {str(i): chain for i, chain in enumerate(all_chains)}


def build_consensus_matrix(cn_df, merge_map, sample_cols):
    """Build consensus feature matrix from a merge map.

    Args:
        cn_df (pd.DataFrame): Original copy-number data.
        merge_map (dict): Segment ID (str) to list of raw row indices.
        sample_cols (list[str]): Sample column names.

    Returns:
        np.ndarray: Consensus matrix (n_samples, n_segments).
    """
    data = cn_df[sample_cols].values.astype(float)
    n_segments = len(merge_map)
    cn_matrix = np.empty((n_segments, len(sample_cols)), dtype=float)

    for seg_id_str in sorted(merge_map.keys(), key=int):
        seg_id = int(seg_id_str)
        raw_indices = merge_map[seg_id_str]
        cn_matrix[seg_id] = np.median(data[raw_indices], axis=0)

    # Transpose to (n_samples, n_segments).
    return cn_matrix.T


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


def evaluate(X, y, k_grid, n_repeats=10, seed=42):
    """Run repeated stratified 5-fold CV with inner k tuning.

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        y (np.ndarray): Binary labels (0=HR+, 1=TN).
        k_grid (list): Values of k to search over.
        n_repeats (int): Number of outer CV repeats.
        seed (int): Random seed.

    Returns:
        tuple: (scores array, list of best k per fold).
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", RobustKWSelector()),
            ("clf", NearestCentroidWithProba()),
        ]
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=seed
    )
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    scores = []
    best_ks = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        valid_k = [k for k in k_grid if k <= X_train.shape[1]]

        gscv = GridSearchCV(
            pipe,
            {"selector__k": valid_k},
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            error_score=0.0,
        )
        gscv.fit(X_train, y_train)
        y_pred = gscv.predict(X_test)
        ba = balanced_accuracy_score(y_test, y_pred)
        scores.append(ba)
        best_ks.append(gscv.best_params_["selector__k"])

    return np.array(scores), best_ks


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    """Run the merging threshold sweep for Stage 2."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    clinical_path = os.path.join(PROJECT_ROOT, "data", "Train_clinical.tsv")
    raw_path = os.path.join(PROJECT_ROOT, "data", "Train_call.tsv")

    n_repeats = 10
    k_grid = [3, 5, 10, 20, 50]
    r_values = [0.7, 0.8, 0.9, None]  # None = no merging (raw).

    print("=" * 65)
    print("Stage 2 (HR+ vs TN) KW+NMC: Merging threshold sweep")
    print("=" * 65)

    # Load raw data.
    cn_df = pd.read_csv(raw_path, sep="\t")
    clinical = pd.read_csv(clinical_path, sep="\t")

    # Filter to HR+ and TN samples.
    s2_clinical = clinical[clinical["Subgroup"].isin(["HR+", "Triple Neg"])]
    s2_samples = s2_clinical["Sample"].tolist()
    sample_cols = [c for c in cn_df.columns if c in s2_samples]

    # Build binary labels: 0=HR+, 1=TN.
    label_map = dict(zip(s2_clinical["Sample"], s2_clinical["Subgroup"]))
    y = np.array([1 if label_map[s] == "Triple Neg" else 0 for s in sample_cols])

    print(f"\nSamples: {len(sample_cols)} (HR+={np.sum(y == 0)}, TN={np.sum(y == 1)})")
    print(f"Outer CV: 5-fold x {n_repeats} repeats = {5 * n_repeats} folds")
    print(f"Inner CV: 5-fold GridSearch over k={k_grid}")
    print(f"Thresholds: r = {[r if r else 'none' for r in r_values]}")

    # Build feature matrices for each threshold.
    datasets = {}
    for r in r_values:
        if r is None:
            X = cn_df[sample_cols].values.astype(float).T
            label = f"raw ({X.shape[1]})"
        else:
            print(f"\n  Merging at r={r}...")
            merge_map = merge_regions(cn_df, sample_cols, threshold=r)
            X = build_consensus_matrix(cn_df, merge_map, sample_cols)
            label = f"r={r} ({X.shape[1]})"
        datasets[label] = X
        print(f"  {label}: {X.shape}")

    # Run evaluations.
    results = {}
    for label, X in datasets.items():
        print(f"\nEvaluating {label}...")
        scores, best_ks = evaluate(X, y, k_grid, n_repeats)
        results[label] = (scores, best_ks)
        print(f"  Mean BA: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Summary table.
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(
        f"\n{'Threshold':<18} {'Mean BA':<12} {'Std':<10} {'Median':<10} {'Best k (mode)'}"
    )
    print("-" * 65)
    for label, (scores, best_ks) in results.items():
        mode_k = max(set(best_ks), key=best_ks.count)
        print(
            f"{label:<18} {scores.mean():<12.4f} {scores.std():<10.4f} "
            f"{np.median(scores):<10.4f} {mode_k}"
        )

    # Pairwise Wilcoxon tests vs r=0.8 (current default).
    print("\n" + "-" * 65)
    print("Pairwise Wilcoxon signed-rank tests vs r=0.8:")
    print("-" * 65)

    labels = list(results.keys())
    ref_label = [name for name in labels if "r=0.8" in name][0]
    ref_scores = results[ref_label][0]

    for label, (scores, _) in results.items():
        if label == ref_label:
            continue
        diff = scores.mean() - ref_scores.mean()
        if np.all(scores == ref_scores):
            print(f"  {label} vs {ref_label}: diff={diff:+.4f}, identical scores")
            continue
        stat, p = wilcoxon(scores, ref_scores)
        sig = "*" if p < 0.05 else "ns"
        print(f"  {label} vs {ref_label}: diff={diff:+.4f}, p={p:.4f} {sig}")

    # Per-repeat breakdown.
    print("\nPer-repeat mean BA (5 folds each):")
    header = f"{'Repeat':<8}"
    for label in labels:
        short = label.split("(")[0].strip()
        header += f"{short:<12}"
    print(header)
    print("-" * (8 + 12 * len(labels)))

    for rep in range(n_repeats):
        s, e = rep * 5, (rep + 1) * 5
        row = f"{rep + 1:<8}"
        for label in labels:
            row += f"{results[label][0][s:e].mean():<12.4f}"
        print(row)


if __name__ == "__main__":
    main()
