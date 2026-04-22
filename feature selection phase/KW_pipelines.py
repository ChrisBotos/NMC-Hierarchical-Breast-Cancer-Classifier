"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: KW_pipelines.py.
Description:
    Kruskal-Wallis feature selection with NMC and RF classifiers.
    Nested stratified repeated k-fold cross-validation.

Usage:
    python3 "feature selection phase/KW_pipelines.py"

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, rich.
"""

"""Imports and Configuration"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

rich.traceback.install()

# Resolve project paths from this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Set up logging with rich console and file output.
console = Console()
LOG_DIR = PROJECT_DIR / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, show_time=False, show_path=False),
        logging.FileHandler(LOG_DIR / "KW_pipelines.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

"""Default Hyperparameter Grids"""

# NMC only tunes the number of KW-selected features.
NMC_PARAM_GRID = {
    "kw__k": [5, 20],
}

# RF tunes KW features plus tree ensemble parameters.
RF_PARAM_GRID = {
    "kw__k": [5, 20],
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5],
}

# Default repeat seeds for the outer CV repetitions.
DEFAULT_REPEAT_SEEDS = [1, 2, 3]

# Default data paths derived from the project root.
DEFAULT_TRAIN_MERGED = PROJECT_DIR / "results" / "data" / "preprocessing_phase" / "train_merged.tsv"
DEFAULT_CLINICAL = PROJECT_DIR / "data" / "Train_clinical.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "data" / "KW_pipelines"


"""Kruskal-Wallis Feature Selector"""


class KruskalWallisSelector(BaseEstimator, TransformerMixin):
    """Selects the top-k features ranked by Kruskal-Wallis H-statistic.

    For each feature, the KW test compares distributions across all classes.
    Features with the highest H-statistics are retained.

    Args:
        k (int): Number of top features to select.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        """Compute KW H-statistics and select the top-k feature indices.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Class labels of shape (n_samples,).

        Returns:
            KruskalWallisSelector: The fitted selector.

        Raises:
            ValueError: If k is less than 1 or exceeds the number of features.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        n_features = X.shape[1]
        if self.k > n_features:
            raise ValueError(
                f"k={self.k} is larger than the number of features ({n_features})."
            )

        scores = []

        for i in range(n_features):
            feature = X[:, i]
            groups = [feature[y == cls] for cls in np.unique(y)]
            stat, _ = kruskal(*groups)
            scores.append(stat)

        self.scores_ = np.array(scores)
        self.indices_ = np.argsort(self.scores_)[::-1][: self.k]
        return self

    def transform(self, X):
        """Reduce X to the selected top-k features.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Reduced feature matrix of shape (n_samples, k).

        Raises:
            AttributeError: If the selector has not been fitted yet.
        """
        if not hasattr(self, "indices_"):
            raise AttributeError("KruskalWallisSelector is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X[:, self.indices_]


"""Data Loading"""


def load_data(train_merged_path=None, clinical_path=None):
    """Load the merged CN data and clinical labels, returning arrays for CV.

    Reads the pre-merged training data and clinical subtype labels,
    constructs feature names from genomic coordinates, transposes the
    data to samples-as-rows format, and encodes labels as integers.

    Args:
        train_merged_path (Path): Path to the merged training TSV file.
            Defaults to the preprocessing phase output.
        clinical_path (Path): Path to the clinical labels TSV file.
            Defaults to the project data directory.

    Returns:
        tuple: A tuple of (X, y, le, X_df) where:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Encoded class labels of shape (n_samples,).
            le (LabelEncoder): Fitted label encoder for inverse transforms.
            X_df (pd.DataFrame): Feature matrix as a DataFrame with named columns.
    """
    if train_merged_path is None:
        train_merged_path = DEFAULT_TRAIN_MERGED
    if clinical_path is None:
        clinical_path = DEFAULT_CLINICAL

    train_merged = pd.read_csv(train_merged_path, sep="\t")
    clinical = pd.read_csv(clinical_path, sep="\t")

    meta_cols = ["Chromosome", "Start", "End", "Nclone"]
    sample_cols = [c for c in train_merged.columns if c not in meta_cols]

    # Build feature names from genomic coordinates.
    feature_names = (
        "chr"
        + train_merged["Chromosome"].astype(str)
        + "_"
        + train_merged["Start"].astype(str)
        + "_"
        + train_merged["End"].astype(str)
    )

    X_df = train_merged[sample_cols].copy()
    X_df.index = feature_names
    # Transpose so that samples are rows and features are columns.
    X_df = X_df.T

    clinical = clinical.set_index("Sample")
    y_series = clinical.loc[X_df.index, "Subgroup"]

    le = LabelEncoder()
    y = le.fit_transform(y_series)
    X = X_df.to_numpy(dtype=float)

    return X, y, le, X_df


"""Pipeline Construction"""


def make_kw_pipeline_and_grid(model_name, random_state=42):
    """Create a sklearn Pipeline and hyperparameter grid for the given model.

    Builds a pipeline with KW feature selection and the specified classifier.
    For NMC, a StandardScaler is included before KW selection. For RF, no
    scaling is applied since tree-based models are scale-invariant.

    Args:
        model_name (str): Either 'nmc' for Nearest Mean Classifier or 'rf'
            for Random Forest.
        random_state (int): Random seed for the RF classifier. Ignored for NMC.

    Returns:
        tuple: A tuple of (pipeline, param_grid) where:
            pipeline (Pipeline): The sklearn Pipeline object.
            param_grid (dict): The hyperparameter grid for GridSearchCV.

    Raises:
        ValueError: If model_name is not 'nmc' or 'rf'.
    """
    if model_name == "nmc":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kw", KruskalWallisSelector()),
            ("clf", NearestCentroid()),
        ])
        param_grid = NMC_PARAM_GRID

    elif model_name == "rf":
        # RF does not need standard scaling.
        pipeline = Pipeline([
            ("kw", KruskalWallisSelector()),
            ("clf", RandomForestClassifier(random_state=random_state)),
        ])
        param_grid = RF_PARAM_GRID

    else:
        raise ValueError("model_name must be 'nmc' or 'rf'.")

    return pipeline, param_grid


"""Nested Cross-Validation"""


def run_repeated_nested_cv(X, y, model_name, repeat_seeds, output_dir):
    """Run repeated nested stratified k-fold CV for a single pipeline.

    For each repeat, a fresh pipeline is constructed so that the RF
    random_state is tied to the repeat seed. The inner loop tunes
    hyperparameters via GridSearchCV with balanced accuracy scoring.
    The outer loop evaluates the best configuration on held-out folds.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Encoded class labels of shape (n_samples,).
        model_name (str): Either 'nmc' or 'rf'.
        repeat_seeds (list[int]): Random seeds for each CV repetition.
        output_dir (Path): Directory to save the results CSV.

    Returns:
        dict: Summary dictionary containing per-repeat and aggregate results,
            including mean and std of balanced accuracy across all outer folds.
    """
    repeat_results = []
    all_outer_scores = []
    # Collect per-fold rows for the structured CSV output.
    csv_rows = []

    for seed in track(repeat_seeds, description="Repeats..."):
        log.info("=" * 40)
        log.info(f"Repeat with seed {seed}")
        log.info("=" * 40)

        # Build a fresh pipeline per repeat so RF gets the correct seed.
        pipeline, param_grid = make_kw_pipeline_and_grid(model_name, random_state=seed)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + seed)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
            refit=True,
        )

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        outer_scores = []
        outer_best_params = []

        for fold_idx, (train_idx, test_idx) in track(
            enumerate(outer_cv.split(X, y), start=1),
            total=5,
            description="  Outer folds...",
        ):
            X_outer_train, X_outer_test = X[train_idx], X[test_idx]
            y_outer_train, y_outer_test = y[train_idx], y[test_idx]

            grid.fit(X_outer_train, y_outer_train)

            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = best_pipeline.predict(X_outer_test)
            bal_acc = balanced_accuracy_score(y_outer_test, y_pred)

            outer_scores.append(bal_acc)
            outer_best_params.append(best_params)

            # Determine the number of features selected by KW.
            n_features_selected = best_params.get("kw__k", None)

            csv_rows.append({
                "pipeline": f"KW+{model_name.upper()}",
                "repeat": seed,
                "outer_fold": fold_idx,
                "balanced_accuracy": bal_acc,
                "best_params": str(best_params),
                "n_features_selected": n_features_selected,
            })

            log.info(f"Repeat {seed}, outer fold {fold_idx}")
            log.info(f"  Best params: {best_params}")
            log.info(f"  Inner CV best score: {grid.best_score_:.4f}")
            log.info(f"  Outer test balanced accuracy: {bal_acc:.4f}")

        repeat_mean = np.mean(outer_scores)
        repeat_std = np.std(outer_scores)

        repeat_results.append({
            "seed": seed,
            "outer_scores": outer_scores,
            "mean_balanced_accuracy": repeat_mean,
            "std_balanced_accuracy": repeat_std,
            "best_params_per_fold": outer_best_params,
        })

        all_outer_scores.extend(outer_scores)

        log.info(f"Repeat {seed} summary")
        log.info(f"  Outer fold balanced accuracies: {outer_scores}")
        log.info(f"  Repeat mean balanced accuracy: {repeat_mean:.4f}")
        log.info(f"  Repeat std balanced accuracy: {repeat_std:.4f}")

    # Aggregate results across all repeats.
    repeat_means = [r["mean_balanced_accuracy"] for r in repeat_results]

    summary = {
        "repeat_results": repeat_results,
        "repeat_mean_balanced_accuracies": repeat_means,
        "mean_of_repeat_means": float(np.mean(repeat_means)),
        "std_of_repeat_means": float(np.std(repeat_means)),
        "all_outer_scores": all_outer_scores,
        "mean_all_outer_scores": float(np.mean(all_outer_scores)),
        "std_all_outer_scores": float(np.std(all_outer_scores)),
        "csv_rows": csv_rows,
    }

    return summary


"""Argument Parsing"""


def parse_args():
    """Parse command-line arguments for the KW pipelines script.

    Returns:
        argparse.Namespace: Parsed arguments with n_repeats, input, clinical,
            and output_dir fields.
    """
    parser = argparse.ArgumentParser(
        description="KW feature selection + NMC/RF nested CV pipelines.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of CV repetitions (default: 3).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TRAIN_MERGED,
        help="Path to merged training TSV.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=DEFAULT_CLINICAL,
        help="Path to clinical labels TSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV results.",
    )
    return parser.parse_args()


"""Main Execution"""


if __name__ == "__main__":
    args = parse_args()

    # Use the first n_repeats seeds from the default list, extending if needed.
    if args.n_repeats <= len(DEFAULT_REPEAT_SEEDS):
        repeat_seeds = DEFAULT_REPEAT_SEEDS[: args.n_repeats]
    else:
        repeat_seeds = list(range(1, args.n_repeats + 1))

    # Ensure output directory exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    X, y, le, X_df = load_data(
        train_merged_path=args.input,
        clinical_path=args.clinical,
    )
    log.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    log.info(f"Classes: {le.classes_}")

    # Collect all CSV rows across both pipelines.
    all_csv_rows = []

    '''KW + NMC Pipeline'''

    log.info("")
    log.info("=" * 50)
    log.info("Running KW + NMC pipeline")
    log.info("=" * 50)

    results_nmc = run_repeated_nested_cv(X, y, "nmc", repeat_seeds, args.output_dir)
    all_csv_rows.extend(results_nmc["csv_rows"])

    log.info("")
    log.info("KW + NMC Final Summary")
    log.info(f"  Mean of repeat means: {results_nmc['mean_of_repeat_means']:.4f}")
    log.info(f"  Std of repeat means:  {results_nmc['std_of_repeat_means']:.4f}")

    '''KW + RF Pipeline'''

    log.info("")
    log.info("=" * 50)
    log.info("Running KW + RF pipeline")
    log.info("=" * 50)

    results_rf = run_repeated_nested_cv(X, y, "rf", repeat_seeds, args.output_dir)
    all_csv_rows.extend(results_rf["csv_rows"])

    log.info("")
    log.info("KW + RF Final Summary")
    log.info(f"  Mean of repeat means: {results_rf['mean_of_repeat_means']:.4f}")
    log.info(f"  Std of repeat means:  {results_rf['std_of_repeat_means']:.4f}")

    '''Save Structured Results'''

    # Save all outer fold results to a single CSV for downstream analysis.
    results_csv_path = args.output_dir / "nested_cv_results.csv"
    results_df = pd.DataFrame(all_csv_rows)
    results_df.to_csv(results_csv_path, index=False)
    log.info(f"Results saved to {results_csv_path}")
