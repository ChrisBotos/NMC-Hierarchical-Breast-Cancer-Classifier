"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: nested_cv_2x2_runner.py.
Description:
    Unified nested cross-validation runner for the 2x2 experimental design.
    Each invocation runs one pipeline (kw_nmc, kw_rf, en_nmc, en_rf) for
    one repeat (a single seed), producing 5 outer-fold balanced accuracy
    scores. Jobs can be parallelized trivially via shell.

Usage:
    python3 code/nested_cv_2x2_runner.py --pipeline kw_nmc --repeat 1
    python3 code/nested_cv_2x2_runner.py --pipeline en_rf --repeat 3 --config production

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, rich.
"""

"""Imports and Configuration"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rich.traceback
from rich.progress import track
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Ensure the code/ directory is on sys.path so utils is importable.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.constants import GENOMIC_COLUMNS
from utils.cv_config import (
    PIPELINE_NAMES,
    PRODUCTION_GRIDS,
    PRODUCTION_REPEATS,
    TRIAL_GRIDS,
    TRIAL_REPEATS,
    build_pipeline,
)
from utils.logging_setup import setup_logging
from utils.paths import DATA_DIR, PROJECT_DIR, get_phase_dirs

rich.traceback.install()


"""Data Loading"""


def load_data(input_path, clinical_path):
    """Load merged CN data and clinical labels, returning arrays for CV.

    Reads the pre-merged training data and clinical subtype labels,
    constructs feature names from genomic coordinates, transposes to
    samples-as-rows format, and encodes labels as integers.

    Args:
        input_path (Path): Path to the merged training TSV file.
        clinical_path (Path): Path to the clinical labels TSV file.

    Returns:
        tuple: (X, y, label_encoder, feature_names) where X is a numpy
            array of shape (n_samples, n_features), y is an integer-
            encoded label array, label_encoder is the fitted LabelEncoder,
            and feature_names is a list of region name strings.
    """
    train_df = pd.read_csv(input_path, sep="\t")
    clinical_df = pd.read_csv(clinical_path, sep="\t")

    # Build region names using the underscore convention.
    feature_names = (
        "chr"
        + train_df["Chromosome"].astype(str)
        + "_"
        + train_df["Start"].astype(str)
        + "_"
        + train_df["End"].astype(str)
    ).tolist()

    sample_cols = [c for c in train_df.columns if c not in GENOMIC_COLUMNS]

    # Transpose: rows become samples, columns become regions.
    X_df = train_df[sample_cols].T
    X_df.columns = feature_names

    # Align labels to the sample order.
    clinical_df = clinical_df.set_index("Sample")
    y_series = clinical_df.loc[X_df.index, "Subgroup"]

    le = LabelEncoder()
    y = le.fit_transform(y_series)
    X = X_df.to_numpy(dtype=float)

    return X, y, le, feature_names


"""Single-Repeat Nested CV Runner"""


def run_single_repeat(X, y, feature_names, pipeline_name, repeat_seed,
                      param_grid, log):
    """Run 5-fold outer CV for one pipeline and one repeat seed.

    Constructs a fresh pipeline per repeat so stochastic components
    (RF, ElasticNet solver) use the repeat seed. The inner CV loop is
    handled entirely by GridSearchCV with balanced accuracy scoring.
    AUROC (weighted one-vs-rest) is computed as a secondary metric using
    predict_proba on the outer test fold.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded labels of shape (n_samples,).
        feature_names (list[str]): Region name strings for the columns of X.
        pipeline_name (str): One of the four pipeline identifiers.
        repeat_seed (int): Random seed for this repeat.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        log (logging.Logger): Logger instance.

    Returns:
        list[dict]: One dict per outer fold with evaluation results.
    """
    pipeline = build_pipeline(pipeline_name, random_state=repeat_seed)

    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=100 + repeat_seed,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=1,
        refit=True,
    )

    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=repeat_seed,
    )

    fold_results = []

    for fold_idx, (train_idx, test_idx) in track(
        enumerate(outer_cv.split(X, y), start=1),
        total=5,
        description=f"  {pipeline_name} repeat {repeat_seed} outer folds",
    ):
        fold_start = time.perf_counter()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_inner_score = grid_search.best_score_

        y_pred = best_pipeline.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Secondary metric: macro-averaged one-vs-rest AUROC from predict_proba.
        y_proba = best_pipeline.predict_proba(X_test)
        auroc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro",
        )

        fold_elapsed = time.perf_counter() - fold_start

        # Determine the number of features selected.
        selector = best_pipeline.named_steps["selector"]
        n_features = len(selector.indices_)

        # Identify which features were selected by name.
        selected_feature_names = [feature_names[i] for i in selector.indices_]

        fold_row = {
            "pipeline": pipeline_name,
            "repeat": repeat_seed,
            "outer_fold": fold_idx,
            "balanced_accuracy": round(bal_acc, 6),
            "auroc_macro": round(auroc, 6),
            "best_inner_score": round(best_inner_score, 6),
            "best_params": json.dumps(best_params, default=str),
            "n_features_selected": n_features,
            "selected_features": ",".join(selected_feature_names),
            "fold_seconds": round(fold_elapsed, 1),
        }
        fold_results.append(fold_row)

        log.info(
            "  Fold %d: bal_acc=%.4f  auroc=%.4f  inner_best=%.4f  "
            "n_features=%d  (%.1fs)",
            fold_idx, bal_acc, auroc, best_inner_score, n_features,
            fold_elapsed,
        )

    return fold_results


"""Argument Parsing"""


def parse_args():
    """Parse command-line arguments for the model training phase runner.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Nested CV runner for one (pipeline, repeat) job. "
            "Produces 5 outer-fold balanced accuracy scores."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=PIPELINE_NAMES,
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Repeat seed (1-indexed). Determines outer/inner CV splits.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="trial",
        choices=("trial", "production"),
        help="Hyperparameter grid size (default: trial).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "results" / "data" / "preprocessing_phase"
        / "train_merged.tsv",
        help="Path to merged training TSV.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=DATA_DIR / "Train_clinical.tsv",
        help="Path to clinical labels TSV.",
    )
    return parser.parse_args()


"""Main Execution"""


def main():
    """Entry point: load data, run nested CV, save results."""
    args = parse_args()

    # Construct a tag for logging and output directories.
    tag = f"{args.pipeline}_r{args.repeat}"
    log, console = setup_logging("nested_cv_2x2_runner", tag=tag)

    log.info("Pipeline: %s", args.pipeline)
    log.info("Repeat seed: %d", args.repeat)
    log.info("Config: %s", args.config)

    # Select the appropriate hyperparameter grid.
    grids = TRIAL_GRIDS if args.config == "trial" else PRODUCTION_GRIDS
    param_grid = grids[args.pipeline]

    # Compute grid size for logging.
    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    log.info("Grid size: %d combinations", grid_size)

    # Load data.
    log.info("Loading data...")
    X, y, le, feature_names = load_data(args.input, args.clinical)
    log.info("Loaded %d samples, %d features.", X.shape[0], X.shape[1])
    log.info("Classes: %s", le.classes_.tolist())

    # Run the nested CV for this (pipeline, repeat).
    job_start = time.perf_counter()

    fold_results = run_single_repeat(
        X, y, feature_names, args.pipeline, args.repeat, param_grid, log,
    )

    job_elapsed = time.perf_counter() - job_start

    # Summarise.
    scores = [r["balanced_accuracy"] for r in fold_results]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    log.info("")
    log.info("Summary for %s repeat %d:", args.pipeline, args.repeat)
    log.info("  Fold scores: %s", [round(s, 4) for s in scores])
    log.info("  Mean balanced accuracy: %.4f (+/- %.4f)", mean_score, std_score)
    log.info("  Total time: %.1fs", job_elapsed)

    # Save fold results CSV.
    _, out_dir = get_phase_dirs("nested_cv_2x2_runner")
    csv_name = f"fold_results_{args.pipeline}_r{args.repeat}.csv"
    csv_path = out_dir / csv_name

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(csv_path, index=False)
    log.info("Fold results saved to %s", csv_path)


if __name__ == "__main__":
    main()
