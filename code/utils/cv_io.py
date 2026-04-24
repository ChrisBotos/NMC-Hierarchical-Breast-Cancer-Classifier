"""Shared I/O helpers for nested cross-validation runners.

Provides data loading, checkpoint management, input path resolution,
and per-fold diagnostic saving used by both the flat and hierarchical
CV runner scripts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.constants import GENOMIC_COLUMNS
from utils.paths import PROJECT_DIR, _find_or_create_run_dir


def load_cv_data(input_path, clinical_path):
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


def checkpoint_path(data_dir, pipeline_name, repeat_seed):
    """Return the path for a fold-level checkpoint file.

    Args:
        data_dir (Path): The phase data directory.
        pipeline_name (str): Pipeline identifier.
        repeat_seed (int): Repeat seed number.

    Returns:
        Path: Checkpoint JSON path.
    """
    return data_dir / f"checkpoint_{pipeline_name}_r{repeat_seed}.json"


def csv_path(data_dir, pipeline_name, repeat_seed):
    """Return the path for the final fold results CSV.

    Args:
        data_dir (Path): The phase data directory.
        pipeline_name (str): Pipeline identifier.
        repeat_seed (int): Repeat seed number.

    Returns:
        Path: Final CSV path.
    """
    return data_dir / f"fold_results_{pipeline_name}_r{repeat_seed}.csv"


def load_checkpoint(ckpt_path, log):
    """Load an existing checkpoint if present.

    Args:
        ckpt_path (Path): Path to checkpoint JSON.
        log (logging.Logger): Logger instance.

    Returns:
        list[dict]: Previously completed fold results, or empty list.
    """
    if not ckpt_path.exists():
        return []
    with open(ckpt_path) as f:
        data = json.load(f)
    n_folds = len(data["fold_results"])
    log.info("Resuming from checkpoint: %d folds already completed.", n_folds)
    return data["fold_results"]


def save_checkpoint(ckpt_path, fold_results, pipeline_name, repeat_seed,
                    pipeline_key="pipeline"):
    """Write fold results to a checkpoint file.

    Args:
        ckpt_path (Path): Path to checkpoint JSON.
        fold_results (list[dict]): Completed fold results so far.
        pipeline_name (str): Pipeline identifier.
        repeat_seed (int): Repeat seed number.
        pipeline_key (str): JSON key name for the pipeline identifier.
            The flat runner uses "pipeline" (default), the hierarchical
            runner uses "stage2_pipeline".
    """
    payload = {
        pipeline_key: pipeline_name,
        "repeat": repeat_seed,
        "n_completed": len(fold_results),
        "fold_results": fold_results,
    }
    with open(ckpt_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def resolve_merged_input(run_name, input_path):
    """Resolve the input merged TSV path with fallback logic.

    Checks the run directory's preprocessing handoff first, then falls
    back to the legacy results/data/preprocessing_phase/ path.

    Args:
        run_name (str): Name of the experiment run.
        input_path (Path or None): Explicit input path from CLI, or None.

    Returns:
        Path: Resolved input path.
    """
    if input_path is not None:
        return input_path

    # Try to find it inside the run directory.
    run_dir = _find_or_create_run_dir(run_name)
    run_path = run_dir / "preprocessing" / "data" / "train_merged.tsv"
    if run_path.exists():
        return run_path

    # Legacy fallback.
    return PROJECT_DIR / "results" / "data" / "preprocessing_phase" / "train_merged.tsv"


def get_selector_scores(selector):
    """Extract feature scores from a fitted KW or EN selector.

    KruskalWallisSelector stores H-statistics in `scores_`, while
    ElasticNetSelector stores summed absolute coefficients in
    `importances_`. This helper normalises the interface.

    Args:
        selector: A fitted KruskalWallisSelector or ElasticNetSelector.

    Returns:
        np.ndarray: Per-feature scores of shape (n_features,).
    """
    if hasattr(selector, "scores_"):
        return selector.scores_
    if hasattr(selector, "importances_"):
        return selector.importances_
    raise AttributeError(
        f"Selector {type(selector).__name__} has neither "
        f"'scores_' nor 'importances_' attribute."
    )


def save_inner_cv_results(fold_dir, fold_idx, cv_results):
    """Save GridSearchCV cv_results_ to a CSV file.

    Args:
        fold_dir (Path): Directory for this pipeline/repeat combination.
        fold_idx (int): 1-based outer fold index.
        cv_results (dict): The cv_results_ attribute from GridSearchCV.
    """
    df = pd.DataFrame(cv_results)
    # Keep only the most useful columns, in a readable order.
    keep_cols = []
    # Parameter columns.
    param_cols = sorted(c for c in df.columns if c.startswith("param_"))
    keep_cols.extend(param_cols)
    # Score summary columns.
    for col in ["mean_test_score", "std_test_score", "rank_test_score"]:
        if col in df.columns:
            keep_cols.append(col)
    # Per-split scores.
    split_cols = sorted(
        (c for c in df.columns if c.startswith("split") and "test_score" in c),
        key=lambda c: int(c.split("split")[1].split("_")[0]),
    )
    keep_cols.extend(split_cols)
    # Timing columns.
    for col in ["mean_fit_time", "mean_score_time"]:
        if col in df.columns:
            keep_cols.append(col)

    df = df[[c for c in keep_cols if c in df.columns]]
    out_path = fold_dir / f"fold{fold_idx}_inner_cv.csv"
    df.to_csv(out_path, index=False)


def save_fold_features_flat(fold_dir, fold_idx, feature_names, selector):
    """Save full feature rankings for a flat (single-stage) fold.

    Args:
        fold_dir (Path): Directory for this pipeline/repeat combination.
        fold_idx (int): 1-based outer fold index.
        feature_names (list[str]): All region name strings.
        selector: The fitted selector from the best estimator.
    """
    scores = get_selector_scores(selector)
    selected_set = set(selector.indices_.tolist())

    df = pd.DataFrame({
        "feature_name": feature_names,
        "score": np.round(scores, 6),
        "selected": [i in selected_set for i in range(len(feature_names))],
    })
    # Sort by score descending for readability.
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    out_path = fold_dir / f"fold{fold_idx}_features.csv"
    df.to_csv(out_path, index=False)


def save_fold_features_hierarchical(fold_dir, fold_idx, feature_names,
                                    stage1_selector, stage2_selector):
    """Save full feature rankings for both stages of a hierarchical fold.

    Args:
        fold_dir (Path): Directory for this pipeline/repeat combination.
        fold_idx (int): 1-based outer fold index.
        feature_names (list[str]): All region name strings.
        stage1_selector: Fitted KruskalWallisSelector from Stage 1.
        stage2_selector: Fitted selector (KW or EN) from Stage 2.
    """
    s1_scores = get_selector_scores(stage1_selector)
    s1_selected = set(stage1_selector.indices_.tolist())

    s2_scores = get_selector_scores(stage2_selector)
    s2_selected = set(stage2_selector.indices_.tolist())

    df = pd.DataFrame({
        "feature_name": feature_names,
        "stage1_kw_score": np.round(s1_scores, 6),
        "stage1_selected": [i in s1_selected for i in range(len(feature_names))],
        "stage2_score": np.round(s2_scores, 6),
        "stage2_selected": [i in s2_selected for i in range(len(feature_names))],
    })
    # Sort by stage2 score descending as the primary interest.
    df = df.sort_values("stage2_score", ascending=False).reset_index(drop=True)

    out_path = fold_dir / f"fold{fold_idx}_features.csv"
    df.to_csv(out_path, index=False)
