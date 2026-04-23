"""Shared I/O helpers for nested cross-validation runners.

Provides data loading, checkpoint management, and input path resolution
used by both the flat and hierarchical CV runner scripts.
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
