"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: hierarchical_nested_cv_runner.py.
Description:
    Hierarchical nested cross-validation runner. Uses cost-sensitive NMC
    (class_weight='balanced' via log-weight bias) and supports k-ensemble
    and pipeline-ensemble variants.

    Supports 11 pipeline variants for Stage 2: 4 base pipelines with
    cost-sensitive NMC, k-ensemble averaging over fixed k values,
    a pipeline ensemble (KW k-ensemble + EN+NMC), k-grid (restricted
    k set with GridSearchCV), standalone EN, and 3 plateau ensemble
    variants that pool inner CV scores from base pipelines to identify
    a stable set of hyperparameter combos for averaging.

    Each invocation runs one Stage 2 pipeline for one repeat seed. Jobs
    can be parallelized via SLURM array (11 pipelines x 50 repeats = 550).
    Supports fold-level checkpointing for crash recovery. Computes metrics
    with and without suspected mislabel samples for sensitivity analysis.

Usage:
    python3 code/hierarchical_nested_cv_runner.py --pipeline kw_nmc --repeat 1 --config local
    python3 code/hierarchical_nested_cv_runner.py --pipeline kw_nmc_kens --repeat 3 --config server
    python3 code/hierarchical_nested_cv_runner.py --pipeline nmc_ensemble --repeat 1 --config local

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, rich, pyyaml.
"""

"""Imports and Configuration"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rich.traceback
from rich.progress import track
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure the code/ directory is on sys.path so utils is importable.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.config_loader import get_grids, load_config
from utils.constants import (
    MAX_PLATEAU_SIZE,
    PLATEAU_ENSEMBLE_BASE,
    GRIDSEARCH_PIPELINES,
    PIPELINE_NAMES,
)
from utils.cv_components import (
    KruskalWallisSelector,
    NearestCentroidWithProba,
)
from utils.cv_config import build_pipeline, build_stage2_pipeline
from utils.cv_io import (
    checkpoint_path,
    csv_path,
    load_checkpoint,
    load_cv_data,
    resolve_merged_input,
    save_checkpoint,
    save_fold_features_hierarchical,
    save_inner_cv_results,
)
from utils.logging_setup import setup_logging
from utils.paths import DATA_DIR, get_run_dirs_no_replace, save_config

rich.traceback.install()


"""Stage 1 Threshold Calibration"""



"""Stage 2 Dispatch Functions"""


def run_stage2_gridsearch(X_train_s2, y_train_s2, X_test, pipeline_name,
                          grid, inner_folds, repeat_seed, config, log):
    """Run Stage 2 via GridSearchCV for a standard pipeline.

    Args:
        X_train_s2 (np.ndarray): Stage 2 training features.
        y_train_s2 (np.ndarray): Binary labels (0=HR+, 1=TN).
        X_test (np.ndarray): Full test set features (for probability output).
        pipeline_name (str): Pipeline identifier.
        grid (dict): Hyperparameter grid for GridSearchCV.
        inner_folds (int): Number of inner CV folds.
        repeat_seed (int): Seed for inner CV splitting.
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (proba_s2, best_estimator, cv_results, en_converged) where
            proba_s2 is shape (n_test, 2), best_estimator is the refitted
            Pipeline, cv_results is the GridSearchCV cv_results_ dict,
            and en_converged is bool or None.
    """
    stage2_pipe = build_stage2_pipeline(
        pipeline_name, random_state=repeat_seed, config=config,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True,
        random_state=200 + repeat_seed,
    )
    gscv = GridSearchCV(
        estimator=stage2_pipe,
        param_grid=grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=1,
        refit=True,
    )

    # Track EN convergence warnings.
    en_converged = None
    if pipeline_name in ("en_nmc", "en_rf"):
        en_converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gscv.fit(X_train_s2, y_train_s2)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    en_converged = False
                    log.warning("    SAGA convergence warning in %s.", pipeline_name)
    else:
        gscv.fit(X_train_s2, y_train_s2)

    best_s2 = gscv.best_estimator_
    proba_s2 = best_s2.predict_proba(X_test)

    return proba_s2, best_s2, gscv.cv_results_, gscv.best_params_, en_converged


def retrain_plateau_models(plateau_params, base_pipeline_name,
                           X_train_s2, y_train_s2, X_test,
                           repeat_seed, config, feature_names, log):
    """Retrain a fixed set of plateau models and average predictions.

    Takes a pre-computed list of plateau hyperparameter combinations
    (from compute_pooled_plateau()), builds and fits a fresh pipeline
    for each, and averages their predict_proba outputs. Also extracts
    feature selection information across all plateau models.

    Args:
        plateau_params (list[dict]): Pre-computed plateau param dicts,
            one dict per plateau combo.
        base_pipeline_name (str): Base pipeline name for building fresh
            pipelines (e.g. "kw_nmc", "en_nmc", "standalone_en").
        X_train_s2 (np.ndarray): Stage 2 training features.
        y_train_s2 (np.ndarray): Binary labels (0=HR+, 1=TN).
        X_test (np.ndarray): Full test set features.
        repeat_seed (int): Random seed for stochastic components.
        config (dict): Loaded configuration dictionary.
        feature_names (list[str]): Region name strings.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (avg_proba, n_plateau, best_params, feature_union,
                feature_intersection, feature_frequency) where avg_proba
                is shape (n_test, 2).
    """
    n_plateau = len(plateau_params)
    log.info("    Retraining %d plateau models", n_plateau)

    all_proba = []
    all_features = []

    for rank, params in enumerate(plateau_params):
        pipe = build_stage2_pipeline(
            base_pipeline_name, random_state=repeat_seed, config=config,
        )
        pipe.set_params(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X_train_s2, y_train_s2)

        proba = pipe.predict_proba(X_test)
        all_proba.append(proba)

        # Extract selected features.
        if "selector" in pipe.named_steps:
            selector = pipe.named_steps["selector"]
            selected = {feature_names[i] for i in selector.indices_}
        else:
            # Standalone EN: features with non-zero coefficients.
            clf = pipe.named_steps["clf"]
            nonzero_mask = np.any(clf.coef_ != 0, axis=0)
            selected = {feature_names[i] for i, nz in enumerate(nonzero_mask) if nz}

        all_features.append(selected)

        if rank < 3 or rank == n_plateau - 1:
            log.info(
                "      [%d/%d] n_features=%d, params=%s",
                rank + 1, n_plateau, len(selected), params,
            )

    avg_proba = np.mean(all_proba, axis=0)

    # Compute feature union, intersection, and frequency.
    feature_union = set()
    feature_intersection = all_features[0].copy() if all_features else set()
    feature_frequency = {}

    for feat_set in all_features:
        feature_union |= feat_set
        feature_intersection &= feat_set
        for f in feat_set:
            feature_frequency[f] = feature_frequency.get(f, 0) + 1

    best_params = plateau_params[0]
    log.info(
        "    Plateau features: union=%d, intersection=%d",
        len(feature_union), len(feature_intersection),
    )

    return (avg_proba, n_plateau, best_params, feature_union,
            feature_intersection, feature_frequency)


def compute_pooled_plateau(base_pipeline_name, run_dir, log):
    """Read all inner_cv.csv files for a base pipeline, pool scores,
    and identify the stable plateau.

    Globs across all phase directories in the run for fold_details files
    matching the base pipeline. Pools mean_test_score across all folds
    and repeats (typically 250 observations per combo), then applies
    the standard plateau threshold (best_mean - best_std) with a cap
    at MAX_PLATEAU_SIZE.

    Args:
        base_pipeline_name (str): Base pipeline name (e.g. "kw_nmc").
        run_dir (Path): Path to the run directory.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (plateau_params_list, n_total_combos, pooled_best_score,
                pooled_best_std, threshold, n_files_read) where
                plateau_params_list is a list of dicts (one per plateau
                combo). Returns ([], 0, 0, 0, 0, 0) if no files found.
    """
    # Glob all inner_cv.csv files for the base pipeline across phases.
    pattern = f"*/fold_details/{base_pipeline_name}/r*/fold*_inner_cv.csv"
    csv_files = sorted(run_dir.glob(pattern))

    if not csv_files:
        log.warning(
            "No inner_cv.csv files found for base pipeline '%s' in %s",
            base_pipeline_name, run_dir,
        )
        return [], 0, 0.0, 0.0, 0.0, 0

    # Read all files and collect mean_test_score per combo. Files with
    # mismatched param columns or row counts are skipped (can happen
    # when old phase directories with different grids coexist in the run).
    all_scores = []
    param_cols = None
    expected_n_rows = None

    # First pass: determine the most common schema (param cols + row count).
    schema_counts = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        file_param_cols = tuple(sorted(c for c in df.columns if c.startswith("param_")))
        key = (file_param_cols, len(df))
        schema_counts[key] = schema_counts.get(key, 0) + 1

    # Use the most common schema.
    best_schema = max(schema_counts, key=schema_counts.get)
    param_cols = list(best_schema[0])
    expected_n_rows = best_schema[1]

    if len(schema_counts) > 1:
        log.warning(
            "Multiple inner_cv schemas found; using majority (%d files "
            "with %d combos). Skipping %d files with different schemas.",
            schema_counts[best_schema], expected_n_rows,
            len(csv_files) - schema_counts[best_schema],
        )

    # Second pass: collect scores from matching files.
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        file_param_cols = tuple(sorted(c for c in df.columns if c.startswith("param_")))
        if (file_param_cols, len(df)) != best_schema:
            continue
        all_scores.append(df["mean_test_score"].values)

    n_files = len(all_scores)
    if n_files == 0:
        return [], 0, 0.0, 0.0, 0.0, 0

    # Stack into (n_files, n_combos) matrix and pool.
    score_matrix = np.array(all_scores)
    pooled_mean = score_matrix.mean(axis=0)
    pooled_std = score_matrix.std(axis=0)

    n_combos = len(pooled_mean)
    best_idx = np.argmax(pooled_mean)
    best_score = float(pooled_mean[best_idx])
    best_std = float(pooled_std[best_idx])
    threshold = best_score - best_std

    plateau_mask = pooled_mean >= threshold
    plateau_indices = np.where(plateau_mask)[0]
    # Sort by pooled score descending.
    plateau_indices = plateau_indices[
        np.argsort(pooled_mean[plateau_indices])[::-1]
    ]

    if len(plateau_indices) > MAX_PLATEAU_SIZE:
        log.info(
            "Pooled plateau capped: %d -> %d combos",
            len(plateau_indices), MAX_PLATEAU_SIZE,
        )
        plateau_indices = plateau_indices[:MAX_PLATEAU_SIZE]

    # Read one matching file to extract the actual param values per combo.
    ref_df = None
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        file_param_cols = tuple(sorted(c for c in df.columns if c.startswith("param_")))
        if (file_param_cols, len(df)) == best_schema:
            ref_df = df
            break

    plateau_params_list = []
    for idx in plateau_indices:
        params = {}
        for col in param_cols:
            val = ref_df[col].iloc[idx]
            param_name = col.replace("param_", "")
            # Convert numpy types to Python native types.
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            params[param_name] = val
        plateau_params_list.append(params)

    log.info(
        "Pooled plateau: %d of %d combos from %d fold files "
        "(best=%.4f, std=%.4f, threshold=%.4f)",
        len(plateau_params_list), n_combos, n_files,
        best_score, best_std, threshold,
    )
    for i, params in enumerate(plateau_params_list):
        log.info(
            "  Plateau combo %d: score=%.4f, params=%s",
            i + 1, float(pooled_mean[plateau_indices[i]]), params,
        )

    return (plateau_params_list, n_combos, best_score,
            best_std, threshold, n_files)


def run_stage2_k_ensemble(X_train_s2, y_train_s2, X_test, feature_names,
                          k_values, repeat_seed, config, log):
    """Run Stage 2 k-ensemble: average probabilities over multiple k values.

    Fits a KW+NMC pipeline (with cost-sensitive class_weight) for each k
    in k_values, then averages predict_proba across all k pipelines.

    Args:
        X_train_s2 (np.ndarray): Stage 2 training features.
        y_train_s2 (np.ndarray): Binary labels (0=HR+, 1=TN).
        X_test (np.ndarray): Full test set features.
        feature_names (list[str]): Region name strings.
        k_values (list[int]): K values for the ensemble.
        repeat_seed (int): Random seed.
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (avg_proba, per_k_info) where avg_proba is shape (n_test, 2)
            and per_k_info is a list of dicts with per-k details.
    """
    nmc_class_weight = None
    if config and "pipelines" in config:
        nmc_class_weight = config["pipelines"].get("nmc_class_weight", None)

    all_proba = []
    per_k_info = []

    for k in k_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", KruskalWallisSelector(k=k)),
            ("clf", NearestCentroidWithProba(class_weight=nmc_class_weight)),
        ])
        pipe.fit(X_train_s2, y_train_s2)
        proba = pipe.predict_proba(X_test)
        all_proba.append(proba)

        selector = pipe.named_steps["selector"]
        selected_features = [feature_names[i] for i in selector.indices_]

        per_k_info.append({
            "k": k,
            "n_features": len(selector.indices_),
            "selected_features": ",".join(selected_features),
            "mean_max_prob": float(np.mean(np.max(proba, axis=1))),
        })
        log.info(
            "      k=%d: mean_max_prob=%.4f",
            k, per_k_info[-1]["mean_max_prob"],
        )

    avg_proba = np.mean(all_proba, axis=0)
    return avg_proba, per_k_info


def run_stage2_pipeline_ensemble(X_train_s2, y_train_s2, X_test, feature_names,
                                 k_values, en_grid, inner_folds, repeat_seed,
                                 config, log):
    """Run Stage 2 pipeline ensemble: average k-ensemble and EN+NMC proba.

    Runs the k-ensemble (KW+NMC over multiple k values) and EN+NMC
    GridSearchCV independently, then averages their probabilities.

    Args:
        X_train_s2 (np.ndarray): Stage 2 training features.
        y_train_s2 (np.ndarray): Binary labels (0=HR+, 1=TN).
        X_test (np.ndarray): Full test set features.
        feature_names (list[str]): Region name strings.
        k_values (list[int]): K values for the k-ensemble component.
        en_grid (dict): Hyperparameter grid for EN+NMC GridSearchCV.
        inner_folds (int): Number of inner CV folds.
        repeat_seed (int): Random seed.
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (avg_proba, kens_proba, en_proba, kens_info, en_cv_results,
                en_best_params, en_converged) with all component details.
    """
    # Component 1: k-ensemble.
    log.info("    Pipeline ensemble component 1: KW k-ensemble")
    kens_proba, kens_info = run_stage2_k_ensemble(
        X_train_s2, y_train_s2, X_test, feature_names,
        k_values, repeat_seed, config, log,
    )

    # Component 2: EN+NMC GridSearchCV.
    log.info("    Pipeline ensemble component 2: EN+NMC GridSearchCV")
    en_proba, en_best, en_cv_results, en_best_params, en_converged = run_stage2_gridsearch(
        X_train_s2, y_train_s2, X_test, "en_nmc",
        en_grid, inner_folds, repeat_seed, config, log,
    )

    # Average probabilities from both components.
    avg_proba = (kens_proba + en_proba) / 2.0

    # Log probability scale comparison.
    kens_mean_max = float(np.mean(np.max(kens_proba, axis=1)))
    en_mean_max = float(np.mean(np.max(en_proba, axis=1)))
    log.info(
        "    Ensemble mean_max_prob: kens=%.4f, en=%.4f, ratio=%.2f",
        kens_mean_max, en_mean_max,
        kens_mean_max / en_mean_max if en_mean_max > 0 else float("inf"),
    )

    return (avg_proba, kens_proba, en_proba, kens_info,
            en_cv_results, en_best_params, en_converged)


"""Mislabel Sensitivity Helpers"""


def compute_metrics_excluding_mislabels(y_true_labels, y_pred_combined,
                                        test_idx, mislabel_indices):
    """Compute balanced accuracy excluding suspected mislabel samples.

    Args:
        y_true_labels (np.ndarray): True class labels (string) for test set.
        y_pred_combined (np.ndarray): Predicted class labels (string).
        test_idx (np.ndarray): Indices of test samples in the original X.
        mislabel_indices (list[int]): 0-indexed positions of suspected
            mislabels in the original X.

    Returns:
        tuple: (ba_excl, n_excluded) where ba_excl is the balanced accuracy
            with mislabels removed, and n_excluded is how many were in this
            test fold. Returns (nan, 0) if no samples remain after exclusion.
    """
    keep_mask = np.array([i not in mislabel_indices for i in test_idx])
    n_excluded = int((~keep_mask).sum())

    if keep_mask.sum() == 0:
        return float("nan"), n_excluded

    ba_excl = balanced_accuracy_score(
        y_true_labels[keep_mask], y_pred_combined[keep_mask],
    )
    return round(ba_excl, 6), n_excluded


"""Hierarchical Nested CV Runner"""


def run_single_repeat(X, y, le, feature_names, sample_names,
                         stage2_pipeline_name, repeat_seed, stage2_grid,
                         config, log, ckpt_path=None, prior_folds=None,
                         details_dir=None, plateau_params=None,
                         pooled_best_score=None, pooled_best_std=None,
                         pooled_n_files=None):
    """Run hierarchical outer CV for one Stage 2 pipeline and one repeat (v2).

    Enhanced version with cost-sensitive NMC, k-ensemble/pipeline-ensemble
    dispatch, and mislabel sensitivity analysis. Stage 1 uses a fixed
    threshold of 0.5 (calibration removed due to tie-breaking artifact).

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded 3-class labels.
        le (LabelEncoder): Fitted encoder (HER2+=0, HR+=1, Triple Neg=2).
        feature_names (list[str]): Region name strings.
        sample_names (list[str]): Sample column names matching row order of X.
        stage2_pipeline_name (str): One of the PIPELINE_NAMES.
        repeat_seed (int): Random seed for this repeat.
        stage2_grid (dict): Hyperparameter grid for Stage 2 (or None for
            ensemble pipelines).
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.
        ckpt_path (Path or None): Checkpoint file path.
        prior_folds (list[dict] or None): Previously completed folds.
        details_dir (Path or None): Root directory for per-fold diagnostics.
        plateau_params (list[dict] or None): Pre-computed plateau param
            dicts for plateau ensemble pipelines. None for other pipelines.
        pooled_best_score (float or None): Best pooled inner CV score from
            compute_pooled_plateau(). Used as s2_best_inner for _pens folds.
        pooled_best_std (float or None): Std of the best pooled combo.
        pooled_n_files (int or None): Number of base fold files pooled.

    Returns:
        list[dict]: One dict per outer fold with hierarchical results.
    """
    cv_cfg = config["cv"]
    outer_folds = cv_cfg.get("outer_folds", 5)
    inner_folds = cv_cfg.get("inner_folds", 5)
    n_completed = len(prior_folds) if prior_folds else 0

    # Stage 1 uses a fixed threshold of 0.5 (no calibration). Threshold
    # calibration was removed because it consistently selected overly
    # permissive thresholds (0.10-0.20) due to a tie-breaking artifact,
    # causing ~31 misroutings per run with no offsetting benefit.

    # K-ensemble parameters.
    kens_cfg = config.get("k_ensemble", {})
    k_values = kens_cfg.get("k_values", [15, 20, 30, 50])

    # Mislabel indices for sensitivity analysis.
    mislabel_cfg = config.get("suspected_mislabels", {})
    mislabel_indices = mislabel_cfg.get("indices", [])

    # Class index mapping from the 3-class LabelEncoder.
    her2_idx = list(le.classes_).index("HER2+")
    hr_idx = list(le.classes_).index("HR+")
    tn_idx = list(le.classes_).index("Triple Neg")

    # Stage 1: binary KW+RF with k=5, no inner CV needed.
    stage1_pipe = build_pipeline(
        "kw_rf", random_state=repeat_seed, config=config,
    )
    stage1_pipe.set_params(selector__k=5)

    # Outer CV stratified on the original 3-class labels.
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=repeat_seed,
    )

    fold_results = list(prior_folds) if prior_folds else []

    for fold_idx, (train_idx, test_idx) in track(
        enumerate(outer_cv.split(X, y), start=1),
        total=outer_folds,
        description=f"  v2_{stage2_pipeline_name} r{repeat_seed}",
    ):
        # Skip folds already completed in the checkpoint.
        if fold_idx <= n_completed:
            continue

        fold_start = time.perf_counter()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Stage 1: HER2+ vs rest (binary KW+RF) ---
        y_train_s1 = (y_train == her2_idx).astype(int)
        y_test_s1 = (y_test == her2_idx).astype(int)

        stage1_pipe.fit(X_train, y_train_s1)

        # Apply fixed threshold to test set.
        best_threshold = 0.5
        proba_s1 = stage1_pipe.predict_proba(X_test)
        p_her2_test = proba_s1[:, 1]
        y_pred_s1 = (p_her2_test >= best_threshold).astype(int)
        stage1_bal_acc = balanced_accuracy_score(y_test_s1, y_pred_s1)

        s1_selector = stage1_pipe.named_steps["selector"]
        s1_n_features = len(s1_selector.indices_)
        s1_features = [feature_names[i] for i in s1_selector.indices_]

        # --- Stage 2: HR+ vs TN on the HR+/TN training subset ---
        mask_train_s2 = np.isin(y_train, [hr_idx, tn_idx])
        X_train_s2 = X_train[mask_train_s2]
        y_train_s2 = (y_train[mask_train_s2] == tn_idx).astype(int)

        # Dispatch based on pipeline type.
        en_converged = None
        s2_selector = None
        s2_best_params = None
        s2_cv_results = None
        s2_best_inner = None
        s2_best_inner_std = None
        pens_n_features_intersection = None
        pens_feature_frequency = None
        is_plateau_ensemble = stage2_pipeline_name in PLATEAU_ENSEMBLE_BASE

        if is_plateau_ensemble:
            # Pooled plateau ensemble: use pre-computed plateau params
            # (pooled across all base pipeline folds/repeats). No
            # GridSearchCV is run per fold - just retrain and average.
            base_name = PLATEAU_ENSEMBLE_BASE[stage2_pipeline_name]

            (proba_s2, n_plateau, best_plateau_params, feat_union,
             feat_inter, feat_freq) = retrain_plateau_models(
                plateau_params, base_name, X_train_s2, y_train_s2,
                X_test, repeat_seed, config, feature_names, log,
            )
            s2_best_params = {
                "ensemble": "plateau_pooled",
                "n_plateau": n_plateau,
                "n_files_pooled": pooled_n_files,
                "best": best_plateau_params,
            }
            s2_n_features = len(feat_union)
            s2_features = sorted(feat_union)
            pens_n_features_intersection = len(feat_inter)
            pens_feature_frequency = feat_freq

            # Use the pooled best score/std since no per-fold inner CV.
            s2_best_inner = pooled_best_score
            s2_best_inner_std = pooled_best_std

        elif stage2_pipeline_name in GRIDSEARCH_PIPELINES:
            # Standard GridSearchCV path.
            proba_s2, best_s2, s2_cv_results, s2_best_params, en_converged = (
                run_stage2_gridsearch(
                    X_train_s2, y_train_s2, X_test, stage2_pipeline_name,
                    stage2_grid, inner_folds, repeat_seed, config, log,
                )
            )
            if "selector" in best_s2.named_steps:
                s2_selector = best_s2.named_steps["selector"]
            # Extract inner CV score for the best configuration.
            gscv_mean_scores = s2_cv_results["mean_test_score"]
            gscv_std_scores = s2_cv_results["std_test_score"]
            best_idx = np.argmax(gscv_mean_scores)
            s2_best_inner = float(gscv_mean_scores[best_idx])
            s2_best_inner_std = float(gscv_std_scores[best_idx])

        elif stage2_pipeline_name == "kw_nmc_kens":
            # K-ensemble path.
            proba_s2, per_k_info = run_stage2_k_ensemble(
                X_train_s2, y_train_s2, X_test, feature_names,
                k_values, repeat_seed, config, log,
            )

        elif stage2_pipeline_name == "nmc_ensemble":
            # Pipeline ensemble path (k-ensemble + EN+NMC).
            en_grid = config["grids"]["en_nmc"]
            (proba_s2, kens_proba, en_proba, kens_info,
             en_cv_results_ens, en_best_params_ens, en_converged) = (
                run_stage2_pipeline_ensemble(
                    X_train_s2, y_train_s2, X_test, feature_names,
                    k_values, en_grid, inner_folds, repeat_seed, config, log,
                )
            )
            s2_best_params = {"ensemble": "kens+en_nmc"}
            s2_cv_results = en_cv_results_ens

        if not is_plateau_ensemble:
            s2_n_features = len(s2_selector.indices_) if s2_selector else 0
            s2_features = (
                [feature_names[i] for i in s2_selector.indices_]
                if s2_selector else []
            )

        # --- Diagnostic: Stage 2 accuracy on true HR+/TN test samples ---
        mask_test_hrt = np.isin(y_test, [hr_idx, tn_idx])
        if mask_test_hrt.any():
            y_test_s2 = (y_test[mask_test_hrt] == tn_idx).astype(int)
            # Use argmax of proba for Stage 2 predictions.
            y_pred_s2_diag = np.argmax(
                proba_s2[mask_test_hrt] if proba_s2.shape[0] == len(y_test)
                else proba_s2, axis=1,
            )
            # Only use the mask_test_hrt subset of proba.
            s2_proba_hrt = proba_s2[mask_test_hrt]
            y_pred_s2_diag = np.argmax(s2_proba_hrt, axis=1)
            stage2_bal_acc = balanced_accuracy_score(y_test_s2, y_pred_s2_diag)
        else:
            stage2_bal_acc = float("nan")

        # --- Combined 3-class predictions via routing ---
        y_pred_combined = np.empty(len(y_test), dtype=object)
        her2_pred_mask = (y_pred_s1 == 1)
        rest_pred_mask = ~her2_pred_mask

        y_pred_combined[her2_pred_mask] = "HER2+"

        n_routed_s2 = int(rest_pred_mask.sum())
        if rest_pred_mask.any():
            # Use argmax of Stage 2 proba for routed samples.
            s2_pred_rest = np.argmax(proba_s2[rest_pred_mask], axis=1)
            y_pred_combined[rest_pred_mask] = np.where(
                s2_pred_rest == 1, "Triple Neg", "HR+",
            )

        y_true_labels = le.inverse_transform(y_test)
        combined_bal_acc = balanced_accuracy_score(
            y_true_labels, y_pred_combined,
        )

        # --- AUROC via Bayesian probability decomposition ---
        p_her2 = proba_s1[:, 1]
        p_rest = proba_s1[:, 0]
        p_hr = p_rest * proba_s2[:, 0]
        p_tn = p_rest * proba_s2[:, 1]

        # Columns match le.classes_ order: HER2+(0), HR+(1), Triple Neg(2).
        proba_combined = np.column_stack([p_her2, p_hr, p_tn])

        auroc = roc_auc_score(
            y_test, proba_combined, multi_class="ovr", average="macro",
        )

        # --- Mislabel sensitivity analysis ---
        if mislabel_indices:
            combined_ba_excl, n_excluded = compute_metrics_excluding_mislabels(
                y_true_labels, y_pred_combined, test_idx, mislabel_indices,
            )
            # Stage 2 exclusion: only among HR+/TN test samples.
            if mask_test_hrt.any():
                test_idx_hrt = test_idx[mask_test_hrt]
                y_true_s2_labels = y_true_labels[mask_test_hrt]
                y_pred_s2_labels = np.where(
                    np.argmax(proba_s2[mask_test_hrt], axis=1) == 1,
                    "Triple Neg", "HR+",
                )
                s2_ba_excl, _ = compute_metrics_excluding_mislabels(
                    y_true_s2_labels, y_pred_s2_labels,
                    test_idx_hrt, mislabel_indices,
                )
            else:
                s2_ba_excl = float("nan")
        else:
            combined_ba_excl = float("nan")
            s2_ba_excl = float("nan")
            n_excluded = 0

        fold_elapsed = time.perf_counter() - fold_start

        fold_row = {
            "stage2_pipeline": stage2_pipeline_name,
            "repeat": repeat_seed,
            "outer_fold": fold_idx,
            "stage1_bal_acc": round(stage1_bal_acc, 6),
            "stage1_threshold": round(best_threshold, 4),
            "stage2_bal_acc": round(stage2_bal_acc, 6),
            "combined_bal_acc": round(combined_bal_acc, 6),
            "auroc_macro": round(auroc, 6),
            "stage2_best_inner": (
                round(s2_best_inner, 6) if s2_best_inner is not None
                else None
            ),
            "stage2_best_inner_std": (
                round(s2_best_inner_std, 6) if s2_best_inner_std is not None
                else None
            ),
            "stage2_best_params": (
                json.dumps(s2_best_params, default=str)
                if s2_best_params is not None else None
            ),
            "stage1_n_features": s1_n_features,
            "stage2_n_features": s2_n_features,
            "stage1_features": ",".join(s1_features),
            "stage2_features": ",".join(s2_features) if s2_features else "",
            "stage2_n_features_intersection": pens_n_features_intersection,
            "stage2_feature_frequency": (
                json.dumps(pens_feature_frequency)
                if pens_feature_frequency is not None else None
            ),
            "n_test": len(y_test),
            "n_train_s2": int(mask_train_s2.sum()),
            "n_routed_to_s2": n_routed_s2,
            "combined_bal_acc_excl": (
                round(combined_ba_excl, 6)
                if not np.isnan(combined_ba_excl) else None
            ),
            "stage2_bal_acc_excl": (
                round(s2_ba_excl, 6)
                if not np.isnan(s2_ba_excl) else None
            ),
            "n_excluded": n_excluded,
            "en_converged": en_converged,
            "test_indices": ",".join(str(i) for i in test_idx),
            "y_true": ",".join(y_true_labels),
            "y_pred": ",".join(y_pred_combined.tolist()),
            "proba_combined": json.dumps(
                np.round(proba_combined, 6).tolist(),
            ),
            "fold_seconds": round(fold_elapsed, 1),
        }
        fold_results.append(fold_row)

        # Save checkpoint after each fold.
        if ckpt_path is not None:
            save_checkpoint(
                ckpt_path, fold_results, stage2_pipeline_name, repeat_seed,
                pipeline_key="stage2_pipeline",
            )

        # Save detailed per-fold diagnostics.
        if details_dir is not None:
            fold_dir = (
                details_dir / stage2_pipeline_name / f"r{repeat_seed}"
            )
            fold_dir.mkdir(parents=True, exist_ok=True)

            if stage2_pipeline_name in GRIDSEARCH_PIPELINES and s2_cv_results is not None:
                save_inner_cv_results(fold_dir, fold_idx, s2_cv_results)
                if s2_selector is not None:
                    save_fold_features_hierarchical(
                        fold_dir, fold_idx, feature_names,
                        stage1_selector=s1_selector,
                        stage2_selector=s2_selector,
                    )

            elif stage2_pipeline_name == "nmc_ensemble" and s2_cv_results is not None:
                # Save EN+NMC inner CV results for the ensemble.
                en_fold_path = fold_dir / f"fold{fold_idx}_inner_cv_ennmc.csv"
                df_en = pd.DataFrame(s2_cv_results)
                keep = sorted(c for c in df_en.columns if c.startswith("param_"))
                for col in ["mean_test_score", "std_test_score", "rank_test_score"]:
                    if col in df_en.columns:
                        keep.append(col)
                df_en = df_en[[c for c in keep if c in df_en.columns]]
                df_en.to_csv(en_fold_path, index=False)

            elif is_plateau_ensemble:
                # Pooled plateau: no per-fold inner CV to save. The
                # plateau was computed once from base pipeline results.
                pass

        log.info(
            "  Fold %d: s1=%.4f (t=%.2f)  s2=%.4f  combined=%.4f  "
            "auroc=%.4f  excl=%.4f (%d)  (%.1fs)",
            fold_idx, stage1_bal_acc, best_threshold, stage2_bal_acc,
            combined_bal_acc, auroc,
            combined_ba_excl if not np.isnan(combined_ba_excl) else 0.0,
            n_excluded, fold_elapsed,
        )

    return fold_results


"""Argument Parsing"""


def parse_args():
    """Parse command-line arguments for the hierarchical CV runner.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical nested CV runner. Stage 1 (KW+RF k=5) separates "
            "HER2+. Stage 2 (--pipeline) discriminates HR+ from Triple Neg. "
            "Supports 7 pipeline variants including k-ensemble and pipeline "
            "ensemble. One (pipeline, repeat) per invocation."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=PIPELINE_NAMES,
        help="Stage 2 pipeline variant to evaluate.",
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
        default="local",
        help=(
            "Config file path or bare name. Bare names resolve to "
            "config_files/<name>.yaml. (default: local)."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default_run",
        help="Run name for the results directory (default: default_run).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to merged training TSV. Defaults to preprocessing "
             "handoff inside the run directory, then legacy path.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=DATA_DIR / "Train_clinical.tsv",
        help="Path to clinical labels TSV.",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Discard any existing checkpoint and start from fold 1.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Exit immediately (code 0) if the final CSV already exists.",
    )
    return parser.parse_args()


"""Main Execution"""


def main():
    """Entry point: run hierarchical nested CV with checkpointing.

    Stage 1 is always KW+RF on binary HER2+ vs rest labels with a fixed
    threshold of 0.5. Stage 2 uses the pipeline specified by --pipeline.
    """
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Set up run directory and logging.
    tag = f"{args.pipeline}_r{args.repeat}"
    fig_dir, data_dir, log_dir, run_dir = get_run_dirs_no_replace(
        args.name, "hierarchical_nested_cv",
    )
    log, console = setup_logging(
        "hierarchical_nested_cv_runner", tag=tag, log_dir=log_dir,
    )

    log.info("Run: %s", run_dir.name)
    log.info("Config: %s", config["_config_path"])
    log.info("Stage 1: kw_rf (fixed, binary HER2+ vs rest, threshold=0.5)")
    log.info("Stage 2: %s (HR+ vs Triple Neg)", args.pipeline)
    log.info("Repeat seed: %d", args.repeat)

    # Early exit if already complete.
    out_csv = csv_path(data_dir, args.pipeline, args.repeat)
    if args.skip_if_complete and out_csv.exists():
        log.info(
            "Final CSV already exists: %s. Skipping (--skip-if-complete).",
            out_csv,
        )
        return

    # Checkpoint handling.
    ckpt = checkpoint_path(data_dir, args.pipeline, args.repeat)
    prior_folds = []

    if args.force_restart:
        if ckpt.exists():
            ckpt.unlink()
            log.info("Removed existing checkpoint (--force-restart).")
    else:
        prior_folds = load_checkpoint(ckpt, log)

    # Resolve input path.
    args.input = resolve_merged_input(args.name, args.input)

    # Hyperparameter grids.
    grids = get_grids(config)
    stage2_grid = grids.get(args.pipeline, None)

    if stage2_grid is not None:
        s2_size = 1
        for v in stage2_grid.values():
            s2_size *= len(v)
        log.info("Stage 2 grid size: %d combinations", s2_size)
    else:
        log.info("Stage 2: no GridSearchCV (ensemble pipeline)")

    # Load data.
    log.info("Loading data...")
    X, y, le, feature_names, sample_names = load_cv_data(
        args.input, args.clinical,
    )
    log.info("Loaded %d samples, %d features.", X.shape[0], X.shape[1])
    log.info("Classes: %s", le.classes_.tolist())

    # Class distribution for logging.
    for cls_idx, cls_name in enumerate(le.classes_):
        n_cls = int((y == cls_idx).sum())
        log.info("  %s: %d samples", cls_name, n_cls)

    # Verify suspected mislabel indices match expected sample names.
    mislabel_cfg = config.get("suspected_mislabels", {})
    mislabel_indices = mislabel_cfg.get("indices", [])
    mislabel_names = mislabel_cfg.get("names", [])
    if mislabel_indices and mislabel_names:
        for idx, expected_name in zip(mislabel_indices, mislabel_names):
            actual_name = sample_names[idx]
            if actual_name != expected_name:
                log.error(
                    "MISLABEL INDEX MISMATCH: index %d is '%s', expected '%s'. "
                    "Aborting to prevent incorrect sensitivity analysis.",
                    idx, actual_name, expected_name,
                )
                sys.exit(1)
        log.info(
            "Mislabel indices verified: %s",
            list(zip(mislabel_indices, mislabel_names)),
        )

    # Per-fold diagnostic output directory.
    details_dir = run_dir / "hierarchical_nested_cv" / "fold_details"

    # For plateau ensembles, compute the pooled plateau from base pipeline
    # results before the fold loop. This gives a single stable plateau
    # used identically for every fold/repeat.
    plateau_params = None
    pooled_best_score = None
    pooled_best_std = None
    pooled_n_files = None

    if args.pipeline in PLATEAU_ENSEMBLE_BASE:
        base_name = PLATEAU_ENSEMBLE_BASE[args.pipeline]
        (plateau_params, n_combos, pooled_best_score,
         pooled_best_std, threshold, pooled_n_files) = (
            compute_pooled_plateau(base_name, run_dir, log)
        )
        if not plateau_params:
            log.error(
                "No base pipeline inner_cv files found for '%s'. "
                "Run the base pipeline (%s) first.",
                args.pipeline, base_name,
            )
            sys.exit(1)

    # Run hierarchical nested CV.
    job_start = time.perf_counter()

    fold_results = run_single_repeat(
        X, y, le, feature_names, sample_names,
        args.pipeline, args.repeat,
        stage2_grid, config, log,
        ckpt_path=ckpt, prior_folds=prior_folds,
        details_dir=details_dir,
        plateau_params=plateau_params,
        pooled_best_score=pooled_best_score,
        pooled_best_std=pooled_best_std,
        pooled_n_files=pooled_n_files,
    )

    job_elapsed = time.perf_counter() - job_start

    # Summarise results.
    s1_scores = [r["stage1_bal_acc"] for r in fold_results]
    s2_scores = [r["stage2_bal_acc"] for r in fold_results]
    combined_scores = [r["combined_bal_acc"] for r in fold_results]

    log.info("")
    log.info("Summary for v2_%s repeat %d:", args.pipeline, args.repeat)
    log.info(
        "  Stage 1 (HER2+ vs rest):  %.4f (+/- %.4f)",
        np.mean(s1_scores), np.std(s1_scores),
    )
    log.info(
        "  Stage 2 (HR+ vs TN):      %.4f (+/- %.4f)",
        np.nanmean(s2_scores), np.nanstd(s2_scores),
    )
    log.info(
        "  Combined (3-class):       %.4f (+/- %.4f)",
        np.mean(combined_scores), np.std(combined_scores),
    )

    # Sensitivity analysis summary.
    excl_scores = [
        r["combined_bal_acc_excl"] for r in fold_results
        if r.get("combined_bal_acc_excl") is not None
    ]
    if excl_scores:
        log.info(
            "  Combined (excl mislabels): %.4f (+/- %.4f)",
            np.mean(excl_scores), np.std(excl_scores),
        )

    log.info(
        "  Combined fold scores: %s",
        [round(s, 4) for s in combined_scores],
    )
    log.info("  Total time: %.1fs", job_elapsed)

    # Save fold results CSV.
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(out_csv, index=False)
    log.info("Fold results saved to %s", out_csv)

    # Remove checkpoint.
    if ckpt.exists():
        ckpt.unlink()
        log.info("Checkpoint removed (job complete).")

    # Save run config snapshot.
    save_config(
        run_dir, "hierarchical_nested_cv_runner",
        stage2_pipeline=args.pipeline,
        repeat=args.repeat,
        config_file=config["_config_path"],
        input_path=str(args.input),
        mean_combined_balanced_accuracy=float(np.mean(combined_scores)),
        mean_stage1_balanced_accuracy=float(np.mean(s1_scores)),
        mean_stage2_balanced_accuracy=float(np.nanmean(s2_scores)),
    )


if __name__ == "__main__":
    main()
