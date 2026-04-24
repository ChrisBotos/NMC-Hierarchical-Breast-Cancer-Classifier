"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: hierarchical_nested_cv_v2_runner.py.
Description:
    Enhanced hierarchical nested cross-validation runner (v2). Extends the
    v1 runner with three principled modifications applied uniformly:
      1. Cost-sensitive NMC (class_weight='balanced' via log-weight bias).
      2. Stage 1 threshold calibration via inner CV.
      3. K-ensemble and pipeline-ensemble variants.

    Supports 7 pipeline variants for Stage 2: the 4 original baselines
    (now with cost-sensitive NMC), k-ensemble averaging over fixed k values,
    a pipeline ensemble (KW k-ensemble + EN+NMC), and k-grid (restricted
    k set with GridSearchCV).

    Each invocation runs one Stage 2 pipeline for one repeat seed. Jobs
    can be parallelised via SLURM array (7 pipelines x 50 repeats = 350).
    Supports fold-level checkpointing for crash recovery. Computes metrics
    with and without suspected mislabel samples for sensitivity analysis.

Usage:
    python3 code/hierarchical_nested_cv_v2_runner.py --pipeline kw_nmc --repeat 1 --config local_v2
    python3 code/hierarchical_nested_cv_v2_runner.py --pipeline kw_nmc_kens --repeat 3 --config server_v2
    python3 code/hierarchical_nested_cv_v2_runner.py --pipeline nmc_ensemble --repeat 1 --config local_v2

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
from utils.constants import V2_GRIDSEARCH_PIPELINES, V2_PIPELINE_NAMES
from utils.cv_components import (
    KruskalWallisSelector,
    NearestCentroidWithProba,
)
from utils.cv_config import build_pipeline, build_v2_stage2_pipeline
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


def calibrate_stage1_threshold(stage1_pipe, X_train, y_train_s1,
                               threshold_range, inner_folds, repeat_seed,
                               log):
    """Find the optimal Stage 1 probability threshold via inner CV.

    For each candidate threshold, runs inner stratified k-fold CV on the
    binary HER2+ vs rest problem. The Stage 1 pipeline is re-fitted on
    each inner train split, then the threshold is applied to predict_proba
    on the inner test split. The threshold with the highest mean balanced
    accuracy across inner folds is selected.

    When all thresholds achieve the same BA (expected when Stage 1 separation
    is perfect), defaults to 0.5.

    Args:
        stage1_pipe (Pipeline): Fitted Stage 1 KW+RF pipeline (will be
            cloned and re-fitted on inner splits).
        X_train (np.ndarray): Outer fold training features.
        y_train_s1 (np.ndarray): Binary labels (1=HER2+, 0=rest).
        threshold_range (list[float]): Candidate thresholds to evaluate.
        inner_folds (int): Number of inner CV folds.
        repeat_seed (int): Seed for inner CV splitting.
        log (logging.Logger): Logger instance.

    Returns:
        float: The selected threshold.
    """
    from sklearn.base import clone

    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True,
        random_state=300 + repeat_seed,
    )

    # Track mean BA per threshold across inner folds.
    threshold_scores = {t: [] for t in threshold_range}

    for inner_train, inner_test in inner_cv.split(X_train, y_train_s1):
        pipe_clone = clone(stage1_pipe)
        pipe_clone.fit(X_train[inner_train], y_train_s1[inner_train])
        proba = pipe_clone.predict_proba(X_train[inner_test])
        # proba[:, 1] = P(HER2+).
        p_her2 = proba[:, 1]
        y_inner_true = y_train_s1[inner_test]

        for t in threshold_range:
            y_pred_t = (p_her2 >= t).astype(int)
            ba = balanced_accuracy_score(y_inner_true, y_pred_t)
            threshold_scores[t].append(ba)

    # Select threshold with highest mean BA.
    mean_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
    best_threshold = max(mean_scores, key=mean_scores.get)
    best_ba = mean_scores[best_threshold]

    # Check if all thresholds tied (perfect separation).
    all_scores = list(mean_scores.values())
    all_tied = all(abs(s - all_scores[0]) < 1e-10 for s in all_scores)

    if all_tied:
        best_threshold = 0.5
        log.info(
            "    Stage 1 threshold: all %d candidates tied at BA=%.4f; "
            "defaulting to 0.5.",
            len(threshold_range), best_ba,
        )
    else:
        log.info(
            "    Stage 1 threshold: %.2f (BA=%.4f, range %.4f-%.4f).",
            best_threshold, best_ba, min(all_scores), max(all_scores),
        )

    return best_threshold


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
    stage2_pipe = build_v2_stage2_pipeline(
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


"""Hierarchical Nested CV v2 Runner"""


def run_single_repeat_v2(X, y, le, feature_names, sample_names,
                         stage2_pipeline_name, repeat_seed, stage2_grid,
                         config, log, ckpt_path=None, prior_folds=None,
                         details_dir=None):
    """Run hierarchical outer CV for one Stage 2 pipeline and one repeat (v2).

    Enhanced version with threshold calibration, cost-sensitive NMC,
    k-ensemble/pipeline-ensemble dispatch, and mislabel sensitivity analysis.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded 3-class labels.
        le (LabelEncoder): Fitted encoder (HER2+=0, HR+=1, Triple Neg=2).
        feature_names (list[str]): Region name strings.
        sample_names (list[str]): Sample column names matching row order of X.
        stage2_pipeline_name (str): One of the V2_PIPELINE_NAMES.
        repeat_seed (int): Random seed for this repeat.
        stage2_grid (dict): Hyperparameter grid for Stage 2 (or None for
            ensemble pipelines).
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.
        ckpt_path (Path or None): Checkpoint file path.
        prior_folds (list[dict] or None): Previously completed folds.
        details_dir (Path or None): Root directory for per-fold diagnostics.

    Returns:
        list[dict]: One dict per outer fold with hierarchical results.
    """
    cv_cfg = config["cv"]
    outer_folds = cv_cfg.get("outer_folds", 5)
    inner_folds = cv_cfg.get("inner_folds", 5)
    n_completed = len(prior_folds) if prior_folds else 0

    # Stage 1 threshold calibration parameters.
    s1_cfg = config.get("stage1", {})
    threshold_range = s1_cfg.get("threshold_range", [0.5])
    s1_inner_folds = s1_cfg.get("inner_folds", 5)

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

        # --- Stage 1 threshold calibration ---
        best_threshold = calibrate_stage1_threshold(
            stage1_pipe, X_train, y_train_s1,
            threshold_range, s1_inner_folds, repeat_seed, log,
        )

        # Apply calibrated threshold to test set.
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

        if stage2_pipeline_name in V2_GRIDSEARCH_PIPELINES:
            # Standard GridSearchCV path.
            proba_s2, best_s2, s2_cv_results, s2_best_params, en_converged = (
                run_stage2_gridsearch(
                    X_train_s2, y_train_s2, X_test, stage2_pipeline_name,
                    stage2_grid, inner_folds, repeat_seed, config, log,
                )
            )
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

            if stage2_pipeline_name in V2_GRIDSEARCH_PIPELINES and s2_cv_results is not None:
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
    """Parse command-line arguments for the v2 hierarchical CV runner.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical nested CV v2 runner. Stage 1 (KW+RF k=5) separates "
            "HER2+. Stage 2 (--pipeline) discriminates HR+ from Triple Neg. "
            "Supports 7 pipeline variants including k-ensemble and pipeline "
            "ensemble. One (pipeline, repeat) per invocation."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=V2_PIPELINE_NAMES,
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
        default="local_v2",
        help=(
            "Config file path or bare name. Bare names resolve to "
            "config_files/<name>.yaml. (default: local_v2)."
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
    """Entry point: run hierarchical nested CV v2 with checkpointing.

    Stage 1 is always KW+RF on binary HER2+ vs rest labels with threshold
    calibration. Stage 2 uses the pipeline specified by --pipeline.
    """
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Set up run directory and logging.
    tag = f"v2_{args.pipeline}_r{args.repeat}"
    fig_dir, data_dir, log_dir, run_dir = get_run_dirs_no_replace(
        args.name, "hierarchical_nested_cv_v2",
    )
    log, console = setup_logging(
        "hierarchical_nested_cv_v2_runner", tag=tag, log_dir=log_dir,
    )

    log.info("Run: %s", run_dir.name)
    log.info("Config: %s", config["_config_path"])
    log.info("Stage 1: kw_rf (fixed, binary HER2+ vs rest, threshold calibrated)")
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
    details_dir = run_dir / "hierarchical_nested_cv_v2" / "fold_details"

    # Run hierarchical nested CV v2.
    job_start = time.perf_counter()

    fold_results = run_single_repeat_v2(
        X, y, le, feature_names, sample_names,
        args.pipeline, args.repeat,
        stage2_grid, config, log,
        ckpt_path=ckpt, prior_folds=prior_folds,
        details_dir=details_dir,
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
        run_dir, "hierarchical_nested_cv_v2_runner",
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
