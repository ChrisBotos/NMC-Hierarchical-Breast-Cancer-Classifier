"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: hierarchical_nested_cv_2x2_runner.py.
Description:
    Hierarchical nested cross-validation runner. Stage 1 uses a fixed
    KW+RF pipeline (binary: HER2+ vs rest) to separate HER2+ samples.
    Stage 2 applies one of the four 2x2 pipelines (specified by
    --pipeline) to discriminate HR+ from Triple Negative among the
    remaining samples. Feature selection is recomputed within each
    stage and each outer fold to prevent leakage.

    Each invocation runs one Stage 2 pipeline for one repeat seed.
    Jobs can be parallelised trivially via shell (4 pipelines x N
    repeats). Supports fold-level checkpointing for crash recovery.

Usage:
    python3 code/hierarchical_nested_cv_2x2_runner.py --pipeline kw_rf --repeat 1 --config local
    python3 code/hierarchical_nested_cv_2x2_runner.py --pipeline en_nmc --repeat 3 --config server

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, rich, pyyaml.
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

# Ensure the code/ directory is on sys.path so utils is importable.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.config_loader import get_grids, load_config
from utils.cv_config import (
    PIPELINE_NAMES,
    build_pipeline,
)
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


"""Hierarchical Nested CV Runner"""


def run_single_repeat(X, y, le, feature_names, stage2_pipeline_name,
                      repeat_seed, stage2_grid, config, log,
                      ckpt_path=None, prior_folds=None,
                      details_dir=None):
    """Run hierarchical outer CV for one Stage 2 pipeline and one repeat.

    Stage 1 (HER2+ vs rest): always uses KW+RF with binary labels.
    Stage 2 (HR+ vs TN): uses the specified pipeline on the HR+/TN
    training subset. Both stages perform feature selection and
    hyperparameter tuning within each outer fold via inner CV.

    The combined 3-class predictions are obtained by routing: Stage 1
    predicts HER2+ or rest, then Stage 2 predicts HR+ or TN for the
    rest samples. AUROC is computed via Bayesian probability
    decomposition: P(HER2+) from Stage 1, P(HR+) = P(rest) * P_s2(HR+),
    P(TN) = P(rest) * P_s2(TN).

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded 3-class labels.
        le (LabelEncoder): Fitted encoder (HER2+=0, HR+=1, Triple Neg=2).
        feature_names (list[str]): Region name strings.
        stage2_pipeline_name (str): One of the four pipeline identifiers
            for Stage 2.
        repeat_seed (int): Random seed for this repeat.
        stage2_grid (dict): Hyperparameter grid for Stage 2.
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.
        ckpt_path (Path or None): Checkpoint file path.
        prior_folds (list[dict] or None): Previously completed folds.
        details_dir (Path or None): Root directory for per-fold diagnostic
            files. If provided, inner CV results and feature rankings are
            saved into fold_details/<pipeline>/r<seed>/ subdirectories.

    Returns:
        list[dict]: One dict per outer fold with hierarchical results.
    """
    cv_cfg = config["cv"]
    outer_folds = cv_cfg.get("outer_folds", 5)
    inner_folds = cv_cfg.get("inner_folds", 5)
    scoring = cv_cfg.get("scoring", "balanced_accuracy")
    n_completed = len(prior_folds) if prior_folds else 0

    # Class index mapping from the 3-class LabelEncoder.
    her2_idx = list(le.classes_).index("HER2+")
    hr_idx = list(le.classes_).index("HR+")
    tn_idx = list(le.classes_).index("Triple Neg")

    # Stage 1: binary KW+RF with k=5, no inner CV needed. In preliminary
    # hierarchical runs (3 repeats x 5 folds = 15 folds with binary
    # HER2+ vs rest KW ranking), inner CV unanimously selected k=5. The
    # HER2 amplicon on chr17q is so dominant in CN space that 5 features
    # achieve perfect binary separation (balanced accuracy = 1.0 in every
    # fold). Tuning k adds 40 unnecessary RF fits per outer fold.
    stage1_pipe = build_pipeline(
        "kw_rf", random_state=repeat_seed, config=config,
    )
    stage1_pipe.set_params(selector__k=5)

    # Stage 2: user-specified pipeline for HR+ vs TN.
    stage2_pipe = build_pipeline(
        stage2_pipeline_name, random_state=repeat_seed, config=config,
    )
    stage2_inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=200 + repeat_seed,
    )
    stage2_gscv = GridSearchCV(
        estimator=stage2_pipe,
        param_grid=stage2_grid,
        cv=stage2_inner_cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
    )

    # Outer CV stratified on the original 3-class labels.
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=repeat_seed,
    )

    fold_results = list(prior_folds) if prior_folds else []

    for fold_idx, (train_idx, test_idx) in track(
        enumerate(outer_cv.split(X, y), start=1),
        total=outer_folds,
        description=f"  hier_{stage2_pipeline_name} r{repeat_seed}",
    ):
        # Skip folds already completed in the checkpoint.
        if fold_idx <= n_completed:
            continue

        fold_start = time.perf_counter()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Stage 1: HER2+ vs rest (binary KW+RF) ---
        y_train_s1 = (y_train == her2_idx).astype(int)  # 1=HER2+, 0=rest.
        y_test_s1 = (y_test == her2_idx).astype(int)

        stage1_pipe.fit(X_train, y_train_s1)

        y_pred_s1 = stage1_pipe.predict(X_test)
        stage1_bal_acc = balanced_accuracy_score(y_test_s1, y_pred_s1)

        s1_selector = stage1_pipe.named_steps["selector"]
        s1_n_features = len(s1_selector.indices_)
        s1_features = [feature_names[i] for i in s1_selector.indices_]

        # --- Stage 2: HR+ vs TN on the HR+/TN training subset ---
        mask_train_s2 = np.isin(y_train, [hr_idx, tn_idx])
        X_train_s2 = X_train[mask_train_s2]
        # Binary encoding: 0=HR+, 1=Triple Neg.
        y_train_s2 = (y_train[mask_train_s2] == tn_idx).astype(int)

        stage2_gscv.fit(X_train_s2, y_train_s2)
        best_s2 = stage2_gscv.best_estimator_

        s2_selector = best_s2.named_steps["selector"]
        s2_n_features = len(s2_selector.indices_)
        s2_features = [feature_names[i] for i in s2_selector.indices_]

        # Diagnostic: Stage 2 accuracy on true HR+/TN test samples.
        mask_test_hrt = np.isin(y_test, [hr_idx, tn_idx])
        if mask_test_hrt.any():
            y_test_s2 = (y_test[mask_test_hrt] == tn_idx).astype(int)
            y_pred_s2_diag = best_s2.predict(X_test[mask_test_hrt])
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
            y_pred_rest = best_s2.predict(X_test[rest_pred_mask])
            y_pred_combined[rest_pred_mask] = np.where(
                y_pred_rest == 1, "Triple Neg", "HR+",
            )

        y_true_labels = le.inverse_transform(y_test)
        combined_bal_acc = balanced_accuracy_score(
            y_true_labels, y_pred_combined,
        )

        # --- AUROC via Bayesian probability decomposition ---
        # Apply both models to all test samples for probability estimates.
        # Stage 1: columns are [P(rest=0), P(HER2+=1)].
        proba_s1 = stage1_pipe.predict_proba(X_test)
        # Stage 2: columns are [P(HR+=0), P(TN=1)].
        proba_s2 = best_s2.predict_proba(X_test)

        p_her2 = proba_s1[:, 1]
        p_rest = proba_s1[:, 0]
        p_hr = p_rest * proba_s2[:, 0]
        p_tn = p_rest * proba_s2[:, 1]

        # Columns match le.classes_ order: HER2+(0), HR+(1), Triple Neg(2).
        proba_combined = np.column_stack([p_her2, p_hr, p_tn])

        auroc = roc_auc_score(
            y_test, proba_combined, multi_class="ovr", average="macro",
        )

        fold_elapsed = time.perf_counter() - fold_start

        # Inner CV score std for the best configuration (stability indicator).
        best_inner_std = stage2_gscv.cv_results_[
            "std_test_score"
        ][stage2_gscv.best_index_]

        fold_row = {
            "stage2_pipeline": stage2_pipeline_name,
            "repeat": repeat_seed,
            "outer_fold": fold_idx,
            "stage1_bal_acc": round(stage1_bal_acc, 6),
            "stage2_bal_acc": round(stage2_bal_acc, 6),
            "combined_bal_acc": round(combined_bal_acc, 6),
            "auroc_macro": round(auroc, 6),
            "stage2_best_inner": round(stage2_gscv.best_score_, 6),
            "stage2_best_inner_std": round(best_inner_std, 6),
            "stage2_best_params": json.dumps(
                stage2_gscv.best_params_, default=str,
            ),
            "stage1_n_features": s1_n_features,
            "stage2_n_features": s2_n_features,
            "stage1_features": ",".join(s1_features),
            "stage2_features": ",".join(s2_features),
            "n_test": len(y_test),
            "n_train_s2": int(mask_train_s2.sum()),
            "n_routed_to_s2": n_routed_s2,
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

        # Save detailed per-fold diagnostics (inner CV and feature scores).
        if details_dir is not None:
            fold_dir = (
                details_dir / stage2_pipeline_name / f"r{repeat_seed}"
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_inner_cv_results(fold_dir, fold_idx, stage2_gscv.cv_results_)
            save_fold_features_hierarchical(
                fold_dir, fold_idx, feature_names,
                stage1_selector=s1_selector,
                stage2_selector=s2_selector,
            )

        log.info(
            "  Fold %d: s1=%.4f  s2=%.4f  combined=%.4f  auroc=%.4f  "
            "s1_k=%d  s2_k=%d  routed=%d/%d  (%.1fs)",
            fold_idx, stage1_bal_acc, stage2_bal_acc, combined_bal_acc,
            auroc, s1_n_features, s2_n_features, n_routed_s2, len(y_test),
            fold_elapsed,
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
            "Hierarchical nested CV runner. Stage 1 (KW+RF) separates "
            "HER2+ from rest. Stage 2 (--pipeline) discriminates HR+ "
            "from Triple Negative. One (pipeline, repeat) per invocation."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=PIPELINE_NAMES,
        help="Stage 2 pipeline to evaluate for HR+ vs TN discrimination.",
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

    Stage 1 is always KW+RF on binary HER2+ vs rest labels.
    Stage 2 uses the pipeline specified by --pipeline on HR+/TN labels.
    Both stages perform feature selection and hyperparameter tuning
    within each outer fold via inner GridSearchCV.
    """
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Set up run directory and logging.
    tag = f"hier_{args.pipeline}_r{args.repeat}"
    fig_dir, data_dir, log_dir, run_dir = get_run_dirs_no_replace(
        args.name, "hierarchical_nested_cv_2x2",
    )
    log, console = setup_logging(
        "hierarchical_nested_cv_2x2_runner", tag=tag, log_dir=log_dir,
    )

    log.info("Run: %s", run_dir.name)
    log.info("Config: %s", config["_config_path"])
    log.info("Stage 1: kw_rf (fixed, binary HER2+ vs rest)")
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

    # Hyperparameter grids. Stage 1 is fixed (KW+RF, k=5, no tuning).
    # Stage 2 uses the grid for the specified pipeline.
    grids = get_grids(config)
    stage2_grid = grids[args.pipeline]

    s2_size = 1
    for v in stage2_grid.values():
        s2_size *= len(v)
    log.info("Stage 1: fixed KW+RF k=5 (no grid search)")
    log.info("Stage 2 grid size: %d combinations", s2_size)

    # Load data.
    log.info("Loading data...")
    X, y, le, feature_names, _sample_names = load_cv_data(args.input, args.clinical)
    log.info("Loaded %d samples, %d features.", X.shape[0], X.shape[1])
    log.info("Classes: %s", le.classes_.tolist())

    # Class distribution for logging.
    for cls_idx, cls_name in enumerate(le.classes_):
        n_cls = int((y == cls_idx).sum())
        log.info("  %s: %d samples", cls_name, n_cls)

    # Per-fold diagnostic output directory.
    details_dir = run_dir / "hierarchical_nested_cv_2x2" / "fold_details"

    # Run hierarchical nested CV.
    job_start = time.perf_counter()

    fold_results = run_single_repeat(
        X, y, le, feature_names, args.pipeline, args.repeat,
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
    log.info("Summary for hier_%s repeat %d:", args.pipeline, args.repeat)
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
        run_dir, "hierarchical_nested_cv_2x2_runner",
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
