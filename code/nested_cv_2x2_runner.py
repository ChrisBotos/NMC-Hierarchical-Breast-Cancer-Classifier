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
    one repeat (a single seed), producing outer-fold balanced accuracy
    scores. Jobs can be parallelized trivially via shell.

    Supports fold-level checkpointing for crash recovery (--force-restart,
    --skip-if-complete). If a job is killed mid-run, re-launching it will
    resume from the last completed fold.

Usage:
    python3 code/nested_cv_2x2_runner.py --pipeline kw_nmc --repeat 1 --config config_files/local.yaml
    python3 code/nested_cv_2x2_runner.py --pipeline en_rf --repeat 3 --config config_files/server.yaml

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
    FLAT_PIPELINE_NAMES,
    build_pipeline,
)
from utils.cv_io import (
    checkpoint_path,
    csv_path,
    load_checkpoint,
    load_cv_data,
    resolve_merged_input,
    save_checkpoint,
    save_fold_features_flat,
    save_inner_cv_results,
)
from utils.logging_setup import setup_logging
from utils.paths import DATA_DIR, get_run_dirs_no_replace, save_config

rich.traceback.install()


"""Single-Repeat Nested CV Runner"""


def run_single_repeat(X, y, le, feature_names, pipeline_name, repeat_seed,
                      param_grid, config, log, ckpt_path=None,
                      prior_folds=None, details_dir=None):
    """Run outer CV for one pipeline and one repeat seed.

    Constructs a fresh pipeline per repeat so stochastic components
    (RF, ElasticNet solver) use the repeat seed. The inner CV loop is
    handled entirely by GridSearchCV. AUROC (weighted one-vs-rest) is
    computed as a secondary metric using predict_proba on the outer
    test fold. CV fold counts and scoring metric are read from config.

    Supports resuming from a checkpoint: if prior_folds is provided,
    folds that have already been completed are skipped. After each new
    fold, the checkpoint is updated on disk.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded labels of shape (n_samples,).
        le (LabelEncoder): Fitted label encoder for decoding class names.
        feature_names (list[str]): Region name strings for the columns of X.
        pipeline_name (str): One of the four pipeline identifiers.
        repeat_seed (int): Random seed for this repeat.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        config (dict): Loaded configuration dictionary.
        log (logging.Logger): Logger instance.
        ckpt_path (Path or None): Path to checkpoint file for incremental
            saving. If None, no checkpoint is written.
        prior_folds (list[dict] or None): Previously completed fold results
            loaded from a checkpoint. These folds are skipped.
        details_dir (Path or None): Root directory for per-fold diagnostic
            files. If provided, inner CV results and feature rankings are
            saved into fold_details/<pipeline>/r<seed>/ subdirectories.

    Returns:
        list[dict]: One dict per outer fold with evaluation results.
    """
    cv_cfg = config["cv"]
    outer_folds = cv_cfg.get("outer_folds", 5)
    inner_folds = cv_cfg.get("inner_folds", 5)
    scoring = cv_cfg.get("scoring", "balanced_accuracy")
    n_completed = len(prior_folds) if prior_folds else 0

    pipeline = build_pipeline(
        pipeline_name, random_state=repeat_seed, config=config,
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=100 + repeat_seed,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
    )

    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=repeat_seed,
    )

    # Start from checkpoint if available.
    fold_results = list(prior_folds) if prior_folds else []

    for fold_idx, (train_idx, test_idx) in track(
        enumerate(outer_cv.split(X, y), start=1),
        total=outer_folds,
        description=f"  {pipeline_name} repeat {repeat_seed} outer folds",
    ):
        # Skip folds already completed in the checkpoint.
        if fold_idx <= n_completed:
            continue

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

        # Decode integer labels back to class names for downstream analysis.
        y_true_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

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
            "y_true": ",".join(y_true_labels),
            "y_pred": ",".join(y_pred_labels),
            "fold_seconds": round(fold_elapsed, 1),
        }
        fold_results.append(fold_row)

        # Save checkpoint after each fold so progress survives crashes.
        if ckpt_path is not None:
            save_checkpoint(ckpt_path, fold_results, pipeline_name, repeat_seed)

        # Save detailed per-fold diagnostics (inner CV and feature scores).
        if details_dir is not None:
            fold_dir = details_dir / pipeline_name / f"r{repeat_seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_inner_cv_results(fold_dir, fold_idx, grid_search.cv_results_)
            save_fold_features_flat(
                fold_dir, fold_idx, feature_names, selector,
            )

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
            "Produces outer-fold balanced accuracy scores."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=FLAT_PIPELINE_NAMES,
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
        default="local",
        help=(
            "Config file path or bare name. Bare names resolve to "
            "config_files/<name>.yaml. Legacy names 'trial' and "
            "'production' map to 'local' and 'server'. "
            "(default: local)."
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
    """Entry point: load config, load data, run nested CV, save results.

    Supports fold-level checkpointing for crash recovery:
    - If --skip-if-complete is set and the final CSV exists, exits immediately.
    - If a checkpoint file exists (and --force-restart is not set), resumes
      from the last completed fold.
    - After each fold, the checkpoint is updated on disk.
    - On successful completion, the checkpoint is removed (the CSV is
      the authoritative record).
    """
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Set up run directory and logging (non-destructive for parallel jobs).
    tag = f"{args.pipeline}_r{args.repeat}"
    fig_dir, data_dir, log_dir, run_dir = get_run_dirs_no_replace(
        args.name, "nested_cv_2x2",
    )
    log, console = setup_logging("nested_cv_2x2_runner", tag=tag, log_dir=log_dir)

    log.info("Run: %s", run_dir.name)
    log.info("Config: %s", config["_config_path"])
    log.info("Pipeline: %s", args.pipeline)
    log.info("Repeat seed: %d", args.repeat)

    # Early exit if this job is already complete.
    out_csv = csv_path(data_dir, args.pipeline, args.repeat)
    if args.skip_if_complete and out_csv.exists():
        log.info("Final CSV already exists: %s. Skipping (--skip-if-complete).", out_csv)
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

    # Select the hyperparameter grid for this pipeline.
    grids = get_grids(config)
    param_grid = grids[args.pipeline]

    # Compute grid size for logging.
    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    log.info("Grid size: %d combinations", grid_size)

    # Load data.
    log.info("Loading data...")
    X, y, le, feature_names, _sample_names = load_cv_data(args.input, args.clinical)
    log.info("Loaded %d samples, %d features.", X.shape[0], X.shape[1])
    log.info("Classes: %s", le.classes_.tolist())

    # Per-fold diagnostic output directory.
    details_dir = run_dir / "nested_cv_2x2" / "fold_details"

    # Run the nested CV for this (pipeline, repeat).
    job_start = time.perf_counter()

    fold_results = run_single_repeat(
        X, y, le, feature_names, args.pipeline, args.repeat, param_grid,
        config, log, ckpt_path=ckpt, prior_folds=prior_folds,
        details_dir=details_dir,
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

    # Save fold results CSV (the authoritative record).
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(out_csv, index=False)
    log.info("Fold results saved to %s", out_csv)

    # Remove checkpoint now that the final CSV is written.
    if ckpt.exists():
        ckpt.unlink()
        log.info("Checkpoint removed (job complete).")

    # Save run config snapshot.
    save_config(
        run_dir, "nested_cv_2x2_runner",
        pipeline=args.pipeline,
        repeat=args.repeat,
        config_file=config["_config_path"],
        input_path=str(args.input),
        mean_balanced_accuracy=mean_score,
    )


if __name__ == "__main__":
    main()
