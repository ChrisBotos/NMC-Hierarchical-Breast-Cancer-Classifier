"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: select_stage1_params.py.
Description:
    Quick parameter selection for Stage 1 (KW+RF, HER2+ vs rest, k=5).
    Since BA1 = 1.0 for every reasonable RF configuration, this script
    evaluates robustness via prediction margins (mean probability gap
    between the correct class and the other class). The configuration
    with the highest minimum margin across folds is selected.

Usage:
    python3 code/select_stage1_params.py --name final_hierarchical

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, rich.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import rich.traceback
from rich.console import Console
from rich.table import Table

rich.traceback.install()

# Allow imports from the code directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.cv_components import KruskalWallisSelector
from utils.cv_io import load_cv_data
from utils.paths import _find_or_create_run_dir

console = Console()


def run_stage1_grid(X, y_binary, rf_grid, n_repeats=10, n_folds=5,
                    seed=42):
    """Evaluate RF hyperparameter combos for Stage 1 via repeated CV.

    For each combo, runs repeated stratified k-fold and records
    balanced accuracy and prediction margin (mean probability gap
    between the correct class and the incorrect class).

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        y_binary (np.ndarray): Binary labels (1=HER2+, 0=rest).
        rf_grid (list[dict]): List of RF param dicts to evaluate.
        n_repeats (int): Number of CV repeats.
        n_folds (int): Number of folds per repeat.
        seed (int): Base random seed.

    Returns:
        pd.DataFrame: Results with columns for each param, mean_ba,
            mean_margin, min_margin, std_margin.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=seed,
    )

    results = []

    for params in rf_grid:
        fold_bas = []
        fold_margins = []

        for train_idx, test_idx in cv.split(X, y_binary):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]

            pipe = Pipeline([
                ("selector", KruskalWallisSelector(k=5)),
                ("clf", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                    **params,
                )),
            ])

            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)
            y_pred = pipe.predict(X_test)

            ba = balanced_accuracy_score(y_test, y_pred)
            fold_bas.append(ba)

            # Compute margin: P(correct) - P(incorrect) for each sample.
            margins = []
            for i, true_label in enumerate(y_test):
                p_correct = proba[i, true_label]
                p_wrong = proba[i, 1 - true_label]
                margins.append(p_correct - p_wrong)
            fold_margins.append(np.min(margins))

        row = dict(params)
        row["mean_ba"] = np.mean(fold_bas)
        row["mean_margin"] = np.mean(fold_margins)
        row["min_margin"] = np.min(fold_margins)
        row["std_margin"] = np.std(fold_margins)
        row["p5_margin"] = np.percentile(fold_margins, 5)
        results.append(row)

    return pd.DataFrame(results)


def main():
    """Select robust RF hyperparameters for Stage 1."""
    parser = argparse.ArgumentParser(
        description="Select Stage 1 RF hyperparameters by prediction margin.",
    )
    parser.add_argument(
        "--name", default="final_hierarchical",
        help="Run name (default: final_hierarchical).",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Number of CV repeats (default: 10).",
    )
    args = parser.parse_args()

    run_dir = _find_or_create_run_dir(args.name)

    # Load data.
    merged_path = run_dir / "preprocessing" / "data" / "train_merged.tsv"
    clinical_path = (
        Path(__file__).resolve().parent.parent / "data" / "Train_clinical.tsv"
    )
    X, y, le, feature_names, sample_names = load_cv_data(
        merged_path, clinical_path,
    )

    # Binary labels: HER2+ = 1, rest = 0.
    her2_idx = list(le.classes_).index("HER2+")
    y_binary = (y == her2_idx).astype(int)
    console.print(
        f"Loaded {len(y)} samples ({y_binary.sum()} HER2+, "
        f"{(~y_binary.astype(bool)).sum()} rest), {X.shape[1]} features.",
    )

    # Small RF grid focused on robustness.
    n_estimators_vals = [100, 500, 1000]
    max_depth_vals = [None, 5, 10]
    min_samples_leaf_vals = [1, 3, 5]
    max_features_vals = ["sqrt", None]

    rf_grid = []
    for n_est, depth, leaf, feat in product(
        n_estimators_vals, max_depth_vals,
        min_samples_leaf_vals, max_features_vals,
    ):
        rf_grid.append({
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": feat,
        })

    console.print(f"Testing {len(rf_grid)} RF configurations "
                  f"({args.repeats} repeats x 5 folds each).")

    results_df = run_stage1_grid(
        X, y_binary, rf_grid, n_repeats=args.repeats,
    )

    # Sort by min_margin descending (most robust first).
    results_df = results_df.sort_values("min_margin", ascending=False)

    # Display top 10 results.
    table = Table(title="Stage 1 RF Parameter Selection (sorted by min margin)")
    table.add_column("n_estimators", style="cyan")
    table.add_column("max_depth", style="cyan")
    table.add_column("min_samples_leaf", style="cyan")
    table.add_column("max_features", style="cyan")
    table.add_column("Mean BA", style="green")
    table.add_column("Mean Margin", style="yellow")
    table.add_column("Min Margin", style="red")
    table.add_column("5th %ile Margin", style="magenta")

    for _, row in results_df.head(10).iterrows():
        depth_str = str(row["max_depth"]) if row["max_depth"] is not None else "None"
        feat_str = str(row["max_features"]) if row["max_features"] is not None else "None"
        table.add_row(
            str(int(row["n_estimators"])),
            depth_str,
            str(int(row["min_samples_leaf"])),
            feat_str,
            f"{row['mean_ba']:.4f}",
            f"{row['mean_margin']:.4f}",
            f"{row['min_margin']:.4f}",
            f"{row['p5_margin']:.4f}",
        )

    console.print(table)

    # Print recommended config.
    best = results_df.iloc[0]
    console.print()
    console.print("[bold]Recommended Stage 1 RF configuration:[/bold]")
    console.print(f"  n_estimators:    {int(best['n_estimators'])}")
    depth_val = best["max_depth"]
    console.print(f"  max_depth:       {depth_val if depth_val is not None else 'None'}")
    console.print(f"  min_samples_leaf: {int(best['min_samples_leaf'])}")
    feat_val = best["max_features"]
    console.print(f"  max_features:    {feat_val if feat_val is not None else 'None'}")
    console.print(f"  class_weight:    balanced")
    console.print(f"  Min margin:      {best['min_margin']:.4f}")
    console.print(f"  Mean BA:         {best['mean_ba']:.4f}")


if __name__ == "__main__":
    main()
