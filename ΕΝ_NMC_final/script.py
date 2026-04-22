"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: script.py.
Description:
    Elastic Net feature selection with Nearest Mean Classifier (NMC).
    Nested stratified repeated k-fold cross-validation.

Usage:
    python3 "ΕΝ_NMC_final/script.py"

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, rich.
"""

"""Imports"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder, StandardScaler

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

"""Path Resolution"""

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

"""Logging Setup"""

rich.traceback.install()
console = Console()
LOG_DIR = PROJECT_DIR / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, show_time=False, show_path=False),
        logging.FileHandler(LOG_DIR / "EN_NMC_nested_cv.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_data(input_path, clinical_path):
    """Load and prepare aCGH copy-number data and clinical labels.

    Reads the merged copy-number call matrix and clinical subtype labels,
    aligns samples, and encodes labels as integers.

    Args:
        input_path (Path): Path to the merged copy-number TSV file.
        clinical_path (Path): Path to the clinical labels TSV file.

    Returns:
        tuple: A tuple of (X, y_encoded, label_encoder) where X is a
            DataFrame of shape (n_samples, n_features), y_encoded is a
            Series of integer-encoded subtype labels, and label_encoder
            is the fitted LabelEncoder instance.
    """
    train_df = pd.read_csv(input_path, sep="\t")
    labels_df = pd.read_csv(clinical_path, sep="\t")

    # Build region names using the underscore convention.
    names_of_region = (
        "chr" + train_df["Chromosome"].astype(str) + "_"
        + train_df["Start"].astype(str) + "_"
        + train_df["End"].astype(str)
    )

    no_array_cols = ["Chromosome", "Start", "End", "Nclone"]
    x_train_raw = train_df.drop(columns=no_array_cols).copy()
    x_train_raw.index = names_of_region

    # Transpose so rows are samples and columns are genomic regions.
    x_train_final = x_train_raw.T.copy()

    y_train = labels_df.set_index("Sample")["Subgroup"].copy()

    # Align samples present in both data and labels.
    same_samples = x_train_final.index.intersection(y_train.index)
    x_train_final = x_train_final.loc[same_samples]
    y_train = y_train.loc[same_samples]

    X = x_train_final.copy()
    y = y_train.copy()

    # Encode subtype labels as integers.
    lencoder = LabelEncoder()
    y_encoded = pd.Series(
        lencoder.fit_transform(y), index=y.index, name="target"
    )

    log.info("Classes: %s", lencoder.classes_)
    log.info("Loaded %d samples with %d features.", X.shape[0], X.shape[1])

    return X, y_encoded, lencoder


def run_inner_cv(x_outer_train, y_outer_train, param_grid, inner_cv, outer_seed):
    """Run inner cross-validation to select the best hyperparameters.

    For each parameter combination in the grid, fits an Elastic Net
    logistic regression for feature selection, selects the top-k features
    by absolute coefficient magnitude, then trains a NearestCentroid
    classifier and evaluates balanced accuracy on the inner validation fold.

    Args:
        x_outer_train (pd.DataFrame): Training features for the outer fold,
            shape (n_outer_train, n_features).
        y_outer_train (pd.Series): Integer-encoded labels for the outer
            training set.
        param_grid (list): List of parameter dictionaries with keys
            'C', 'l1_ratio', and 'top_k'.
        inner_cv (StratifiedKFold): Inner cross-validation splitter.
        outer_seed (int): Random seed for the current outer repeat,
            used as the ElasticNet random state.

    Returns:
        tuple: A tuple of (best_params, inner_results) where best_params
            is the dict with the best hyperparameters, and inner_results
            is a list of dicts recording all inner CV scores.
    """
    best_inner_score = -np.inf
    best_inner_std = None
    best_params = None
    inner_results = []

    for params in param_grid:
        inner_fold_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(
            x_outer_train, y_outer_train
        ):
            x_inner_train = x_outer_train.iloc[inner_train_idx].copy()
            x_inner_val = x_outer_train.iloc[inner_val_idx].copy()
            y_inner_train = y_outer_train.iloc[inner_train_idx].copy()
            y_inner_val = y_outer_train.iloc[inner_val_idx].copy()

            '''Step 1: Fit Elastic Net logistic regression on inner train.'''

            scaler = StandardScaler()
            x_inner_train_scaled = scaler.fit_transform(x_inner_train)
            x_inner_val_scaled = scaler.transform(x_inner_val)

            elastic_net_model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=10000,
                C=params["C"],
                l1_ratio=params["l1_ratio"],
                random_state=outer_seed,
                multi_class="multinomial",
            )

            # Catch convergence warnings and log them instead of printing.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                elastic_net_model.fit(x_inner_train_scaled, y_inner_train)
                for w in caught:
                    if issubclass(w.category, ConvergenceWarning):
                        log.warning(
                            "SAGA did not converge: C=%s, l1_ratio=%s",
                            params["C"],
                            params["l1_ratio"],
                        )

            '''Step 2: Rank features by absolute coefficient magnitude.'''

            cfs = elastic_net_model.coef_
            importance = np.abs(cfs).sum(axis=0)
            importance_df = pd.DataFrame(
                {
                    "feature": x_inner_train.columns,
                    "importance": importance,
                }
            ).sort_values(by="importance", ascending=False)

            selected_features = importance_df.head(params["top_k"])[
                "feature"
            ].tolist()

            '''Step 3: Train NMC on top-k selected features.'''

            x_inner_train_selected = pd.DataFrame(
                x_inner_train_scaled,
                index=x_inner_train.index,
                columns=x_inner_train.columns,
            )[selected_features]

            x_inner_val_selected = pd.DataFrame(
                x_inner_val_scaled,
                index=x_inner_val.index,
                columns=x_inner_val.columns,
            )[selected_features]

            nmc = NearestCentroid()
            nmc.fit(x_inner_train_selected, y_inner_train)

            y_inner_pred = nmc.predict(x_inner_val_selected)
            inner_bal_acc = balanced_accuracy_score(y_inner_val, y_inner_pred)
            inner_fold_scores.append(inner_bal_acc)

        mean_inner_bal_acc = np.mean(inner_fold_scores)
        std_inner_bal_acc = np.std(inner_fold_scores)

        inner_results.append(
            {
                "C": params["C"],
                "l1_ratio": params["l1_ratio"],
                "top_k": params["top_k"],
                "mean_inner_balanced_accuracy": mean_inner_bal_acc,
                "std_inner_balanced_accuracy": std_inner_bal_acc,
            }
        )

        if mean_inner_bal_acc > best_inner_score:
            best_inner_score = mean_inner_bal_acc
            best_inner_std = std_inner_bal_acc
            best_params = params

    return best_params, inner_results


def refit_and_evaluate(
    x_outer_train,
    x_outer_test,
    y_outer_train,
    y_outer_test,
    best_params,
    outer_seed,
    label_encoder,
):
    """Refit the best pipeline on outer train and evaluate on outer test.

    Scales the outer training data, fits an Elastic Net for feature
    selection using the best hyperparameters, selects top-k features,
    trains a NearestCentroid, and evaluates balanced accuracy on the
    held-out outer test fold.

    Args:
        x_outer_train (pd.DataFrame): Outer training features.
        x_outer_test (pd.DataFrame): Outer test features.
        y_outer_train (pd.Series): Integer-encoded outer training labels.
        y_outer_test (pd.Series): Integer-encoded outer test labels.
        best_params (dict): Best hyperparameters from inner CV with keys
            'C', 'l1_ratio', and 'top_k'.
        outer_seed (int): Random seed for reproducibility.
        label_encoder (LabelEncoder): Fitted label encoder for decoding
            integer labels back to subtype strings.

    Returns:
        dict: A dictionary containing evaluation metrics including
            balanced_accuracy, selected features, coefficient DataFrames,
            prediction DataFrame, confusion matrix DataFrame, and
            classification report DataFrame.
    """
    scaler = StandardScaler()
    x_outer_train_scaled = scaler.fit_transform(x_outer_train)
    x_outer_test_scaled = scaler.transform(x_outer_test)

    # Refit Elastic Net with best parameters on full outer training set.
    best_elastic_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=10000,
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        random_state=outer_seed,
        multi_class="multinomial",
    )

    # Catch convergence warnings during refit.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        best_elastic_model.fit(x_outer_train_scaled, y_outer_train)
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                log.warning(
                    "SAGA did not converge during refit: C=%s, l1_ratio=%s",
                    best_params["C"],
                    best_params["l1_ratio"],
                )

    # Rank and select features by absolute coefficient magnitude.
    best_cfs = best_elastic_model.coef_
    best_importance = np.abs(best_cfs).sum(axis=0)

    best_importance_df = pd.DataFrame(
        {
            "feature": x_outer_train.columns,
            "importance": best_importance,
        }
    ).sort_values(by="importance", ascending=False)

    best_selected_features = best_importance_df.head(best_params["top_k"])[
        "feature"
    ].tolist()

    x_outer_train_selected = pd.DataFrame(
        x_outer_train_scaled,
        index=x_outer_train.index,
        columns=x_outer_train.columns,
    )[best_selected_features]

    x_outer_test_selected = pd.DataFrame(
        x_outer_test_scaled,
        index=x_outer_test.index,
        columns=x_outer_test.columns,
    )[best_selected_features]

    """Train NMC on selected features and evaluate on outer test."""

    final_nmc = NearestCentroid()
    final_nmc.fit(x_outer_train_selected, y_outer_train)

    y_outer_pred = final_nmc.predict(x_outer_test_selected)
    outer_bal_acc = balanced_accuracy_score(y_outer_test, y_outer_pred)

    log.info("Outer balanced accuracy: %.4f", outer_bal_acc)

    # Decode labels for interpretable reports.
    y_outer_test_labels = label_encoder.inverse_transform(y_outer_test)
    y_outer_pred_labels = label_encoder.inverse_transform(y_outer_pred)

    cm = confusion_matrix(
        y_outer_test_labels,
        y_outer_pred_labels,
        labels=label_encoder.classes_,
    )
    cm_df = pd.DataFrame(
        cm, index=label_encoder.classes_, columns=label_encoder.classes_
    )

    report_dict = classification_report(
        y_outer_test_labels,
        y_outer_pred_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    coef_df = pd.DataFrame(
        best_cfs.T,
        index=x_outer_train.columns,
        columns=label_encoder.classes_,
    )
    coef_df["importance"] = np.abs(coef_df[label_encoder.classes_]).sum(axis=1)
    coef_selected_df = coef_df.loc[best_selected_features].sort_values(
        by="importance", ascending=False
    )

    pred_df = pd.DataFrame(
        {
            "sample": x_outer_test.index,
            "true_label": y_outer_test_labels,
            "predicted_label": y_outer_pred_labels,
        }
    )

    return {
        "balanced_accuracy": outer_bal_acc,
        "best_selected_features": best_selected_features,
        "n_features_selected": len(best_selected_features),
        "pred_df": pred_df,
        "best_importance_df": best_importance_df,
        "coef_df": coef_df,
        "coef_selected_df": coef_selected_df,
        "cm_df": cm_df,
        "report_df": report_df,
    }


def main():
    """Run the full EN + NMC nested stratified repeated k-fold CV pipeline.

    Parses command-line arguments, loads data, runs the nested CV with
    Elastic Net feature selection and NearestCentroid classification,
    and saves all results to CSV files.
    """

    """Argument Parsing"""

    parser = argparse.ArgumentParser(
        description="Elastic Net + NMC nested cross-validation pipeline."
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of CV repeats (default: 3).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "results" / "data" / "preprocessing_phase" / "train_merged.tsv",
        help="Path to the merged copy-number TSV file.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=PROJECT_DIR / "data" / "Train_clinical.tsv",
        help="Path to the clinical labels TSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "data" / "EN_NMC_nested_cv",
        help="Directory for output CSV files.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    """Data Loading"""

    X, y_encoded, lencoder = load_data(args.input, args.clinical)

    """Hyperparameter Grid"""

    C_grid = [0.01, 0.1, 1, 10]
    l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    top_k_grid = [5, 10, 20, 50]

    param_grid = list(
        ParameterGrid(
            {
                "C": C_grid,
                "l1_ratio": l1_ratio_grid,
                "top_k": top_k_grid,
            }
        )
    )

    """Repeated Nested CV Settings"""

    # Generate seeds for each repeat starting from 1.
    outer_seeds = list(range(1, args.n_repeats + 1))

    all_runs_results = []
    all_outer_results = []
    all_inner_results = []

    # Unified results for the structured output CSV.
    unified_rows = []

    """Repeated Nested CV"""

    for run_id, outer_seed in track(
        list(enumerate(outer_seeds, start=1)),
        description="Repeats",
        console=console,
    ):
        log.info("Run %d (seed=%d)", run_id, outer_seed)

        outer_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=outer_seed
        )

        outer_results = []
        inner_results = []
        selected_features_by_fold = {}

        fold_iter = list(
            enumerate(outer_cv.split(X, y_encoded), start=1)
        )

        for outer_fold, (outer_train_idx, outer_test_idx) in track(
            fold_iter,
            description=f"  Run {run_id} outer folds",
            console=console,
        ):
            log.info("  Outer fold %d", outer_fold)

            x_outer_train = X.iloc[outer_train_idx].copy()
            x_outer_test = X.iloc[outer_test_idx].copy()
            y_outer_train = y_encoded.iloc[outer_train_idx].copy()
            y_outer_test = y_encoded.iloc[outer_test_idx].copy()

            '''Inner CV loop for hyperparameter selection.'''

            # Use offset seed to decorrelate inner and outer splits.
            inner_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=100 + outer_seed
            )

            best_params, fold_inner_results = run_inner_cv(
                x_outer_train,
                y_outer_train,
                param_grid,
                inner_cv,
                outer_seed,
            )

            # Annotate inner results with run and fold identifiers.
            for row in fold_inner_results:
                row["run_id"] = run_id
                row["outer_seed"] = outer_seed
                row["outer_fold"] = outer_fold
            inner_results.extend(fold_inner_results)

            log.info("  Best inner params: %s", best_params)

            '''Refit best pipeline on full outer train and evaluate.'''

            eval_metrics = refit_and_evaluate(
                x_outer_train,
                x_outer_test,
                y_outer_train,
                y_outer_test,
                best_params,
                outer_seed,
                lencoder,
            )

            selected_features_by_fold[
                f"run_{run_id}_outer_fold_{outer_fold}"
            ] = eval_metrics["best_selected_features"]

            '''Save per-fold CSV files.'''

            prefix = f"run{run_id}_outer_fold_{outer_fold}"
            eval_metrics["pred_df"].to_csv(
                output_dir / f"{prefix}_predictions_ENLR_NMC.csv",
                index=False,
            )
            eval_metrics["best_importance_df"].to_csv(
                output_dir / f"{prefix}_feature_ranking_ENLR_NMC.csv",
                index=False,
            )
            eval_metrics["coef_df"].to_csv(
                output_dir / f"{prefix}_all_coefficients_ENLR_NMC.csv"
            )
            eval_metrics["coef_selected_df"].to_csv(
                output_dir / f"{prefix}_selected_coefficients_ENLR_NMC.csv"
            )
            eval_metrics["cm_df"].to_csv(
                output_dir / f"{prefix}_confusion_matrix_ENLR_NMC.csv"
            )
            eval_metrics["report_df"].to_csv(
                output_dir / f"{prefix}_classification_report_ENLR_NMC.csv"
            )

            outer_results.append(
                {
                    "run_id": run_id,
                    "outer_seed": outer_seed,
                    "outer_fold": outer_fold,
                    "best_C": best_params["C"],
                    "best_l1_ratio": best_params["l1_ratio"],
                    "best_top_k": best_params["top_k"],
                    "outer_test_balanced_accuracy": eval_metrics[
                        "balanced_accuracy"
                    ],
                    "n_selected_features": eval_metrics["n_features_selected"],
                }
            )

            # Append row for unified structured output.
            unified_rows.append(
                {
                    "pipeline": "EN_NMC",
                    "repeat": run_id,
                    "outer_fold": outer_fold,
                    "balanced_accuracy": eval_metrics["balanced_accuracy"],
                    "best_C": best_params["C"],
                    "best_l1_ratio": best_params["l1_ratio"],
                    "best_top_k": best_params["top_k"],
                    "n_features_selected": eval_metrics["n_features_selected"],
                }
            )

        '''Summary per repeat iteration.'''

        outer_results_df = pd.DataFrame(outer_results)
        inner_results_df = pd.DataFrame(inner_results)

        iter_mean_outer_bal_acc = outer_results_df[
            "outer_test_balanced_accuracy"
        ].mean()
        iter_std_outer_bal_acc = outer_results_df[
            "outer_test_balanced_accuracy"
        ].std()

        log.info(
            "Run %d mean balanced accuracy: %.4f (+/- %.4f)",
            run_id,
            iter_mean_outer_bal_acc,
            iter_std_outer_bal_acc,
        )

        outer_results_df.to_csv(
            output_dir / f"run{run_id}_nested_cv_outer_results_ENLR_NMC.csv",
            index=False,
        )
        inner_results_df.to_csv(
            output_dir / f"run{run_id}_nested_cv_inner_results_ENLR_NMC.csv",
            index=False,
        )

        all_runs_results.append(
            {
                "run_id": run_id,
                "outer_seed": outer_seed,
                "mean_outer_balanced_acc": iter_mean_outer_bal_acc,
                "std_outer_balanced_acc": iter_std_outer_bal_acc,
            }
        )

        all_outer_results.append(outer_results_df)
        all_inner_results.append(inner_results_df)

    """Final Summary Across All Iterations"""

    all_runs_df = pd.DataFrame(all_runs_results)
    all_outer_results_df = pd.concat(all_outer_results, ignore_index=True)
    all_inner_results_df = pd.concat(all_inner_results, ignore_index=True)

    all_runs_df.to_csv(
        output_dir / "all_runs_summary_ENLR_NMC.csv", index=False
    )
    all_outer_results_df.to_csv(
        output_dir / "all_outer_results_ENLR_NMC.csv", index=False
    )
    all_inner_results_df.to_csv(
        output_dir / "all_inner_results_ENLR_NMC.csv", index=False
    )

    # Save the unified structured output CSV.
    unified_df = pd.DataFrame(unified_rows)
    unified_df.to_csv(
        output_dir / "nested_cv_results.csv", index=False
    )

    log.info("All results saved to %s", output_dir)

    # Log overall summary.
    overall_mean = all_outer_results_df[
        "outer_test_balanced_accuracy"
    ].mean()
    overall_std = all_outer_results_df[
        "outer_test_balanced_accuracy"
    ].std()
    log.info(
        "Overall balanced accuracy across all folds: %.4f (+/- %.4f)",
        overall_mean,
        overall_std,
    )


if __name__ == "__main__":
    main()
