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
    Elastic Net feature selection with Random Forest classifier.
    Nested stratified repeated k-fold cross-validation.

Usage:
    python3 EN_RF_final/script.py

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, rich.
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
import rich.traceback

"""Path Configuration"""

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
        logging.FileHandler(LOG_DIR / "EN_RF_nested_cv.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_data(input_path, clinical_path):
    """Load training data and clinical labels, returning features, encoded labels, and encoder.

    Reads the copy-number call matrix and clinical labels, aligns samples,
    and constructs feature names using the underscore convention
    (chr<Chromosome>_<Start>_<End>).

    Args:
        input_path (Path): Path to the merged training TSV file.
        clinical_path (Path): Path to the clinical labels TSV file.

    Returns:
        tuple: A tuple of (X, y_encoded, label_encoder) where X is a
            DataFrame of shape (n_samples, n_features), y_encoded is a
            Series of integer-encoded labels, and label_encoder is the
            fitted LabelEncoder.
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

    # Align samples present in both feature matrix and labels.
    same_samples = x_train_final.index.intersection(y_train.index)
    x_train_final = x_train_final.loc[same_samples]
    y_train = y_train.loc[same_samples]

    X = x_train_final.copy()
    y = y_train.copy()

    # Encode string labels to integers.
    lencoder = LabelEncoder()
    y_encoded = pd.Series(
        lencoder.fit_transform(y), index=y.index, name="target"
    )

    log.info("Classes: %s", lencoder.classes_)
    log.info("Loaded %d samples with %d features.", X.shape[0], X.shape[1])

    return X, y_encoded, lencoder


def run_inner_cv(x_outer_train, y_outer_train, param_grid, inner_cv, outer_seed):
    """Run inner cross-validation to select the best hyperparameters.

    For each parameter combination in the grid, performs inner k-fold
    cross-validation using Elastic Net feature selection followed by
    Random Forest classification. Returns the best parameters and a list
    of all inner CV results.

    Args:
        x_outer_train (pd.DataFrame): Outer training fold features.
        y_outer_train (pd.Series): Outer training fold encoded labels.
        param_grid (list): List of parameter dictionaries from ParameterGrid.
        inner_cv (StratifiedKFold): Inner CV splitter object.
        outer_seed (int): Random seed for the current outer repeat.

    Returns:
        tuple: A tuple of (best_params, inner_results) where best_params
            is the dict with highest mean balanced accuracy, and
            inner_results is a list of dicts recording all param evaluations.
    """
    best_inner_score = -np.inf
    best_inner_std = None
    best_params = None
    inner_results = []

    for params in track(
        param_grid,
        description="  Param grid...",
        total=len(param_grid),
    ):
        inner_fold_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(
            x_outer_train, y_outer_train
        ):
            x_inner_train = x_outer_train.iloc[inner_train_idx].copy()
            x_inner_val = x_outer_train.iloc[inner_val_idx].copy()
            y_inner_train = y_outer_train.iloc[inner_train_idx].copy()
            y_inner_val = y_outer_train.iloc[inner_val_idx].copy()

            '''Step 1: Fit Elastic Net logistic regression on inner fold.'''

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

            # Catch convergence warnings and log them instead of suppressing.
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

            '''Step 2: Rank features by absolute coefficient sums across classes.'''

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

            '''Step 3: Train Random Forest on the top-k selected features.'''

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

            rf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=outer_seed,
                n_jobs=1,
            )

            rf.fit(x_inner_train_selected, y_inner_train)

            y_inner_pred = rf.predict(x_inner_val_selected)
            inner_bal_acc = balanced_accuracy_score(y_inner_val, y_inner_pred)
            inner_fold_scores.append(inner_bal_acc)

        mean_inner_bal_acc = np.mean(inner_fold_scores)
        std_inner_bal_acc = np.std(inner_fold_scores)

        inner_results.append(
            {
                "C": params["C"],
                "l1_ratio": params["l1_ratio"],
                "top_k": params["top_k"],
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "mean_inner_balanced_accuracy": mean_inner_bal_acc,
                "std_inner_balanced_accuracy": std_inner_bal_acc,
            }
        )

        if mean_inner_bal_acc > best_inner_score:
            best_inner_score = mean_inner_bal_acc
            best_inner_std = std_inner_bal_acc
            best_params = params

    return best_params, best_inner_score, best_inner_std, inner_results


def refit_and_evaluate(
    x_outer_train,
    x_outer_test,
    y_outer_train,
    y_outer_test,
    best_params,
    outer_seed,
    label_encoder,
    output_dir,
    run_id,
    outer_fold,
):
    """Refit the best pipeline on the full outer train set and evaluate on outer test.

    Fits Elastic Net for feature selection, selects top-k features, trains
    Random Forest, and evaluates balanced accuracy on the held-out outer
    test fold. Saves per-fold CSVs for predictions, feature rankings,
    coefficients, confusion matrix, and classification report.

    Args:
        x_outer_train (pd.DataFrame): Outer training fold features.
        x_outer_test (pd.DataFrame): Outer test fold features.
        y_outer_train (pd.Series): Outer training fold encoded labels.
        y_outer_test (pd.Series): Outer test fold encoded labels.
        best_params (dict): Best hyperparameters from inner CV.
        outer_seed (int): Random seed for the current outer repeat.
        label_encoder (LabelEncoder): Fitted label encoder for inverse transforms.
        output_dir (Path): Directory to save per-fold output CSVs.
        run_id (int): Current repeat number (1-indexed).
        outer_fold (int): Current outer fold number (1-indexed).

    Returns:
        dict: Metrics dictionary with balanced accuracy, best params, and
            number of selected features.
    """
    scaler = StandardScaler()
    x_outer_train_scaled = scaler.fit_transform(x_outer_train)
    x_outer_test_scaled = scaler.transform(x_outer_test)

    best_elastic_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=10000,
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        random_state=outer_seed,
        multi_class="multinomial",
    )

    # Catch convergence warnings during outer refit as well.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        best_elastic_model.fit(x_outer_train_scaled, y_outer_train)
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                log.warning(
                    "SAGA did not converge on outer refit: C=%s, l1_ratio=%s",
                    best_params["C"],
                    best_params["l1_ratio"],
                )

    # Rank features by summed absolute coefficients across all classes.
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

    # Train Random Forest on the selected features from the full outer train set.
    final_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=outer_seed,
        n_jobs=1,
    )

    final_rf.fit(x_outer_train_selected, y_outer_train)

    y_outer_pred = final_rf.predict(x_outer_test_selected)
    outer_bal_acc = balanced_accuracy_score(y_outer_test, y_outer_pred)

    log.info("  Outer balanced accuracy: %.4f", outer_bal_acc)

    # Decode labels back to original string names for reporting.
    y_outer_test_labels = label_encoder.inverse_transform(y_outer_test)
    y_outer_pred_labels = label_encoder.inverse_transform(y_outer_pred)

    cm = confusion_matrix(
        y_outer_test_labels, y_outer_pred_labels, labels=label_encoder.classes_
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

    # Save per-fold detailed CSVs into the output directory.
    prefix = f"run{run_id}_outer_fold_{outer_fold}"
    pred_df.to_csv(
        output_dir / f"{prefix}_predictions_ENLR_RF.csv", index=False
    )
    best_importance_df.to_csv(
        output_dir / f"{prefix}_feature_ranking_ENLR_RF.csv", index=False
    )
    coef_df.to_csv(output_dir / f"{prefix}_all_coefficients_ENLR_RF.csv")
    coef_selected_df.to_csv(
        output_dir / f"{prefix}_selected_coefficients_ENLR_RF.csv"
    )
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix_ENLR_RF.csv")
    report_df.to_csv(
        output_dir / f"{prefix}_classification_report_ENLR_RF.csv"
    )

    return {
        "pipeline": "EN_RF",
        "repeat": run_id,
        "outer_fold": outer_fold,
        "balanced_accuracy": outer_bal_acc,
        "best_C": best_params["C"],
        "best_l1_ratio": best_params["l1_ratio"],
        "best_top_k": best_params["top_k"],
        "best_n_estimators": best_params["n_estimators"],
        "best_max_depth": best_params["max_depth"],
        "best_min_samples_split": best_params["min_samples_split"],
        "n_features_selected": len(best_selected_features),
    }


def main():
    """Run the full Elastic Net + Random Forest nested CV pipeline.

    Parses command-line arguments, loads data, performs repeated nested
    stratified k-fold cross-validation with Elastic Net feature selection
    and Random Forest classification, and saves structured results.
    """
    """Argument Parsing"""

    parser = argparse.ArgumentParser(
        description="Elastic Net + Random Forest nested CV pipeline.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of outer CV repeats (default: 3).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "results" / "data" / "preprocessing_phase" / "train_merged.tsv",
        help="Path to the merged training TSV file.",
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
        default=PROJECT_DIR / "results" / "data" / "EN_RF_nested_cv",
        help="Directory to save output files.",
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

    rf_n_estimators_grid = [100, 200]
    rf_max_depth_grid = [None, 5, 10]
    rf_min_samples_split_grid = [2, 5]

    param_grid = list(
        ParameterGrid(
            {
                "C": C_grid,
                "l1_ratio": l1_ratio_grid,
                "top_k": top_k_grid,
                "n_estimators": rf_n_estimators_grid,
                "max_depth": rf_max_depth_grid,
                "min_samples_split": rf_min_samples_split_grid,
            }
        )
    )

    log.info("Total parameter combinations: %d", len(param_grid))

    """Repeated Nested Cross-Validation"""

    # Use seeds 1..n_repeats for the outer CV repeats.
    outer_seeds = list(range(1, args.n_repeats + 1))

    all_structured_results = []
    all_outer_results = []
    all_inner_results = []
    all_runs_results = []

    for run_id, outer_seed in track(
        enumerate(outer_seeds, start=1),
        total=len(outer_seeds),
        description="Repeats...",
    ):
        log.info("Run %d / %d (seed=%d)", run_id, len(outer_seeds), outer_seed)

        outer_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=outer_seed
        )

        outer_fold_results = []

        for outer_fold, (outer_train_idx, outer_test_idx) in track(
            enumerate(outer_cv.split(X, y_encoded), start=1),
            total=5,
            description=" Outer folds...",
        ):
            log.info("  Outer fold: %d", outer_fold)

            x_outer_train = X.iloc[outer_train_idx].copy()
            x_outer_test = X.iloc[outer_test_idx].copy()
            y_outer_train = y_encoded.iloc[outer_train_idx].copy()
            y_outer_test = y_encoded.iloc[outer_test_idx].copy()

            # Use offset seed for inner CV to decorrelate from outer splits.
            inner_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=100 + outer_seed
            )

            best_params, best_inner_score, best_inner_std, inner_results = (
                run_inner_cv(
                    x_outer_train,
                    y_outer_train,
                    param_grid,
                    inner_cv,
                    outer_seed,
                )
            )

            log.info("  Best inner params: %s", best_params)
            log.info("  Best inner balanced accuracy: %.4f", best_inner_score)

            # Record inner results with run and fold identifiers.
            for rec in inner_results:
                rec["run_id"] = run_id
                rec["outer_seed"] = outer_seed
                rec["outer_fold"] = outer_fold
            all_inner_results.extend(inner_results)

            '''Refit best pipeline on outer train and evaluate on outer test.'''

            metrics = refit_and_evaluate(
                x_outer_train,
                x_outer_test,
                y_outer_train,
                y_outer_test,
                best_params,
                outer_seed,
                lencoder,
                output_dir,
                run_id,
                outer_fold,
            )

            all_structured_results.append(metrics)

            outer_fold_results.append(
                {
                    "run_id": run_id,
                    "outer_seed": outer_seed,
                    "outer_fold": outer_fold,
                    "best_C": best_params["C"],
                    "best_l1_ratio": best_params["l1_ratio"],
                    "best_top_k": best_params["top_k"],
                    "best_n_estimators": best_params["n_estimators"],
                    "best_max_depth": best_params["max_depth"],
                    "best_min_samples_split": best_params["min_samples_split"],
                    "best_inner_balanced_accuracy": best_inner_score,
                    "best_inner_std": best_inner_std,
                    "outer_test_balanced_accuracy": metrics["balanced_accuracy"],
                    "n_selected_features": metrics["n_features_selected"],
                }
            )

        """Summary Per Repeat"""

        outer_results_df = pd.DataFrame(outer_fold_results)
        iter_mean_outer_bal_acc = outer_results_df[
            "outer_test_balanced_accuracy"
        ].mean()
        iter_std_outer_bal_acc = outer_results_df[
            "outer_test_balanced_accuracy"
        ].std()

        # Save per-repeat outer and inner results.
        outer_results_df.to_csv(
            output_dir / f"run{run_id}_nested_cv_outer_results_ENLR_RF.csv",
            index=False,
        )
        inner_run_df = pd.DataFrame(
            [r for r in all_inner_results if r["run_id"] == run_id]
        )
        inner_run_df.to_csv(
            output_dir / f"run{run_id}_nested_cv_inner_results_ENLR_RF.csv",
            index=False,
        )

        all_outer_results.append(outer_results_df)

        all_runs_results.append(
            {
                "run_id": run_id,
                "outer_seed": outer_seed,
                "mean_outer_balanced_accuracy": iter_mean_outer_bal_acc,
                "std_outer_balanced_accuracy": iter_std_outer_bal_acc,
            }
        )

        log.info(
            "  Run %d mean outer balanced accuracy: %.4f (+/- %.4f)",
            run_id,
            iter_mean_outer_bal_acc,
            iter_std_outer_bal_acc,
        )

    """Final Summary Across All Repeats"""

    all_runs_df = pd.DataFrame(all_runs_results)
    all_outer_results_df = pd.concat(all_outer_results, ignore_index=True)
    all_inner_results_df = pd.DataFrame(all_inner_results)

    all_runs_df.to_csv(
        output_dir / "all_runs_summary_ENLR_RF.csv", index=False
    )
    all_outer_results_df.to_csv(
        output_dir / "all_outer_results_ENLR_RF.csv", index=False
    )
    all_inner_results_df.to_csv(
        output_dir / "all_inner_results_ENLR_RF.csv", index=False
    )

    # Save the structured unified results CSV.
    structured_df = pd.DataFrame(all_structured_results)
    structured_df.to_csv(
        output_dir / "nested_cv_results.csv", index=False
    )

    log.info("Structured results saved to: %s", output_dir / "nested_cv_results.csv")

    # Report overall summary.
    overall_mean = structured_df["balanced_accuracy"].mean()
    overall_std = structured_df["balanced_accuracy"].std()
    log.info(
        "Overall balanced accuracy: %.4f (+/- %.4f)",
        overall_mean,
        overall_std,
    )


if __name__ == "__main__":
    main()
