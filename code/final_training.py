"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: final_training.py.
Description:
    Train the final hierarchical classifier on all 100 labelled samples
    and generate predictions for the 57 validation samples.

    The hierarchical architecture uses two stages:
        Stage 1 (fixed): KW feature selection (k=5) + Random Forest.
            Classifies HER2+ vs rest.
        Stage 2 (plateau ensemble): 15 ElasticNet + NMC models with
            varied hyperparameters, averaged for robustness.
            Classifies HR+ vs Triple Neg on samples not routed to HER2+.

    Outputs:
        - results/prediction.txt  (57 validation predictions).
        - results/estimate.txt    (expected correct count out of 57).
        - model/model.pkl         (serialised model dictionary).

Usage:
    python3 code/final_training.py --name final_hierarchical

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, joblib, rich.
"""

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from utils import get_run_dirs, save_config, setup_logging
from utils.cv_config import build_pipeline, build_stage2_pipeline
from utils.cv_io import load_cv_data
from utils.paths import DATA_DIR, PROJECT_DIR, RESULTS_DIR

MODEL_DIR = PROJECT_DIR / "model"

import rich.traceback
rich.traceback.install()


"""Constants"""

# Plateau ensemble hyperparameter configurations for Stage 2.
# Extracted from pooled inner CV results across 200 repeats x 5 folds
# of the en_nmc base pipeline. Threshold: best_mean - best_std, capped
# at 15. All configurations use selector__top_k=50.
PLATEAU_CONFIGS = [
    {"C": 0.0464159, "l1_ratio": 0.1, "shrink_threshold": None},
    {"C": 0.0464159, "l1_ratio": 0.1, "shrink_threshold": 0.2},
    {"C": 0.0464159, "l1_ratio": 0.1, "shrink_threshold": 0.1},
    {"C": 0.16681, "l1_ratio": 0.5, "shrink_threshold": 0.2},
    {"C": 0.16681, "l1_ratio": 0.5, "shrink_threshold": 0.1},
    {"C": 0.16681, "l1_ratio": 0.5, "shrink_threshold": 0.5},
    {"C": 0.16681, "l1_ratio": 0.5, "shrink_threshold": None},
    {"C": 0.0464159, "l1_ratio": 0.1, "shrink_threshold": 0.5},
    {"C": 0.599484, "l1_ratio": 0.7, "shrink_threshold": 0.2},
    {"C": 0.16681, "l1_ratio": 0.3, "shrink_threshold": 0.1},
    {"C": 0.16681, "l1_ratio": 0.3, "shrink_threshold": 0.2},
    {"C": 0.16681, "l1_ratio": 0.1, "shrink_threshold": None},
    {"C": 0.16681, "l1_ratio": 0.1, "shrink_threshold": 0.2},
    {"C": 0.16681, "l1_ratio": 0.1, "shrink_threshold": 0.1},
    {"C": 0.599484, "l1_ratio": 0.7, "shrink_threshold": 0.1},
]

# NMC class weighting must match the CV configuration.
STAGE2_CONFIG = {
    "pipelines": {
        "nmc_class_weight": "balanced",
    },
}

# Mean combined balanced accuracy from 200-repeat nested CV.
MEAN_COMBINED_BA = 0.8395


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with name attribute.
    """
    parser = argparse.ArgumentParser(
        description="Train the final hierarchical classifier and predict.",
    )
    parser.add_argument(
        "--name", default="final_hierarchical",
        help="Run name for output directories (default: final_hierarchical).",
    )
    return parser.parse_args()


def load_training_data(run_dir, log):
    """Load merged training data and clinical labels.

    Resolves the preprocessed merged TSV from the run directory and loads
    clinical labels from the raw data directory.

    Args:
        run_dir (Path): Run root directory containing preprocessing output.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (X_train, y, le, feature_names) where X_train is a DataFrame
            with samples as rows and features as columns, y is the integer-
            encoded label array, le is the fitted LabelEncoder, and
            feature_names is the list of region name strings.
    """
    train_merged = run_dir / "preprocessing" / "data" / "train_merged.tsv"
    train_clinical = DATA_DIR / "Train_clinical.tsv"

    X, y, le, feature_names, _ = load_cv_data(train_merged, train_clinical)
    X_train = pd.DataFrame(X, columns=feature_names)
    log.info("Loaded training data: %d samples x %d features.", *X_train.shape)
    log.info("Class distribution: %s", dict(zip(le.classes_, np.bincount(y))))
    return X_train, y, le, feature_names


def load_validation_data(run_dir, feature_names, log):
    """Load and transpose merged validation data.

    Args:
        run_dir (Path): Run root directory containing preprocessing output.
        feature_names (list[str]): Feature names matching training columns.
        log (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Validation features with samples as rows.
    """
    val_path = run_dir / "preprocessing" / "data" / "validation_merged.tsv"
    val_df = pd.read_csv(val_path, sep="\t")
    sample_cols = [
        c for c in val_df.columns
        if c not in ("Chromosome", "Start", "End", "Nclone")
    ]
    X_val = val_df[sample_cols].T
    X_val.columns = feature_names
    log.info("Loaded validation data: %d samples x %d features.", *X_val.shape)
    return X_val


def train_stage1(X_train, y, log):
    """Train the Stage 1 classifier (HER2+ vs rest).

    Stage 1 is fixed: KW feature selection (k=5) + Random Forest.
    Binary labels: 1 = HER2+, 0 = rest.

    Args:
        X_train (pd.DataFrame): Training features.
        y (np.ndarray): Integer-encoded 3-class labels (0=HER2+).
        log (logging.Logger): Logger instance.

    Returns:
        Pipeline: The fitted Stage 1 pipeline.
    """
    her2_idx = 0
    y_s1 = (y == her2_idx).astype(int)
    log.info(
        "Stage 1 training: %d HER2+ vs %d rest.",
        y_s1.sum(), len(y_s1) - y_s1.sum(),
    )

    stage1_pipe = build_pipeline("kw_rf", random_state=42)
    stage1_pipe.set_params(selector__k=5)
    stage1_pipe.fit(X_train, y_s1)

    log.info("Stage 1 training complete.")
    return stage1_pipe


def train_stage2_ensemble(X_train, y, log):
    """Train the Stage 2 plateau ensemble (HR+ vs Triple Neg).

    Trains 15 EN+NMC pipelines with different hyperparameter combinations
    from the plateau analysis. Each pipeline is fitted on only the HR+ and
    Triple Neg samples.

    Args:
        X_train (pd.DataFrame): Training features (all samples).
        y (np.ndarray): Integer-encoded 3-class labels.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (stage2_models, le_s2) where stage2_models is a list of
            fitted pipelines, and le_s2 is the Stage 2 LabelEncoder
            mapping subset indices back to original class indices.
    """
    # Exclude HER2+ samples (class index 0) for Stage 2 training.
    her2_idx = 0
    mask_s2 = (y != her2_idx)
    X_s2 = X_train[mask_s2]
    y_s2 = y[mask_s2]

    # Encode the HR+/TN subset to binary labels for the Stage 2 models.
    le_s2 = LabelEncoder()
    y_s2_enc = le_s2.fit_transform(y_s2)
    log.info(
        "Stage 2 training: %d samples, classes %s -> encoded [0, 1].",
        len(y_s2), le_s2.classes_,
    )

    stage2_models = []
    for i, cfg in enumerate(PLATEAU_CONFIGS):
        pipe = build_stage2_pipeline(
            "en_nmc", random_state=42, config=STAGE2_CONFIG,
        )
        pipe.set_params(
            selector__top_k=50,
            selector__C=cfg["C"],
            selector__l1_ratio=cfg["l1_ratio"],
            clf__shrink_threshold=cfg["shrink_threshold"],
        )
        pipe.fit(X_s2, y_s2_enc)
        stage2_models.append(pipe)
        log.info(
            "  Plateau model %2d/%d trained (C=%.4g, l1=%.1f, shrink=%s).",
            i + 1, len(PLATEAU_CONFIGS),
            cfg["C"], cfg["l1_ratio"], cfg["shrink_threshold"],
        )

    log.info("Stage 2 training complete: %d models.", len(stage2_models))
    return stage2_models, le_s2


def predict_hierarchical(stage1_pipe, stage2_models, le_s2, X_val, log):
    """Generate hierarchical predictions for validation samples.

    Stage 1 identifies HER2+ samples (predict_proba > 0.5).
    Stage 2 classifies remaining samples as HR+ or Triple Neg using
    the averaged probabilities from all plateau ensemble models.

    Args:
        stage1_pipe (Pipeline): Fitted Stage 1 pipeline.
        stage2_models (list[Pipeline]): Fitted Stage 2 pipelines.
        le_s2 (LabelEncoder): Stage 2 subset label encoder.
        X_val (pd.DataFrame): Validation features.
        log (logging.Logger): Logger instance.

    Returns:
        list[str]: Predicted subtype labels for each validation sample.
    """
    # Stage 1: classify HER2+ vs rest.
    proba_s1 = stage1_pipe.predict_proba(X_val)[:, 1]
    pred_s1 = (proba_s1 >= 0.5).astype(int)
    n_her2 = pred_s1.sum()
    log.info("Stage 1 predictions: %d HER2+ out of %d.", n_her2, len(pred_s1))

    # Stage 2: classify non-HER2+ samples.
    mask_s2 = (pred_s1 == 0)
    X_val_s2 = X_val[mask_s2]

    probs = [model.predict_proba(X_val_s2) for model in stage2_models]
    avg_proba = np.mean(probs, axis=0)
    pred_s2 = np.argmax(avg_proba, axis=1)

    # Combine Stage 1 and Stage 2 predictions into final labels.
    final_pred = []
    idx_s2 = 0
    for i in range(len(pred_s1)):
        if pred_s1[i] == 1:
            final_pred.append("HER2+")
        else:
            # Inverse transform maps 0 -> original HR+ index (1),
            # and 1 -> original TN index (2).
            label = le_s2.inverse_transform([pred_s2[idx_s2]])[0]
            if label == 1:
                final_pred.append("HR+")
            else:
                final_pred.append("Triple Neg")
            idx_s2 += 1

    log.info(
        "Final predictions: %d HER2+, %d HR+, %d TN.",
        final_pred.count("HER2+"),
        final_pred.count("HR+"),
        final_pred.count("Triple Neg"),
    )
    return final_pred


def save_predictions(X_val, final_pred, log):
    """Save predictions to results/prediction.txt in the required format.

    Format: tab-separated, two columns with quoted headers, 58 lines total.

    Args:
        X_val (pd.DataFrame): Validation data (index = sample names).
        final_pred (list[str]): Predicted subtype labels.
        log (logging.Logger): Logger instance.
    """
    pred_path = RESULTS_DIR / "prediction.txt"
    with open(pred_path, "w", newline="\n") as f:
        f.write('"Sample"\t"Subgroup"\n')
        for sample, label in zip(X_val.index, final_pred):
            f.write(f"{sample}\t{label}\n")
    log.info("Predictions saved to %s (%d samples).", pred_path, len(final_pred))


def save_estimate(log):
    """Save the accuracy estimate to results/estimate.txt.

    The estimate is the expected number of correct predictions out of 57,
    derived from the mean combined balanced accuracy across 200 repeats
    of nested cross-validation.

    Args:
        log (logging.Logger): Logger instance.
    """
    estimate = round(MEAN_COMBINED_BA * 57)
    est_path = RESULTS_DIR / "estimate.txt"
    with open(est_path, "w", newline="\n") as f:
        f.write(str(estimate))
    log.info("Estimate saved to %s: %d / 57 (BA=%.4f).", est_path, estimate, MEAN_COMBINED_BA)


def save_model(stage1_pipe, stage2_models, le_s2, feature_names, run_dir, log):
    """Serialise the full model dictionary to model/model.pkl.

    The model dictionary contains everything needed by run_model.py to
    reproduce predictions from raw validation data.

    Args:
        stage1_pipe (Pipeline): Fitted Stage 1 pipeline.
        stage2_models (list[Pipeline]): Fitted Stage 2 pipelines.
        le_s2 (LabelEncoder): Stage 2 label encoder.
        feature_names (list[str]): Feature name strings.
        run_dir (Path): Run directory containing the merge map.
        log (logging.Logger): Logger instance.
    """
    merge_map_path = run_dir / "preprocessing" / "data" / "merge_map.json"
    with open(merge_map_path) as f:
        merge_map = json.load(f)

    model_dict = {
        "stage1_pipeline": stage1_pipe,
        "stage2_pipelines": stage2_models,
        "label_encoder_stage2": le_s2,
        "feature_names": feature_names,
        "merge_map": merge_map,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(model_dict, model_path)
    log.info("Model saved to %s.", model_path)


def main():
    """Train the final hierarchical classifier and save all deliverables."""
    args = parse_args()

    # Resolve the run directory (reuses existing date-prefixed directory).
    from utils.paths import _find_or_create_run_dir
    run_dir = _find_or_create_run_dir(args.name)

    # Set up logging into a final_training phase directory.
    log_dir = run_dir / "final_training" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log, console = setup_logging("final_training", log_dir=log_dir)
    save_config(run_dir, "final_training", plateau_size=len(PLATEAU_CONFIGS))

    log.info("Run directory: %s", run_dir)

    """Load Data"""

    X_train, y, le, feature_names = load_training_data(run_dir, log)
    X_val = load_validation_data(run_dir, feature_names, log)

    """Train Stage 1: HER2+ vs Rest"""

    stage1_pipe = train_stage1(X_train, y, log)

    """Train Stage 2: HR+ vs Triple Neg (Plateau Ensemble)"""

    stage2_models, le_s2 = train_stage2_ensemble(X_train, y, log)

    """Generate Predictions"""

    final_pred = predict_hierarchical(
        stage1_pipe, stage2_models, le_s2, X_val, log,
    )

    """Save Deliverables"""

    save_predictions(X_val, final_pred, log)
    save_estimate(log)
    save_model(stage1_pipe, stage2_models, le_s2, feature_names, run_dir, log)

    log.info("Final training complete.")


if __name__ == "__main__":
    main()
