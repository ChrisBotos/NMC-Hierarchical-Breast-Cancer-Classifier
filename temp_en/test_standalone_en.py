"""Quick experiment: standalone Elastic Net classifier for Stage 2.

Instead of using ElasticNet as a feature *selector* feeding into NMC/RF,
this script uses LogisticRegression(penalty='elasticnet') directly as the
Stage 2 classifier. This is what "just elastic net" means - one model
handles both implicit feature selection (via L1) and classification.

Stage 1 remains fixed KW+RF (k=5) as always.

Usage:
    python3 temp_en/test_standalone_en.py
"""

import sys
from pathlib import Path

# Allow imports from code/.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "code"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.cv_components import KruskalWallisSelector
from utils.cv_config import build_pipeline
from utils.cv_io import load_cv_data

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
N_REPEATS = 3
OUTER_FOLDS = 5
INNER_FOLDS = 5

# Grid for the standalone elastic net classifier.
EN_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# For comparison: also test KW feature selection + EN classifier.
KW_EN_GRID = {
    "selector__k": [5, 10, 20, 50],
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__l1_ratio": [0.1, 0.5, 0.9],
}

# Merged data path.
MERGED_PATH = (
    PROJECT_DIR
    / "results"
    / "2026-04-22_server_run_hierarchical_nested_CV_no_fe"
    / "preprocessing"
    / "data"
    / "train_merged.tsv"
)
CLINICAL_PATH = PROJECT_DIR / "data" / "Train_clinical.tsv"

# ---------------------------------------------------------------------------
# Pipeline builders.
# ---------------------------------------------------------------------------

def build_standalone_en(random_state=42):
    """StandardScaler -> LogisticRegression(elasticnet). No separate selector."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=random_state,
            n_jobs=1,
        )),
    ])


def build_kw_en(random_state=42):
    """StandardScaler -> KW selector -> LogisticRegression(elasticnet)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", KruskalWallisSelector()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=random_state,
            n_jobs=1,
        )),
    ])

# ---------------------------------------------------------------------------
# Main experiment.
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Standalone Elastic Net - Stage 2 experiment")
    print("=" * 60)

    # Load data.
    X, y, le, feature_names, sample_names = load_cv_data(
        MERGED_PATH, CLINICAL_PATH,
    )
    classes = list(le.classes_)
    her2_idx = classes.index("HER2+")
    hr_idx = classes.index("HR+")
    tn_idx = classes.index("Triple Neg")

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Classes: {classes}")
    print(f"Class counts: {np.bincount(y)}")
    print()

    # Config dict for build_pipeline (Stage 1).
    config = {"pipelines": {"rf_n_estimators": 500, "rf_class_weight": "balanced"}}

    # Two variants to test.
    variants = {
        "standalone_en": (build_standalone_en, EN_GRID),
        "kw_en": (build_kw_en, KW_EN_GRID),
    }

    all_results = []

    for variant_name, (builder_fn, param_grid) in variants.items():
        print(f"--- Variant: {variant_name} ---")
        fold_scores = []

        for repeat in range(1, N_REPEATS + 1):
            repeat_seed = repeat

            # Outer CV.
            outer_cv = StratifiedKFold(
                n_splits=OUTER_FOLDS, shuffle=True, random_state=repeat_seed,
            )

            for fold_idx, (train_idx, test_idx) in enumerate(
                outer_cv.split(X, y), start=1,
            ):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # --- Stage 1: Fixed KW+RF (k=5) ---
                stage1_pipe = build_pipeline(
                    "kw_rf", random_state=repeat_seed, config=config,
                )
                stage1_pipe.set_params(selector__k=5)

                y_train_s1 = (y_train == her2_idx).astype(int)
                y_test_s1 = (y_test == her2_idx).astype(int)
                stage1_pipe.fit(X_train, y_train_s1)
                y_pred_s1 = stage1_pipe.predict(X_test)
                s1_ba = balanced_accuracy_score(y_test_s1, y_pred_s1)

                # --- Stage 2: Standalone EN or KW+EN ---
                # Train on HR+/TN only.
                mask_train_s2 = np.isin(y_train, [hr_idx, tn_idx])
                X_train_s2 = X_train[mask_train_s2]
                y_train_s2 = (y_train[mask_train_s2] == tn_idx).astype(int)

                stage2_pipe = builder_fn(random_state=repeat_seed)
                inner_cv = StratifiedKFold(
                    n_splits=INNER_FOLDS,
                    shuffle=True,
                    random_state=200 + repeat_seed,
                )
                gscv = GridSearchCV(
                    estimator=stage2_pipe,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="balanced_accuracy",
                    n_jobs=1,
                    refit=True,
                )
                gscv.fit(X_train_s2, y_train_s2)

                # --- Combine predictions ---
                y_pred_combined = np.full(len(y_test), -1, dtype=int)
                # Samples predicted HER2+ by Stage 1.
                s1_her2_mask = y_pred_s1 == 1
                y_pred_combined[s1_her2_mask] = her2_idx

                # Route rest to Stage 2.
                s2_test_mask = ~s1_her2_mask
                if s2_test_mask.any():
                    X_test_s2 = X_test[s2_test_mask]
                    y_pred_s2_binary = gscv.predict(X_test_s2)
                    # Map back: 0 -> HR+, 1 -> TN.
                    y_pred_s2_orig = np.where(
                        y_pred_s2_binary == 1, tn_idx, hr_idx,
                    )
                    y_pred_combined[s2_test_mask] = y_pred_s2_orig

                # 3-class balanced accuracy.
                combined_ba = balanced_accuracy_score(y_test, y_pred_combined)

                # Stage 2 BA on test samples that are actually HR+/TN.
                mask_test_s2 = np.isin(y_test, [hr_idx, tn_idx]) & s2_test_mask
                if mask_test_s2.any():
                    s2_ba = balanced_accuracy_score(
                        (y_test[mask_test_s2] == tn_idx).astype(int),
                        (y_pred_combined[mask_test_s2] == tn_idx).astype(int),
                    )
                else:
                    s2_ba = np.nan

                fold_scores.append({
                    "variant": variant_name,
                    "repeat": repeat,
                    "fold": fold_idx,
                    "stage1_ba": s1_ba,
                    "stage2_ba": s2_ba,
                    "combined_ba": combined_ba,
                    "best_params": str(gscv.best_params_),
                    "best_inner_score": gscv.best_score_,
                })

            print(f"  Repeat {repeat}: mean combined BA = "
                  f"{np.mean([s['combined_ba'] for s in fold_scores if s['repeat'] == repeat]):.3f}")

        all_results.extend(fold_scores)

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(all_results)
    for variant_name in variants:
        subset = df[df["variant"] == variant_name]
        mean_combined = subset["combined_ba"].mean()
        std_combined = subset["combined_ba"].std()
        mean_s2 = subset["stage2_ba"].mean()
        std_s2 = subset["stage2_ba"].std()
        mean_s1 = subset["stage1_ba"].mean()
        print(f"\n{variant_name}:")
        print(f"  Stage 1 BA:       {mean_s1:.3f} (should be 1.000)")
        print(f"  Stage 2 BA:       {mean_s2:.3f} +/- {std_s2:.3f}")
        print(f"  Combined 3c BA:   {mean_combined:.3f} +/- {std_combined:.3f}")

        # Most common best params.
        param_counts = subset["best_params"].value_counts()
        print(f"  Most common params: {param_counts.index[0]}")

    # Reference comparison.
    print()
    print("-" * 60)
    print("REFERENCE (50-repeat server run, hierarchical):")
    print("  EN+NMC:  combined BA = 0.812 +/- 0.025")
    print("  KW+NMC:  combined BA = 0.811 +/- 0.032")
    print("  KW+RF:   combined BA = 0.800 +/- 0.027")
    print("  EN+RF:   combined BA = 0.791 +/- 0.030")
    print("-" * 60)

    # Save raw results.
    out_path = Path(__file__).resolve().parent / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
