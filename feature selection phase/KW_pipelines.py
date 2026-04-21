import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import kruskal


class KruskalWallisSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        n_features = X.shape[1]
        if self.k > n_features:
            raise ValueError(
                f"k={self.k} is larger than the number of features ({n_features})."
            )

        scores = []

        for i in range(n_features):
            feature = X[:, i]
            groups = [feature[y == cls] for cls in np.unique(y)]
            stat, _ = kruskal(*groups)
            scores.append(stat)

        self.scores_ = np.array(scores)
        self.indices_ = np.argsort(self.scores_)[::-1][:self.k]
        return self

    def transform(self, X):
        if not hasattr(self, "indices_"):
            raise AttributeError("KruskalWallisSelector is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X[:, self.indices_]


def load_data(train_merged_path="../results/data/preprocessing_phase/train_merged.tsv",
              clinical_path="../data/Train_clinical.tsv"):
    train_merged = pd.read_csv(train_merged_path, sep="\t")
    clinical = pd.read_csv(clinical_path, sep="\t")

    meta_cols = ["Chromosome", "Start", "End", "Nclone"]
    sample_cols = [c for c in train_merged.columns if c not in meta_cols]

    feature_names = (
        "chr"
        + train_merged["Chromosome"].astype(str)
        + "_"
        + train_merged["Start"].astype(str)
        + "_"
        + train_merged["End"].astype(str)
    )

    X_df = train_merged[sample_cols].copy()
    X_df.index = feature_names
    X_df = X_df.T  # samples as rows

    clinical = clinical.set_index("Sample")
    y_series = clinical.loc[X_df.index, "Subgroup"]

    le = LabelEncoder()
    y = le.fit_transform(y_series)
    X = X_df.to_numpy(dtype=float)

    return X, y, le, X_df


def make_kw_pipeline_and_grid(model_name):
    if model_name == "nmc":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kw", KruskalWallisSelector()),
            ("clf", NearestCentroid())
        ])

        param_grid = {
            "kw__k": [5, 20] #add more later!
        }

    elif model_name == "rf":
        pipeline = Pipeline([
            ("kw", KruskalWallisSelector()), #rf does not need standard scaler
            ("clf", RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            "kw__k": [5, 20], #add more later!
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5]
        }

    else:
        raise ValueError("model_name must be 'nmc' or 'rf'.")

    return pipeline, param_grid


def run_repeated_nested_cv(X, y, pipeline, param_grid, repeat_seeds):
    repeat_results = []
    all_outer_scores = []

    for s in repeat_seeds:
        print(f"\n==============================")
        print(f"Repeat with seed {s}")
        print(f"==============================")

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + s)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True
        )

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)

        outer_scores = []
        outer_best_params = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_outer_train, X_outer_test = X[train_idx], X[test_idx]
            y_outer_train, y_outer_test = y[train_idx], y[test_idx]

            grid.fit(X_outer_train, y_outer_train)

            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = best_pipeline.predict(X_outer_test)
            acc = accuracy_score(y_outer_test, y_pred)

            outer_scores.append(acc)
            outer_best_params.append(best_params)

            print(f"\nRepeat {s}, outer fold {fold_idx}")
            print("Best params:", best_params)
            print("Inner CV best score:", grid.best_score_)
            print("Outer test accuracy:", acc)

        repeat_mean = np.mean(outer_scores)
        repeat_std = np.std(outer_scores)

        repeat_results.append({
            "seed": s,
            "outer_scores": outer_scores,
            "mean_accuracy": repeat_mean,
            "std_accuracy": repeat_std,
            "best_params_per_fold": outer_best_params
        })

        all_outer_scores.extend(outer_scores)

        print(f"\nRepeat {s} summary")
        print("Outer fold accuracies:", outer_scores)
        print("Repeat mean accuracy:", repeat_mean)
        print("Repeat std accuracy:", repeat_std)

    repeat_means = [r["mean_accuracy"] for r in repeat_results]

    summary = {
        "repeat_results": repeat_results,
        "repeat_mean_accuracies": repeat_means,
        "mean_of_repeat_means": float(np.mean(repeat_means)),
        "std_of_repeat_means": float(np.std(repeat_means)),
        "all_outer_scores": all_outer_scores,
        "mean_all_outer_scores": float(np.mean(all_outer_scores)),
        "std_all_outer_scores": float(np.std(all_outer_scores)),
    }

    return summary


X, y, le, X_df = load_data()

repeat_seeds = [1, 2, 3] #add more later

# KW + NMC
pipeline_nmc, grid_nmc = make_kw_pipeline_and_grid("nmc")
results_nmc = run_repeated_nested_cv(X, y, pipeline_nmc, grid_nmc, repeat_seeds)

print("\nKW + NMC")
print("Mean of repeat means:", results_nmc["mean_of_repeat_means"])
print("Std of repeat means:", results_nmc["std_of_repeat_means"])

# KW + RF
pipeline_rf, grid_rf = make_kw_pipeline_and_grid("rf")
results_rf = run_repeated_nested_cv(X, y, pipeline_rf, grid_rf, repeat_seeds)

print("\nKW + RF")
print("Mean of repeat means:", results_rf["mean_of_repeat_means"])
print("Std of repeat means:", results_rf["std_of_repeat_means"])