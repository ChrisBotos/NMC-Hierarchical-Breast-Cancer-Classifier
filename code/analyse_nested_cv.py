"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: analyse_nested_cv.py.
Description:
    Aggregates per-fold nested CV results from the 2x2 experimental design,
    computes summary statistics, performs statistical comparisons (Friedman +
    Wilcoxon + Nadeau-Bengio corrected t-test), identifies the winning
    pipeline, and produces five publication-quality figures including violin
    plots of balanced accuracy distributions.

Usage:
    python3 code/analyse_nested_cv.py --name default_run --config local
    python3 code/analyse_nested_cv.py --name my_run --config path/to/config.yaml --phase hierarchical_nested_cv_2x2

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""

"""Imports and Configuration"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback
from scipy import stats
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

# Ensure the code/ directory is on sys.path so utils is importable.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.config_loader import load_config
from utils.constants import PIPELINE_COLORS, PIPELINE_LABELS
from utils.cv_config import PIPELINE_NAMES
from utils.logging_setup import setup_logging
from utils.paths import _find_or_create_run_dir, get_run_dirs, save_config
from utils.plotting import annotate_heatmap, apply_plot_style, draw_significance_brackets
from utils.statistics import nadeau_bengio_test, pairwise_wilcoxon

rich.traceback.install()


"""Data Loading and Aggregation"""


def load_fold_results(nested_cv_data_dir, log):
    """Load and concatenate all fold result CSVs from the nested CV phase.

    Globs for fold_results_*.csv files in the given directory and
    concatenates them into a single DataFrame.

    Args:
        nested_cv_data_dir (Path): Path to nested_cv_2x2/data/ directory.
        log (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Concatenated fold results with columns: pipeline,
            repeat, outer_fold, balanced_accuracy, auroc_macro,
            best_inner_score, best_params, n_features_selected,
            selected_features, fold_seconds.

    Raises:
        FileNotFoundError: If no fold result CSVs are found.
    """
    csv_files = sorted(nested_cv_data_dir.glob("fold_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No fold_results_*.csv files found in {nested_cv_data_dir}"
        )

    log.info("Found %d fold result files.", len(csv_files))
    dfs = [pd.read_csv(f) for f in csv_files]
    all_results = pd.concat(dfs, ignore_index=True)

    # Harmonise column names from hierarchical runner to match flat runner.
    if "stage2_pipeline" in all_results.columns and "pipeline" not in all_results.columns:
        rename_map = {
            "stage2_pipeline": "pipeline",
            "combined_bal_acc": "balanced_accuracy",
            "stage2_best_inner": "best_inner_score",
            "stage2_best_params": "best_params",
            "stage2_n_features": "n_features_selected",
            "stage2_features": "selected_features",
        }
        all_results = all_results.rename(
            columns={k: v for k, v in rename_map.items() if k in all_results.columns},
        )
        log.info("Renamed hierarchical columns to standard analysis names.")

    log.info(
        "Aggregated %d fold rows across %d pipelines and %d repeats.",
        len(all_results),
        all_results["pipeline"].nunique(),
        all_results["repeat"].nunique(),
    )
    return all_results


def compute_per_repeat_means(all_results):
    """Average fold scores within each (pipeline, repeat) to get one value per repeat.

    This produces the repeated-measures data needed for statistical testing:
    each repeat gives one balanced accuracy per pipeline.

    Args:
        all_results (pd.DataFrame): Raw fold-level results.

    Returns:
        pd.DataFrame: Columns: pipeline, repeat, mean_balanced_accuracy,
            mean_auroc_macro, mean_n_features.
    """
    grouped = all_results.groupby(["pipeline", "repeat"], as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc_macro=("auroc_macro", "mean"),
        mean_n_features=("n_features_selected", "mean"),
    )
    return grouped


def compute_summary_statistics(per_repeat):
    """Compute overall summary statistics per pipeline.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.

    Returns:
        pd.DataFrame: One row per pipeline with mean, std, median, min, max
            for balanced accuracy and AUROC.
    """
    summary = per_repeat.groupby("pipeline").agg(
        mean_bal_acc=("mean_balanced_accuracy", "mean"),
        std_bal_acc=("mean_balanced_accuracy", "std"),
        median_bal_acc=("mean_balanced_accuracy", "median"),
        min_bal_acc=("mean_balanced_accuracy", "min"),
        max_bal_acc=("mean_balanced_accuracy", "max"),
        mean_auroc=("mean_auroc_macro", "mean"),
        std_auroc=("mean_auroc_macro", "std"),
        mean_n_features=("mean_n_features", "mean"),
        n_repeats=("mean_balanced_accuracy", "count"),
    )
    # Sort by mean balanced accuracy descending.
    summary = summary.sort_values("mean_bal_acc", ascending=False)
    return summary


"""Statistical Testing"""


def run_statistical_tests(per_repeat, log):
    """Run Friedman test and pairwise Wilcoxon signed-rank tests.

    Thin wrapper around utils.statistics.pairwise_wilcoxon that pivots
    per-repeat data into the required wide format.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, mean_balanced_accuracy.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (friedman_stat, friedman_p, pairwise_df) where pairwise_df
            is a DataFrame with columns: pipeline_a, pipeline_b, statistic,
            p_value, p_corrected, significant. Returns None for pairwise_df
            if Friedman test is not significant.
    """
    # Pivot to wide format: rows=repeats, columns=pipelines.
    wide = per_repeat.pivot(
        index="repeat", columns="pipeline", values="mean_balanced_accuracy",
    )
    present = [p for p in PIPELINE_NAMES if p in wide.columns]
    return pairwise_wilcoxon(wide, present, log)


def run_grouped_classifier_test(per_repeat, log):
    """Paired Wilcoxon test comparing NMC vs RF, pooling over feature selectors.

    For each repeat, the mean balanced accuracy of the two NMC pipelines
    (kw_nmc, en_nmc) and the two RF pipelines (kw_rf, en_rf) are computed.
    A Wilcoxon signed-rank test then compares these paired observations.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, mean_balanced_accuracy.
        log (logging.Logger): Logger instance.

    Returns:
        dict or None: Test results with keys: nmc_mean, rf_mean, mean_diff,
            statistic, p_value, n_repeats. None if insufficient data.
    """
    nmc_pipes = [p for p in ["kw_nmc", "en_nmc"] if p in per_repeat["pipeline"].values]
    rf_pipes = [p for p in ["kw_rf", "en_rf"] if p in per_repeat["pipeline"].values]

    if not nmc_pipes or not rf_pipes:
        log.warning("Cannot run grouped classifier test: need at least one NMC and one RF pipeline.")
        return None

    # Compute per-repeat mean across NMC pipelines and across RF pipelines.
    nmc_scores = (
        per_repeat[per_repeat["pipeline"].isin(nmc_pipes)]
        .groupby("repeat")["mean_balanced_accuracy"]
        .mean()
    )
    rf_scores = (
        per_repeat[per_repeat["pipeline"].isin(rf_pipes)]
        .groupby("repeat")["mean_balanced_accuracy"]
        .mean()
    )

    # Align on common repeats.
    common = nmc_scores.index.intersection(rf_scores.index)
    if len(common) < 3:
        log.warning("Fewer than 3 common repeats for grouped test; skipping.")
        return None

    nmc_vals = nmc_scores.loc[common].values
    rf_vals = rf_scores.loc[common].values

    stat_val, p_val = stats.wilcoxon(nmc_vals, rf_vals)
    mean_diff = np.mean(nmc_vals - rf_vals)

    result = {
        "nmc_mean": np.mean(nmc_vals),
        "rf_mean": np.mean(rf_vals),
        "mean_diff": round(mean_diff, 6),
        "statistic": round(stat_val, 4),
        "p_value": p_val,
        "n_repeats": len(common),
    }

    log.info(
        "Grouped classifier test (NMC vs RF, pooled over feature selectors):"
    )
    log.info(
        "  NMC mean=%.4f, RF mean=%.4f, diff=%.4f, W=%.1f, p=%.6f %s",
        result["nmc_mean"], result["rf_mean"], mean_diff,
        stat_val, p_val, "*" if p_val < 0.05 else "",
    )

    return result


def run_nadeau_bengio_tests(all_results, config, log):
    """Pairwise Nadeau-Bengio corrected resampled t-tests.

    Thin wrapper around utils.statistics.nadeau_bengio_test that pivots
    fold-level results into the required format.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, balanced_accuracy.
        config (dict): Configuration with cv.outer_folds.
        log (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame or None: Pairwise test results.
    """
    cv_cfg = config["cv"]
    k = cv_cfg.get("outer_folds", 5)

    pivot = all_results.pivot_table(
        index=["repeat", "outer_fold"],
        columns="pipeline",
        values="balanced_accuracy",
    )

    return nadeau_bengio_test(pivot, list(PIPELINE_NAMES), k, n_samples=100, log=log)


"""Plotting"""


def plot_pipeline_comparison(per_repeat, pairwise_df, fig_dir, log):
    """Violin plot comparing balanced accuracy across pipelines.

    Shows per-repeat mean balanced accuracy as violin plots with overlaid
    strip points and box plot quartile indicators. Significance brackets
    from pairwise Wilcoxon signed-rank tests (Bonferroni-corrected) are
    annotated above the violins.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        pairwise_df (pd.DataFrame or None): Pairwise Wilcoxon test results
            with columns: pipeline_a, pipeline_b, p_corrected, significant.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Order pipelines by the canonical order.
    pipelines = [p for p in PIPELINE_NAMES if p in per_repeat["pipeline"].unique()]
    data_by_pipeline = [
        per_repeat.loc[per_repeat["pipeline"] == p, "mean_balanced_accuracy"].values
        for p in pipelines
    ]
    labels = [PIPELINE_LABELS.get(p, p) for p in pipelines]
    colors = [PIPELINE_COLORS.get(p, "#888888") for p in pipelines]

    # Violin plot.
    parts = ax.violinplot(
        data_by_pipeline,
        positions=range(len(pipelines)),
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    # Overlay box plot indicators for quartiles and median.
    bp = ax.boxplot(
        data_by_pipeline,
        positions=range(len(pipelines)),
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.5),
        boxprops=dict(facecolor="black", alpha=0.6),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    # Overlay individual points with jitter.
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data_by_pipeline, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=color, edgecolors="black", linewidths=0.5,
            s=15, zorder=3, alpha=0.7,
        )

    # Add significance brackets from pairwise Wilcoxon tests.
    draw_significance_brackets(ax, pairwise_df, pipelines, data_by_pipeline)

    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean balanced accuracy (across 5 outer folds)")
    ax.set_xlabel("Pipeline")

    fig.tight_layout()
    out_path = fig_dir / "01_pipeline_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_interaction(per_repeat, fig_dir, log):
    """Interaction plot showing the 2x2 factorial design.

    X-axis: feature selection method (Kruskal-Wallis, Elastic Net).
    Two lines: NMC (simple) and RF (complex). Y-axis: mean balanced
    accuracy. Error bars show 95% confidence intervals.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Map pipelines to factorial components.
    fs_map = {"kw_nmc": "KW", "kw_rf": "KW", "en_nmc": "EN", "en_rf": "EN"}
    clf_map = {"kw_nmc": "NMC", "kw_rf": "RF", "en_nmc": "NMC", "en_rf": "RF"}

    per_repeat = per_repeat.copy()
    per_repeat["feature_selection"] = per_repeat["pipeline"].map(fs_map)
    per_repeat["classifier"] = per_repeat["pipeline"].map(clf_map)

    fs_methods = ["KW", "EN"]
    x_pos = np.array([0, 1])

    clf_styles = {
        "NMC": {"color": "#4DBBD5", "marker": "o", "label": "NMC (simple)"},
        "RF": {"color": "#E64B35", "marker": "s", "label": "RF (complex)"},
    }

    for clf_name, style in clf_styles.items():
        means = []
        ci_lower = []
        ci_upper = []

        for fs in fs_methods:
            vals = per_repeat.loc[
                (per_repeat["feature_selection"] == fs)
                & (per_repeat["classifier"] == clf_name),
                "mean_balanced_accuracy",
            ].values

            if len(vals) == 0:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                continue

            m = np.mean(vals)
            # 95% CI using t-distribution.
            se = stats.sem(vals)
            n = len(vals)
            if n > 1:
                ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
                ci_lower.append(m - ci[0])
                ci_upper.append(ci[1] - m)
            else:
                ci_lower.append(0)
                ci_upper.append(0)
            means.append(m)

        means = np.array(means)
        yerr = np.array([ci_lower, ci_upper])

        ax.errorbar(
            x_pos, means, yerr=yerr,
            color=style["color"], marker=style["marker"],
            markersize=8, linewidth=2, capsize=4, capthick=1.5,
            label=style["label"],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Kruskal-Wallis", "Elastic Net"])
    ax.set_xlabel("Feature selection method")
    ax.set_ylabel("Mean balanced accuracy")
    ax.legend(loc="best", frameon=True, edgecolor="grey")

    fig.tight_layout()
    out_path = fig_dir / "02_interaction_plot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_repeat_convergence(per_repeat, fig_dir, log):
    """Cumulative mean balanced accuracy across repeats.

    Shows how the per-pipeline mean stabilises as more repeats are
    added. Useful for verifying that the number of repeats is
    sufficient for stable estimates.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    pipelines = [p for p in PIPELINE_NAMES if p in per_repeat["pipeline"].unique()]

    for p in pipelines:
        subset = per_repeat.loc[per_repeat["pipeline"] == p].sort_values("repeat")
        scores = subset["mean_balanced_accuracy"].values
        cumulative_mean = np.cumsum(scores) / np.arange(1, len(scores) + 1)
        ax.plot(
            range(1, len(scores) + 1), cumulative_mean,
            color=PIPELINE_COLORS.get(p, "#888888"),
            label=PIPELINE_LABELS.get(p, p),
            linewidth=1.5,
        )

    ax.set_xlabel("Number of repeats included")
    ax.set_ylabel("Cumulative mean balanced accuracy")
    ax.legend(loc="best", frameon=True, edgecolor="grey")

    fig.tight_layout()
    out_path = fig_dir / "03_repeat_convergence.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_feature_importance(all_results, fig_dir, log):
    """Horizontal bar chart of the most frequently selected features.

    Counts how often each feature is selected across all outer folds
    and repeats, separately for each pipeline. Shows the top 20
    features ranked by total selection count.

    Args:
        all_results (pd.DataFrame): Fold-level results with
            selected_features column.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    if "selected_features" not in all_results.columns:
        log.warning("No selected_features column; skipping feature importance plot.")
        return

    apply_plot_style()

    # Count feature frequencies per pipeline.
    pipelines = [p for p in PIPELINE_NAMES if p in all_results["pipeline"].unique()]
    freq_by_pipeline = {}
    for p in pipelines:
        subset = all_results.loc[all_results["pipeline"] == p, "selected_features"]
        counter = Counter()
        for features_str in subset.dropna():
            counter.update(features_str.split(","))
        freq_by_pipeline[p] = counter

    # Find top 20 features by total count across all pipelines.
    total_counter = Counter()
    for c in freq_by_pipeline.values():
        total_counter.update(c)
    top_features = [f for f, _ in total_counter.most_common(20)]

    if not top_features:
        log.warning("No features found; skipping feature importance plot.")
        return

    # Grouped horizontal bar chart.
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(top_features))
    bar_height = 0.8 / len(pipelines)

    for i, p in enumerate(pipelines):
        counts = [freq_by_pipeline[p].get(f, 0) for f in top_features]
        ax.barh(
            y_pos + i * bar_height, counts, bar_height,
            color=PIPELINE_COLORS.get(p, "#888888"),
            label=PIPELINE_LABELS.get(p, p),
            alpha=0.8,
        )

    ax.set_yticks(y_pos + bar_height * (len(pipelines) - 1) / 2)
    ax.set_yticklabels(top_features, fontsize=7)
    ax.set_xlabel("Selection count (across all outer folds and repeats)")
    ax.set_ylabel("Genomic region")
    ax.legend(loc="lower right", frameon=True, edgecolor="grey", fontsize=7)
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = fig_dir / "04_feature_importance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_confusion_matrices(all_results, fig_dir, log):
    """2x2 grid of normalised confusion matrices (one per pipeline).

    Aggregates y_true/y_pred across all outer folds and repeats for
    each pipeline and plots row-normalised confusion matrices showing
    per-class recall.

    Args:
        all_results (pd.DataFrame): Fold-level results with y_true
            and y_pred columns (comma-separated class labels).
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    if "y_true" not in all_results.columns or "y_pred" not in all_results.columns:
        log.warning("No y_true/y_pred columns; skipping confusion matrix plot.")
        return

    apply_plot_style(scale="compact")

    pipelines = [p for p in PIPELINE_NAMES if p in all_results["pipeline"].unique()]
    n_pipes = len(pipelines)
    if n_pipes == 0:
        return

    # Determine subplot grid (2x2 for 4 pipelines).
    ncols = 2
    nrows = (n_pipes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 5))
    axes = np.atleast_2d(axes)

    # Collect all unique class labels in sorted order.
    all_true = []
    for s in all_results["y_true"].dropna():
        all_true.extend(s.split(","))
    class_labels = sorted(set(all_true))

    im = None
    for idx, p in enumerate(pipelines):
        ax = axes[idx // ncols, idx % ncols]
        subset = all_results.loc[all_results["pipeline"] == p]

        y_true_all = []
        y_pred_all = []
        for _, row in subset.iterrows():
            if pd.isna(row.get("y_true")) or pd.isna(row.get("y_pred")):
                continue
            y_true_all.extend(row["y_true"].split(","))
            y_pred_all.extend(row["y_pred"].split(","))

        if not y_true_all:
            ax.set_visible(False)
            continue

        cm = sklearn_confusion_matrix(y_true_all, y_pred_all, labels=class_labels)
        # Row-normalise to get recall per class.
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(float), row_sums,
            out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
        )

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        annotate_heatmap(ax, cm_norm, fmt=".2f", raw_counts=cm, threshold=0.6, fontsize=7)

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=7)
        ax.set_yticklabels(class_labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        # Pipeline label as subplot annotation (not matplotlib title).
        ax.text(
            0.5, 1.05, PIPELINE_LABELS.get(p, p),
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold",
        )

    # Hide unused axes.
    for idx in range(n_pipes, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    # Add colourbar.
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Recall (row-normalised)")

    out_path = fig_dir / "05_confusion_matrices.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


"""Error Agreement Analysis"""


def compute_error_agreement(all_results, log):
    """Compute pairwise error agreement between pipelines.

    For each pair of pipelines, measures how often they make the same
    errors on the same samples within shared (repeat, fold) splits.
    This quantifies error correlation and indicates whether ensembling
    the pipelines could improve accuracy.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, y_true, y_pred.
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (agreement_matrix, conditional_matrix, pipelines) where
            agreement_matrix is a symmetric matrix of error overlap rates
            (fraction of all samples where both pipelines are wrong),
            conditional_matrix[i,j] is P(j wrong | i wrong), and
            pipelines is the list of pipeline names.
    """
    if "y_true" not in all_results.columns or "y_pred" not in all_results.columns:
        log.warning("No y_true/y_pred columns; skipping error agreement.")
        return None, None, None

    pipelines = [p for p in PIPELINE_NAMES if p in all_results["pipeline"].unique()]
    n_pipes = len(pipelines)

    # Build a dict mapping (pipeline, repeat, fold) -> arrays of (true, pred).
    pred_dict = {}
    for _, row in all_results.iterrows():
        if pd.isna(row.get("y_true")) or pd.isna(row.get("y_pred")):
            continue
        key = (row["pipeline"], row["repeat"], row["outer_fold"])
        y_true = row["y_true"].split(",")
        y_pred = row["y_pred"].split(",")
        pred_dict[key] = (y_true, y_pred)

    # For each (repeat, fold), compute per-sample error vectors for each pipeline.
    # Then compare across pipeline pairs.
    total_samples = 0
    both_wrong = np.zeros((n_pipes, n_pipes), dtype=int)
    either_wrong = np.zeros((n_pipes, n_pipes), dtype=int)
    pipe_wrong_count = np.zeros(n_pipes, dtype=int)

    repeats = sorted(all_results["repeat"].unique())
    folds = sorted(all_results["outer_fold"].unique())

    for r in repeats:
        for f in folds:
            # Collect error masks for each pipeline in this (repeat, fold).
            error_masks = []
            valid = True
            for p in pipelines:
                key = (p, r, f)
                if key not in pred_dict:
                    valid = False
                    break
                y_true, y_pred = pred_dict[key]
                errors = np.array([t != p_ for t, p_ in zip(y_true, y_pred)])
                error_masks.append(errors)

            if not valid:
                continue

            n = len(error_masks[0])
            total_samples += n

            for i in range(n_pipes):
                pipe_wrong_count[i] += error_masks[i].sum()
                for j in range(n_pipes):
                    both_wrong[i, j] += (error_masks[i] & error_masks[j]).sum()
                    either_wrong[i, j] += (error_masks[i] | error_masks[j]).sum()

    # Agreement matrix: P(both wrong) / P(either wrong) = Jaccard index of errors.
    agreement_matrix = np.zeros((n_pipes, n_pipes))
    for i in range(n_pipes):
        for j in range(n_pipes):
            if either_wrong[i, j] > 0:
                agreement_matrix[i, j] = both_wrong[i, j] / either_wrong[i, j]

    # Conditional matrix: P(j wrong | i wrong).
    conditional_matrix = np.zeros((n_pipes, n_pipes))
    for i in range(n_pipes):
        if pipe_wrong_count[i] > 0:
            for j in range(n_pipes):
                conditional_matrix[i, j] = both_wrong[i, j] / pipe_wrong_count[i]

    # Log summary.
    log.info("Error agreement analysis (%d total sample-predictions):", total_samples)
    for i in range(n_pipes):
        error_rate = pipe_wrong_count[i] / total_samples if total_samples > 0 else 0
        log.info(
            "  %s: %d errors (%.1f%% error rate)",
            PIPELINE_LABELS.get(pipelines[i], pipelines[i]),
            pipe_wrong_count[i], error_rate * 100,
        )

    log.info("Pairwise error overlap (Jaccard index of error sets):")
    for i in range(n_pipes):
        for j in range(i + 1, n_pipes):
            log.info(
                "  %s vs %s: Jaccard=%.3f, P(%s wrong | %s wrong)=%.3f, "
                "P(%s wrong | %s wrong)=%.3f",
                PIPELINE_LABELS.get(pipelines[i], pipelines[i]),
                PIPELINE_LABELS.get(pipelines[j], pipelines[j]),
                agreement_matrix[i, j],
                PIPELINE_LABELS.get(pipelines[j], pipelines[j]),
                PIPELINE_LABELS.get(pipelines[i], pipelines[i]),
                conditional_matrix[i, j],
                PIPELINE_LABELS.get(pipelines[i], pipelines[i]),
                PIPELINE_LABELS.get(pipelines[j], pipelines[j]),
                conditional_matrix[j, i],
            )

    return agreement_matrix, conditional_matrix, pipelines


def plot_error_agreement(agreement_matrix, conditional_matrix, pipelines, fig_dir, log):
    """Heatmap of pairwise error agreement between pipelines.

    Left panel: Jaccard index of error sets (symmetric).
    Right panel: Conditional error probability P(col wrong | row wrong).

    Args:
        agreement_matrix (np.ndarray): Symmetric Jaccard matrix.
        conditional_matrix (np.ndarray): Asymmetric conditional matrix.
        pipelines (list): Pipeline name strings.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    if agreement_matrix is None:
        return

    apply_plot_style()
    labels = [PIPELINE_LABELS.get(p, p) for p in pipelines]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Jaccard index.
    im1 = ax1.imshow(agreement_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    annotate_heatmap(ax1, agreement_matrix)
    ax1.set_xticks(range(len(pipelines)))
    ax1.set_yticks(range(len(pipelines)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.text(0.5, 1.06, "Error overlap (Jaccard index)",
             transform=ax1.transAxes, ha="center", fontsize=9, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right panel: conditional error probability.
    im2 = ax2.imshow(conditional_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    annotate_heatmap(ax2, conditional_matrix)
    ax2.set_xticks(range(len(pipelines)))
    ax2.set_yticks(range(len(pipelines)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Pipeline (wrong?)", fontsize=8)
    ax2.set_ylabel("Pipeline (given wrong)", fontsize=8)
    ax2.text(0.5, 1.06, "P(column wrong | row wrong)",
             transform=ax2.transAxes, ha="center", fontsize=9, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = fig_dir / "06_error_agreement.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


"""Argument Parsing"""


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with name and config.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate and analyse nested CV results from the 2x2 design.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default_run",
        help="Run name for the results directory (default: default_run).",
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
        "--phase",
        type=str,
        default="nested_cv_2x2",
        help=(
            "Phase directory name inside the run directory that contains "
            "the fold result CSVs. Use 'hierarchical_nested_cv_2x2' for "
            "hierarchical runs. (default: nested_cv_2x2)."
        ),
    )
    return parser.parse_args()


"""Main Execution"""


def main():
    """Entry point: load fold results, compute statistics, generate figures."""
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Locate the existing run directory (do not create a new one).
    run_dir = _find_or_create_run_dir(args.name)
    nested_cv_data_dir = run_dir / args.phase / "data"

    if not nested_cv_data_dir.exists():
        print(
            f"ERROR: {args.phase}/data/ not found in {run_dir}. "
            "Run the nested CV phase first."
        )
        sys.exit(1)

    # Create output directories for this analysis phase.
    fig_dir, data_dir, log_dir, _ = get_run_dirs(args.name, "nested_cv_analysis")
    log, console = setup_logging("analyse_nested_cv", log_dir=log_dir)

    log.info("Run directory: %s", run_dir)
    log.info("Reading fold results from: %s", nested_cv_data_dir)

    """Step 1: Load and Aggregate Results"""

    all_results = load_fold_results(nested_cv_data_dir, log)

    # Save aggregated results.
    all_results_path = data_dir / "all_fold_results.csv"
    all_results.to_csv(all_results_path, index=False)
    log.info("Saved aggregated fold results: %s", all_results_path)

    """Step 2: Compute Per-Repeat Means"""

    per_repeat = compute_per_repeat_means(all_results)
    log.info("Computed per-repeat means for %d pipeline-repeat pairs.", len(per_repeat))

    """Step 3: Summary Statistics"""

    summary = compute_summary_statistics(per_repeat)
    summary_path = data_dir / "summary_statistics.csv"
    summary.to_csv(summary_path)
    log.info("Summary statistics:")
    for pipeline in summary.index:
        row = summary.loc[pipeline]
        log.info(
            "  %s: mean=%.4f +/- %.4f, median=%.4f, n_repeats=%d",
            PIPELINE_LABELS.get(pipeline, pipeline),
            row["mean_bal_acc"], row["std_bal_acc"],
            row["median_bal_acc"], int(row["n_repeats"]),
        )
    log.info("Saved summary statistics: %s", summary_path)

    """Step 4: Data Completeness Check"""

    # Warn about incomplete runs.
    expected_pipelines = set(config.get("pipelines", {}).get("names", PIPELINE_NAMES))
    actual_pipelines = set(all_results["pipeline"].unique())
    missing_pipelines = expected_pipelines - actual_pipelines
    if missing_pipelines:
        log.warning(
            "Missing pipelines (incomplete run?): %s", sorted(missing_pipelines),
        )

    expected_repeats = config.get("cv", {}).get("n_repeats", None)
    if expected_repeats is not None:
        for p in actual_pipelines:
            actual = all_results.loc[all_results["pipeline"] == p, "repeat"].nunique()
            if actual < expected_repeats:
                log.warning(
                    "Pipeline %s has %d/%d repeats.", p, actual, expected_repeats,
                )

    """Step 5: Statistical Testing (Friedman + Wilcoxon)"""

    friedman_stat, friedman_p, pairwise_df = run_statistical_tests(per_repeat, log)

    if pairwise_df is not None:
        pairwise_path = data_dir / "pairwise_wilcoxon.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        log.info("Saved pairwise Wilcoxon tests: %s", pairwise_path)
    elif friedman_p is not None:
        pairwise_path = data_dir / "pairwise_wilcoxon.csv"
        pd.DataFrame({
            "note": [f"Friedman test not significant (p={friedman_p:.6f}); "
                     "no pairwise tests performed."]
        }).to_csv(pairwise_path, index=False)
        log.info("Saved pairwise tests (no significant differences): %s", pairwise_path)

    """Step 5b: Grouped Classifier Test (NMC vs RF)"""

    grouped_result = run_grouped_classifier_test(per_repeat, log)
    if grouped_result is not None:
        grouped_path = data_dir / "grouped_nmc_vs_rf.csv"
        pd.DataFrame([grouped_result]).to_csv(grouped_path, index=False)
        log.info("Saved grouped NMC vs RF test: %s", grouped_path)

    """Step 6: Statistical Testing (Nadeau-Bengio Corrected t-test)"""

    nb_df = run_nadeau_bengio_tests(all_results, config, log)
    if nb_df is not None:
        nb_path = data_dir / "pairwise_nadeau_bengio.csv"
        nb_df.to_csv(nb_path, index=False)
        log.info("Saved Nadeau-Bengio tests: %s", nb_path)

    """Step 7: Identify Winner"""

    winner = summary.index[0]
    log.info(
        "Winning pipeline: %s (mean balanced accuracy = %.4f)",
        PIPELINE_LABELS.get(winner, winner),
        summary.loc[winner, "mean_bal_acc"],
    )

    """Step 8: Error Agreement Analysis"""

    agreement_matrix, conditional_matrix, agree_pipelines = compute_error_agreement(
        all_results, log,
    )

    # Save error agreement data.
    if agreement_matrix is not None:
        agree_labels = [PIPELINE_LABELS.get(p, p) for p in agree_pipelines]
        pd.DataFrame(
            agreement_matrix, index=agree_labels, columns=agree_labels,
        ).to_csv(data_dir / "error_agreement_jaccard.csv")
        pd.DataFrame(
            conditional_matrix, index=agree_labels, columns=agree_labels,
        ).to_csv(data_dir / "error_agreement_conditional.csv")
        log.info("Saved error agreement matrices.")

    """Step 9: Generate Figures"""

    plot_pipeline_comparison(per_repeat, pairwise_df, fig_dir, log)
    plot_interaction(per_repeat, fig_dir, log)
    plot_repeat_convergence(per_repeat, fig_dir, log)
    plot_feature_importance(all_results, fig_dir, log)
    plot_confusion_matrices(all_results, fig_dir, log)
    plot_error_agreement(agreement_matrix, conditional_matrix, agree_pipelines, fig_dir, log)

    """Step 10: Save Config Snapshot"""

    save_config(
        run_dir, "analyse_nested_cv",
        config_file=config["_config_path"],
        n_fold_files=len(list(nested_cv_data_dir.glob("fold_results_*.csv"))),
        n_total_folds=len(all_results),
        n_pipelines=all_results["pipeline"].nunique(),
        n_repeats=all_results["repeat"].nunique(),
        winning_pipeline=winner,
        winning_mean_bal_acc=float(summary.loc[winner, "mean_bal_acc"]),
        friedman_p=float(friedman_p) if friedman_p is not None else None,
    )

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
