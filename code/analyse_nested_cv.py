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
    pipeline, and produces five publication-quality figures.

Usage:
    python3 code/analyse_nested_cv.py --name default_run --config local

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
from utils.cv_config import PIPELINE_NAMES
from utils.logging_setup import setup_logging
from utils.paths import _find_or_create_run_dir, get_run_dirs, save_config
from utils.plotting import apply_plot_style

rich.traceback.install()

# Pipeline display names for figures and tables.
PIPELINE_LABELS = {
    "kw_nmc": "KW + NMC",
    "kw_rf": "KW + RF",
    "en_nmc": "EN + NMC",
    "en_rf": "EN + RF",
}

# Pipeline colours (consistent 4-colour palette).
PIPELINE_COLORS = {
    "kw_nmc": "#4DBBD5",
    "kw_rf": "#E64B35",
    "en_nmc": "#00A087",
    "en_rf": "#F39B7F",
}


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

    The Friedman test is a non-parametric repeated-measures alternative
    to repeated-measures ANOVA, appropriate here because each repeat
    provides paired observations across all four pipelines.

    If the Friedman test is significant (p < 0.05), pairwise Wilcoxon
    signed-rank tests are performed with Bonferroni correction for 6
    comparisons.

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

    # Ensure all four pipelines are present.
    present = [p for p in PIPELINE_NAMES if p in wide.columns]
    if len(present) < 2:
        log.warning("Fewer than 2 pipelines found; skipping statistical tests.")
        return None, None, None

    # Friedman test.
    samples = [wide[p].values for p in present]
    friedman_stat, friedman_p = stats.friedmanchisquare(*samples)
    log.info("Friedman test: chi2=%.4f, p=%.6f", friedman_stat, friedman_p)

    if friedman_p >= 0.05:
        log.info("Friedman test not significant (p >= 0.05); skipping pairwise tests.")
        return friedman_stat, friedman_p, None

    # Pairwise Wilcoxon signed-rank tests with Bonferroni correction.
    n_comparisons = len(present) * (len(present) - 1) // 2
    pairwise_rows = []

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            stat_val, p_val = stats.wilcoxon(wide[a].values, wide[b].values)
            p_corrected = min(p_val * n_comparisons, 1.0)
            pairwise_rows.append({
                "pipeline_a": a,
                "pipeline_b": b,
                "statistic": round(stat_val, 4),
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            })
            log.info(
                "  %s vs %s: W=%.1f, p=%.6f, p_corrected=%.6f %s",
                a, b, stat_val, p_val, p_corrected,
                "*" if p_corrected < 0.05 else "",
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    return friedman_stat, friedman_p, pairwise_df


def run_nadeau_bengio_tests(all_results, config, log):
    """Pairwise Nadeau-Bengio corrected resampled t-tests.

    The Nadeau-Bengio correction accounts for the non-independence
    of test sets in repeated k-fold CV, preventing inflation of the
    test statistic that occurs with naive paired t-tests on
    overlapping folds.

    Reference: Nadeau & Bengio (2003), "Inference for the
    Generalization Error", Machine Learning 52(3):239-281.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, balanced_accuracy.
        config (dict): Configuration with cv.outer_folds.
        log (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame or None: Pairwise test results with columns:
            pipeline_a, pipeline_b, mean_diff, t_statistic, df,
            p_value, p_corrected, significant. None if fewer than
            2 pipelines.
    """
    cv_cfg = config["cv"]
    k = cv_cfg.get("outer_folds", 5)
    n_samples = 100  # Fixed for this dataset.
    n_test = n_samples // k
    n_train = n_samples - n_test

    # Pivot to get scores indexed by (repeat, fold) for each pipeline.
    pivot = all_results.pivot_table(
        index=["repeat", "outer_fold"],
        columns="pipeline",
        values="balanced_accuracy",
    )

    present = [p for p in PIPELINE_NAMES if p in pivot.columns]
    if len(present) < 2:
        log.warning("Fewer than 2 pipelines; skipping Nadeau-Bengio tests.")
        return None

    n_comparisons = len(present) * (len(present) - 1) // 2
    r = pivot.index.get_level_values("repeat").nunique()
    kr = k * r
    correction = 1.0 / kr + n_test / n_train

    log.info(
        "Nadeau-Bengio tests: k=%d folds, r=%d repeats, "
        "n_test=%d, n_train=%d, correction_factor=%.4f",
        k, r, n_test, n_train, correction,
    )

    rows = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            # Compute pairwise differences at the fold level.
            d = (pivot[a] - pivot[b]).dropna().values
            d_bar = np.mean(d)
            s2 = np.var(d, ddof=1)

            # Corrected variance (Nadeau-Bengio equation 8).
            sigma2 = correction * s2
            if sigma2 == 0:
                t_stat = 0.0
                p_val = 1.0
            else:
                t_stat = d_bar / np.sqrt(sigma2)
                p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=kr - 1))

            p_corrected = min(p_val * n_comparisons, 1.0)
            rows.append({
                "pipeline_a": a,
                "pipeline_b": b,
                "mean_diff": round(d_bar, 6),
                "t_statistic": round(t_stat, 4),
                "df": kr - 1,
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            })
            log.info(
                "  %s vs %s: diff=%.4f, t=%.3f, df=%d, p=%.6f, "
                "p_corrected=%.6f %s",
                a, b, d_bar, t_stat, kr - 1, p_val, p_corrected,
                "*" if p_corrected < 0.05 else "",
            )

    return pd.DataFrame(rows)


"""Plotting"""


def plot_pipeline_comparison(per_repeat, pairwise_df, fig_dir, log):
    """Box plot comparing balanced accuracy across pipelines.

    Shows per-repeat mean balanced accuracy as box plots with overlaid
    strip points. Significance brackets from pairwise tests are annotated
    above the boxes.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        pairwise_df (pd.DataFrame or None): Pairwise test results.
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

    # Box plot.
    bp = ax.boxplot(
        data_by_pipeline,
        positions=range(len(pipelines)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points with jitter.
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data_by_pipeline, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=color, edgecolors="black", linewidths=0.5,
            s=20, zorder=3, alpha=0.8,
        )

    # Add significance brackets if pairwise tests were run.
    if pairwise_df is not None:
        sig_pairs = pairwise_df[pairwise_df["significant"]]
        if not sig_pairs.empty:
            y_max = max(v.max() for v in data_by_pipeline if len(v) > 0)
            y_range = y_max - min(v.min() for v in data_by_pipeline if len(v) > 0)
            bracket_height = y_range * 0.05
            y_offset = y_max + y_range * 0.08

            for idx, row in sig_pairs.iterrows():
                x1 = pipelines.index(row["pipeline_a"])
                x2 = pipelines.index(row["pipeline_b"])
                y_bar = y_offset + idx * bracket_height * 2.5

                ax.plot(
                    [x1, x1, x2, x2],
                    [y_bar, y_bar + bracket_height, y_bar + bracket_height, y_bar],
                    color="black", linewidth=0.8,
                )
                # Format p-value for annotation.
                p_corr = row["p_corrected"]
                if p_corr < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_corr:.3f}"
                ax.text(
                    (x1 + x2) / 2, y_bar + bracket_height,
                    p_text, ha="center", va="bottom", fontsize=7,
                )

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

    This directly visualises main effects and interaction, tying to
    the Wessels et al. research question.

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

        # Annotate cells with both normalised and raw counts.
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text_color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(
                    j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color=text_color,
                )

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
    return parser.parse_args()


"""Main Execution"""


def main():
    """Entry point: load fold results, compute statistics, generate figures."""
    args = parse_args()

    # Load experiment configuration.
    config = load_config(args.config)

    # Locate the existing run directory (do not create a new one).
    run_dir = _find_or_create_run_dir(args.name)
    nested_cv_data_dir = run_dir / "nested_cv_2x2" / "data"

    if not nested_cv_data_dir.exists():
        print(
            f"ERROR: nested_cv_2x2/data/ not found in {run_dir}. "
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

    """Step 8: Generate Figures"""

    plot_pipeline_comparison(per_repeat, pairwise_df, fig_dir, log)
    plot_interaction(per_repeat, fig_dir, log)
    plot_repeat_convergence(per_repeat, fig_dir, log)
    plot_feature_importance(all_results, fig_dir, log)
    plot_confusion_matrices(all_results, fig_dir, log)

    """Step 9: Save Config Snapshot"""

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
