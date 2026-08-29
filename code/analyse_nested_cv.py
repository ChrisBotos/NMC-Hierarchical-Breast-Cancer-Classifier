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
    python3 code/analyse_nested_cv.py --name server_run --config configs/server.yaml --phase hierarchical_nested_cv

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""

"""Imports and Configuration"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback
from scipy import stats
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from statsmodels.stats.multitest import multipletests

# Ensure the code/ directory is on sys.path so utils is importable.
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.config_loader import load_config
from utils.constants import PIPELINE_COLORS, PIPELINE_LABELS, PIPELINE_NAMES
from utils.cv_config import FLAT_PIPELINE_NAMES
from utils.logging_setup import setup_logging
from utils.paths import _find_or_create_run_dir, get_run_dirs, save_config
from utils.plotting import (
    annotate_heatmap,
    apply_plot_style,
    draw_significance_brackets,
)
from utils.statistics import nadeau_bengio_test, pairwise_wilcoxon

rich.traceback.install()

# Auto-generated fallback colours for pipelines not in PIPELINE_COLORS.
_FALLBACK_COLORS = {}
_TAB10 = plt.cm.tab10


def get_pipeline_color(pipeline_name):
    """Return a colour for a pipeline, generating one if not predefined.

    Known pipelines use their entry in PIPELINE_COLORS. Unknown ones get
    a distinct colour from the tab10 colourmap so they are visually
    distinguishable rather than all falling back to the same grey.

    Args:
        pipeline_name (str): Internal pipeline name.

    Returns:
        str: Hex colour string.
    """
    if pipeline_name in PIPELINE_COLORS:
        return PIPELINE_COLORS[pipeline_name]
    if pipeline_name not in _FALLBACK_COLORS:
        idx = len(_FALLBACK_COLORS)
        rgba = _TAB10(idx % 10)
        _FALLBACK_COLORS[pipeline_name] = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
        )
    return _FALLBACK_COLORS[pipeline_name]


"""Pipeline Ordering"""


def get_pipeline_order(phase, all_results):
    """Determine canonical pipeline ordering based on phase and available data.

    For hierarchical phases, uses PIPELINE_NAMES ordering. Falls back
    to hierarchical ordering if any hierarchical-only pipeline names are found
    in the data. Otherwise uses the original 4-pipeline ordering.

    Any pipelines present in the data but absent from the canonical ordering
    (e.g. filename-derived variants like en_nmc_pens_p50) are appended in
    sorted order so they are not silently excluded from analysis.

    Args:
        phase (str): Phase directory name (e.g. 'nested_cv_2x2',
            'hierarchical_nested_cv').
        all_results (pd.DataFrame): Loaded fold results with pipeline column.

    Returns:
        tuple: Pipeline name ordering covering all pipelines in the data.
    """
    actual_pipelines = set(all_results["pipeline"].unique())
    hierarchical_only = {
        "nmc_ensemble",
        "kw_nmc_pens",
        "standalone_en_pens",
        "en_nmc_pens",
        "nmc_pens_ensemble",
        "standalone_en",
    }

    if phase in ("hierarchical_nested_cv",) or actual_pipelines & hierarchical_only:
        canonical = PIPELINE_NAMES
    else:
        canonical = FLAT_PIPELINE_NAMES

    # Append any data-present pipelines not in the canonical ordering.
    extra = sorted(actual_pipelines - set(canonical))
    if extra:
        return tuple(list(canonical) + extra)
    return canonical


"""Data Loading and Aggregation"""


def _extract_pipeline_from_filename(filepath):
    """Extract the full pipeline name (including variant suffix) from a fold results filename.

    Filenames follow the pattern: fold_results_<pipeline_name>_r<repeat>.csv
    where <pipeline_name> may include variant suffixes like _p50, _p80.

    Args:
        filepath (Path): Path to a fold_results CSV file.

    Returns:
        str or None: Extracted pipeline name, or None if the filename
            does not match the expected pattern.
    """
    match = re.match(r"fold_results_(.+)_r(\d+)\.csv", filepath.name)
    if match:
        return match.group(1)
    return None


def load_fold_results(nested_cv_data_dir, log):
    """Load and concatenate all fold result CSVs from the nested CV phase.

    Globs for fold_results_*.csv files in the given directory and
    concatenates them into a single DataFrame. Pipeline names are derived
    from filenames rather than CSV contents to correctly disambiguate
    variants (e.g. en_nmc_pens_p50 vs en_nmc_pens_p80) that share the
    same internal pipeline column value.

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
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Derive the pipeline name from the filename to disambiguate variants.
        fname_pipeline = _extract_pipeline_from_filename(f)
        if fname_pipeline is not None:
            df["_fname_pipeline"] = fname_pipeline
        dfs.append(df)
    all_results = pd.concat(dfs, ignore_index=True)

    # Harmonise column names from hierarchical runner to match flat runner.
    if (
        "stage2_pipeline" in all_results.columns
        and "pipeline" not in all_results.columns
    ):
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

    # Override pipeline column with filename-derived name when variants exist.
    if "_fname_pipeline" in all_results.columns:
        internal_names = all_results["pipeline"].unique()
        fname_names = all_results["_fname_pipeline"].unique()
        if len(fname_names) > len(internal_names):
            log.info(
                "Filename-derived pipeline names (%d) exceed internal names (%d); "
                "using filename variants for disambiguation.",
                len(fname_names),
                len(internal_names),
            )
            all_results["pipeline"] = all_results["_fname_pipeline"]
        all_results = all_results.drop(columns=["_fname_pipeline"])

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
    each repeat gives one balanced accuracy per pipeline. When stage2_bal_acc
    is present (hierarchical runs), it is also aggregated.

    Args:
        all_results (pd.DataFrame): Raw fold-level results.

    Returns:
        pd.DataFrame: Columns: pipeline, repeat, mean_balanced_accuracy,
            mean_auroc_macro, mean_n_features, and optionally
            mean_stage2_bal_acc.
    """
    agg_dict = {
        "mean_balanced_accuracy": ("balanced_accuracy", "mean"),
        "mean_auroc_macro": ("auroc_macro", "mean"),
        "mean_n_features": ("n_features_selected", "mean"),
    }
    if "stage2_bal_acc" in all_results.columns:
        agg_dict["mean_stage2_bal_acc"] = ("stage2_bal_acc", "mean")

    grouped = all_results.groupby(["pipeline", "repeat"], as_index=False).agg(
        **agg_dict
    )
    return grouped


def compute_summary_statistics(per_repeat, metric="mean_balanced_accuracy"):
    """Compute overall summary statistics per pipeline.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        metric (str): Column name to summarise (default:
            mean_balanced_accuracy).

    Returns:
        pd.DataFrame: One row per pipeline with mean, std, median, min, max
            for the chosen metric and AUROC.
    """
    summary = per_repeat.groupby("pipeline").agg(
        mean_bal_acc=(metric, "mean"),
        std_bal_acc=(metric, "std"),
        median_bal_acc=(metric, "median"),
        min_bal_acc=(metric, "min"),
        max_bal_acc=(metric, "max"),
        mean_auroc=("mean_auroc_macro", "mean"),
        std_auroc=("mean_auroc_macro", "std"),
        mean_n_features=("mean_n_features", "mean"),
        n_repeats=(metric, "count"),
    )
    # Sort by mean balanced accuracy descending.
    summary = summary.sort_values("mean_bal_acc", ascending=False)
    return summary


"""Statistical Testing"""


def run_statistical_tests(
    per_repeat, log, pipeline_order=PIPELINE_NAMES, metric="mean_balanced_accuracy"
):
    """Run Friedman test and pairwise Wilcoxon signed-rank tests.

    Thin wrapper around utils.statistics.pairwise_wilcoxon that pivots
    per-repeat data into the required wide format.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, and the metric column.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering to use.
        metric (str): Column name in per_repeat to compare.

    Returns:
        tuple: (friedman_stat, friedman_p, pairwise_df) where pairwise_df
            is a DataFrame with columns: pipeline_a, pipeline_b, statistic,
            p_value, p_corrected, significant. Returns None for pairwise_df
            if Friedman test is not significant.
    """
    # Pivot to wide format: rows=repeats, columns=pipelines.
    wide = per_repeat.pivot(
        index="repeat",
        columns="pipeline",
        values=metric,
    )
    present = [p for p in pipeline_order if p in wide.columns]
    return pairwise_wilcoxon(wide, present, log)


def run_grouped_classifier_test(per_repeat, log, metric="mean_balanced_accuracy"):
    """Paired Wilcoxon test comparing NMC vs RF, pooling over feature selectors.

    For each repeat, the mean balanced accuracy of the two NMC pipelines
    (kw_nmc, en_nmc) and the two RF pipelines (kw_rf, en_rf) are computed.
    A Wilcoxon signed-rank test then compares these paired observations.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, and the metric column.
        log (logging.Logger): Logger instance.
        metric (str): Column name in per_repeat to compare.

    Returns:
        dict or None: Test results with keys: nmc_mean, rf_mean, mean_diff,
            statistic, p_value, n_repeats. None if insufficient data.
    """
    nmc_pipes = [p for p in ["kw_nmc", "en_nmc"] if p in per_repeat["pipeline"].values]
    rf_pipes = [p for p in ["kw_rf", "en_rf"] if p in per_repeat["pipeline"].values]

    if not nmc_pipes or not rf_pipes:
        log.warning(
            "Cannot run grouped classifier test: need at least one NMC and one RF pipeline."
        )
        return None

    # Compute per-repeat mean across NMC pipelines and across RF pipelines.
    nmc_scores = (
        per_repeat[per_repeat["pipeline"].isin(nmc_pipes)]
        .groupby("repeat")[metric]
        .mean()
    )
    rf_scores = (
        per_repeat[per_repeat["pipeline"].isin(rf_pipes)]
        .groupby("repeat")[metric]
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

    log.info("Grouped classifier test (NMC vs RF, pooled over feature selectors):")
    log.info(
        "  NMC mean=%.4f, RF mean=%.4f, diff=%.4f, W=%.1f, p=%.6f %s",
        result["nmc_mean"],
        result["rf_mean"],
        mean_diff,
        stat_val,
        p_val,
        "*" if p_val < 0.05 else "",
    )

    return result


# Pre-registered primary comparisons (uncorrected).
PREREGISTERED_PAIRS = [
    ("kw_nmc", "en_nmc"),  # Feature selection effect on simple classifier.
    ("kw_rf", "en_rf"),  # Feature selection effect on complex classifier.
]


def run_preregistered_tests(per_repeat, summary, log, metric="mean_balanced_accuracy"):
    """Run pre-registered primary comparisons (Wilcoxon, uncorrected).

    These comparisons test the core research question (Wessels replication)
    and were specified before seeing the final results. The grouped NMC vs RF
    test is handled by run_grouped_classifier_test separately.

    Additionally tests the best base pipeline against its plateau ensemble
    variant (if present).

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, and the metric column.
        summary (pd.DataFrame): Summary statistics indexed by pipeline name
            (sorted by mean_bal_acc descending).
        log (logging.Logger): Logger instance.
        metric (str): Column name in per_repeat to compare.

    Returns:
        pd.DataFrame: Pre-registered test results with columns: pipeline_a,
            pipeline_b, comparison_type, statistic, p_value, p_corrected,
            significant. p_corrected equals p_value (no correction).
    """
    wide = per_repeat.pivot(
        index="repeat",
        columns="pipeline",
        values=metric,
    )
    available = set(wide.columns)
    rows = []

    log.info("=== Pre-registered Primary Comparisons (uncorrected) ===")

    # Fixed pairwise comparisons from the research design.
    for a, b in PREREGISTERED_PAIRS:
        if a not in available or b not in available:
            log.info("  Skipping %s vs %s (not in data).", a, b)
            continue
        clean = wide[[a, b]].dropna()
        if len(clean) < 3:
            log.info("  Skipping %s vs %s (fewer than 3 repeats).", a, b)
            continue
        stat_val, p_val = stats.wilcoxon(clean[a].values, clean[b].values)
        rows.append(
            {
                "pipeline_a": a,
                "pipeline_b": b,
                "comparison_type": "preregistered",
                "statistic": round(stat_val, 4),
                "p_value": p_val,
                "p_corrected": p_val,
                "significant": p_val < 0.05,
            }
        )
        log.info(
            "  %s vs %s: W=%.1f, p=%.6f %s",
            PIPELINE_LABELS.get(a, a),
            PIPELINE_LABELS.get(b, b),
            stat_val,
            p_val,
            "*" if p_val < 0.05 else "",
        )

    # Best base pipeline vs its plateau ensemble variant.
    base_to_pens = {
        "kw_nmc": "kw_nmc_pens",
        "en_nmc": "en_nmc_pens",
        "standalone_en": "standalone_en_pens",
    }
    for pipe in summary.index:
        if pipe in base_to_pens and base_to_pens[pipe] in available:
            pens_pipe = base_to_pens[pipe]
            if pipe not in available:
                continue
            clean = wide[[pipe, pens_pipe]].dropna()
            if len(clean) < 3:
                continue
            stat_val, p_val = stats.wilcoxon(
                clean[pipe].values,
                clean[pens_pipe].values,
            )
            rows.append(
                {
                    "pipeline_a": pipe,
                    "pipeline_b": pens_pipe,
                    "comparison_type": "preregistered_pens",
                    "statistic": round(stat_val, 4),
                    "p_value": p_val,
                    "p_corrected": p_val,
                    "significant": p_val < 0.05,
                }
            )
            log.info(
                "  %s vs %s (plateau): W=%.1f, p=%.6f %s",
                PIPELINE_LABELS.get(pipe, pipe),
                PIPELINE_LABELS.get(pens_pipe, pens_pipe),
                stat_val,
                p_val,
                "*" if p_val < 0.05 else "",
            )
            break  # Only test the best base pipeline.

    return pd.DataFrame(rows) if rows else None


def run_exploratory_tests(
    per_repeat,
    log,
    pipeline_order=PIPELINE_NAMES,
    metric="mean_balanced_accuracy",
    exclude_pairs=None,
):
    """Run exploratory pairwise Wilcoxon tests (Bonferroni-corrected).

    Performs all pairwise Wilcoxon signed-rank tests except those already
    tested as pre-registered comparisons. Applies Bonferroni correction
    across only the exploratory comparisons.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
        metric (str): Column name in per_repeat to compare.
        exclude_pairs (set or None): Set of (a, b) tuples to skip
            (already tested in pre-registered).

    Returns:
        pd.DataFrame or None: Exploratory test results with columns:
            pipeline_a, pipeline_b, comparison_type, statistic, p_value,
            p_corrected, significant.
    """
    wide = per_repeat.pivot(
        index="repeat",
        columns="pipeline",
        values=metric,
    )
    present = [p for p in pipeline_order if p in wide.columns]
    if len(present) < 2:
        return None

    exclude = exclude_pairs or set()
    clean = wide[present].dropna()
    if len(clean) < 3:
        log.warning("Fewer than 3 complete rows; skipping exploratory tests.")
        return None

    # Collect all exploratory pairs.
    exploratory_pairs = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            pair = tuple(sorted((a, b)))
            if pair not in exclude:
                exploratory_pairs.append((a, b))

    if not exploratory_pairs:
        return None

    n_comparisons = len(exploratory_pairs)
    log.info(
        "=== Exploratory Comparisons (%d tests, Bonferroni-corrected) ===",
        n_comparisons,
    )

    rows = []
    for a, b in exploratory_pairs:
        stat_val, p_val = stats.wilcoxon(clean[a].values, clean[b].values)
        p_corrected = min(p_val * n_comparisons, 1.0)
        rows.append(
            {
                "pipeline_a": a,
                "pipeline_b": b,
                "comparison_type": "exploratory",
                "statistic": round(stat_val, 4),
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            }
        )
        log.info(
            "  %s vs %s: W=%.1f, p=%.6f, p_corrected=%.6f %s",
            a,
            b,
            stat_val,
            p_val,
            p_corrected,
            "*" if p_corrected < 0.05 else "",
        )

    return pd.DataFrame(rows) if rows else None


def run_nadeau_bengio_tests(
    all_results,
    config,
    log,
    pipeline_order=PIPELINE_NAMES,
    fold_metric="balanced_accuracy",
):
    """Pairwise Nadeau-Bengio corrected resampled t-tests.

    Thin wrapper around utils.statistics.nadeau_bengio_test that pivots
    fold-level results into the required format.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, and the fold_metric column.
        config (dict): Configuration with cv.outer_folds.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering to use.
        fold_metric (str): Column name in all_results for fold-level scores.

    Returns:
        pd.DataFrame or None: Pairwise test results.
    """
    cv_cfg = config["cv"]
    k = cv_cfg.get("outer_folds", 5)

    pivot = all_results.pivot_table(
        index=["repeat", "outer_fold"],
        columns="pipeline",
        values=fold_metric,
    )

    present = [p for p in pipeline_order if p in pivot.columns]
    return nadeau_bengio_test(pivot, present, k, n_samples=100, log=log)


def run_bootstrap_ci_tests(
    per_repeat,
    log,
    pipeline_order=PIPELINE_NAMES,
    metric="mean_balanced_accuracy",
    n_bootstrap=10000,
    alpha=0.05,
    seed=42,
):
    """Paired bootstrap confidence intervals on mean differences.

    For each pair of pipelines, computes the mean of paired differences
    and a bootstrap 95% CI. If the CI excludes zero, the pipelines are
    distinguishable.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores with columns
            pipeline, repeat, and the metric column.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
        metric (str): Column name in per_repeat to compare.
        n_bootstrap (int): Number of bootstrap resamples.
        alpha (float): Significance level for CI (default 0.05 for 95% CI).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame or None: Pairwise bootstrap results with columns:
            pipeline_a, pipeline_b, mean_diff, ci_lower, ci_upper,
            bootstrap_se, distinguishable.
    """
    wide = per_repeat.pivot(
        index="repeat",
        columns="pipeline",
        values=metric,
    )
    present = [p for p in pipeline_order if p in wide.columns]
    if len(present) < 2:
        return None

    rng = np.random.default_rng(seed)
    rows = []

    log.info(
        "=== Paired Bootstrap CIs (%d resamples, %d%% CI) ===",
        n_bootstrap,
        int((1 - alpha) * 100),
    )

    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            clean = wide[[a, b]].dropna()
            diffs = clean[a].values - clean[b].values
            observed_mean = diffs.mean()
            n = len(diffs)

            # Bootstrap the mean difference.
            boot_means = np.empty(n_bootstrap)
            for bi in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot_means[bi] = diffs[idx].mean()

            ci_lo = np.percentile(boot_means, 100 * alpha / 2)
            ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
            boot_se = boot_means.std()
            distinguishable = (ci_lo > 0) or (ci_hi < 0)

            rows.append(
                {
                    "pipeline_a": a,
                    "pipeline_b": b,
                    "mean_diff": round(observed_mean, 6),
                    "ci_lower": round(ci_lo, 6),
                    "ci_upper": round(ci_hi, 6),
                    "bootstrap_se": round(boot_se, 6),
                    "distinguishable": distinguishable,
                }
            )
            log.info(
                "  %s vs %s: diff=%+.4f, 95%% CI=[%+.4f, %+.4f], SE=%.4f %s",
                PIPELINE_LABELS.get(a, a),
                PIPELINE_LABELS.get(b, b),
                observed_mean,
                ci_lo,
                ci_hi,
                boot_se,
                "*" if distinguishable else "",
            )

    return pd.DataFrame(rows) if rows else None


"""Plotting"""


def plot_pipeline_comparison(
    per_repeat,
    pairwise_df,
    fig_dir,
    log,
    pipeline_order=PIPELINE_NAMES,
    metric="mean_balanced_accuracy",
    metric_label="Mean balanced accuracy (across 5 outer folds)",
):
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
        pipeline_order (tuple): Canonical pipeline ordering.
        metric (str): Column name in per_repeat to plot.
        metric_label (str): Y-axis label for the metric.
    """
    apply_plot_style()
    n_pipes = len([p for p in pipeline_order if p in per_repeat["pipeline"].unique()])
    fig_width = max(6, n_pipes * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # Order pipelines by the canonical order.
    pipelines = [p for p in pipeline_order if p in per_repeat["pipeline"].unique()]
    data_by_pipeline = [
        per_repeat.loc[per_repeat["pipeline"] == p, metric].values for p in pipelines
    ]
    labels = [PIPELINE_LABELS.get(p, p) for p in pipelines]
    colors = [get_pipeline_color(p) for p in pipelines]

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
    ax.boxplot(
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
            np.full(len(vals), i) + jitter,
            vals,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            s=15,
            zorder=3,
            alpha=0.7,
        )

    # Add significance brackets from pairwise Wilcoxon tests.
    draw_significance_brackets(ax, pairwise_df, pipelines, data_by_pipeline)

    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Pipeline")

    fig.tight_layout()
    out_path = fig_dir / "01_pipeline_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_interaction(
    per_repeat,
    fig_dir,
    log,
    pipeline_order=PIPELINE_NAMES,
    metric="mean_balanced_accuracy",
    metric_label="Mean balanced accuracy",
):
    """Interaction plot showing the 2x2 factorial design.

    X-axis: feature selection method (Kruskal-Wallis, Elastic Net).
    Two lines: NMC (simple) and RF (complex). Y-axis: mean balanced
    accuracy. Error bars show 95% confidence intervals.

    Only drawn when all 4 baseline pipelines (kw_nmc, kw_rf, en_nmc, en_rf)
    are present in the data. Skipped for phases with non-baseline variants.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
        metric (str): Column name in per_repeat to plot.
        metric_label (str): Y-axis label for the metric.
    """
    # Only draw for phases with all 4 baseline pipelines.
    baseline_pipes = {"kw_nmc", "kw_rf", "en_nmc", "en_rf"}
    actual = set(per_repeat["pipeline"].unique())
    if not baseline_pipes.issubset(actual):
        log.info("Skipping interaction plot: not all 4 baseline pipelines present.")
        return

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
                metric,
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
            x_pos,
            means,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            label=style["label"],
        )

    # Add standalone EN as a single point at the EN x-position.
    if "standalone_en" in actual:
        se_vals = per_repeat.loc[
            per_repeat["pipeline"] == "standalone_en",
            metric,
        ].values
        if len(se_vals) > 0:
            m = np.mean(se_vals)
            se = stats.sem(se_vals)
            n = len(se_vals)
            if n > 1:
                ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
                yerr_se = np.array([[m - ci[0]], [ci[1] - m]])
            else:
                yerr_se = np.array([[0], [0]])
            ax.errorbar(
                [1],
                [m],
                yerr=yerr_se,
                color="#B09C85",
                marker="D",
                markersize=8,
                linewidth=0,
                capsize=4,
                capthick=1.5,
                label="Standalone EN",
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Kruskal-Wallis", "Elastic Net"])
    ax.set_xlabel("Feature selection method")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", frameon=True, edgecolor="grey")

    fig.tight_layout()
    out_path = fig_dir / "02_interaction_plot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_repeat_convergence(
    per_repeat,
    fig_dir,
    log,
    pipeline_order=PIPELINE_NAMES,
    metric="mean_balanced_accuracy",
    metric_label="Cumulative mean balanced accuracy",
):
    """Cumulative mean balanced accuracy across repeats.

    Shows how the per-pipeline mean stabilises as more repeats are
    added. Useful for verifying that the number of repeats is
    sufficient for stable estimates.

    Args:
        per_repeat (pd.DataFrame): Per-repeat mean scores.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
        metric (str): Column name in per_repeat to plot.
        metric_label (str): Y-axis label for the metric.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    pipelines = [p for p in pipeline_order if p in per_repeat["pipeline"].unique()]

    for p in pipelines:
        subset = per_repeat.loc[per_repeat["pipeline"] == p].sort_values("repeat")
        scores = subset[metric].values
        cumulative_mean = np.cumsum(scores) / np.arange(1, len(scores) + 1)
        ax.plot(
            range(1, len(scores) + 1),
            cumulative_mean,
            color=get_pipeline_color(p),
            label=PIPELINE_LABELS.get(p, p),
            linewidth=1.5,
        )

    ax.set_xlabel("Number of repeats included")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", frameon=True, edgecolor="grey")

    fig.tight_layout()
    out_path = fig_dir / "03_repeat_convergence.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_feature_importance(all_results, fig_dir, log, pipeline_order=PIPELINE_NAMES):
    """Horizontal bar chart of the most frequently selected features.

    Counts how often each feature is selected across all outer folds
    and repeats, separately for each pipeline. Shows the top 20
    features ranked by total selection count.

    Args:
        all_results (pd.DataFrame): Fold-level results with
            selected_features column.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
    """
    if "selected_features" not in all_results.columns:
        log.warning("No selected_features column; skipping feature importance plot.")
        return

    apply_plot_style()

    # Count feature frequencies per pipeline. Skip ensemble pipelines with no features.
    pipelines = [
        p
        for p in pipeline_order
        if p in all_results["pipeline"].unique()
        and all_results.loc[all_results["pipeline"] == p, "selected_features"]
        .notna()
        .any()
    ]
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
            y_pos + i * bar_height,
            counts,
            bar_height,
            color=get_pipeline_color(p),
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


def plot_confusion_matrices(all_results, fig_dir, log, pipeline_order=PIPELINE_NAMES):
    """Grid of normalised confusion matrices (one per pipeline).

    Aggregates y_true/y_pred across all outer folds and repeats for
    each pipeline and plots row-normalised confusion matrices showing
    per-class recall.

    Args:
        all_results (pd.DataFrame): Fold-level results with y_true
            and y_pred columns (comma-separated class labels).
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.
    """
    if "y_true" not in all_results.columns or "y_pred" not in all_results.columns:
        log.warning("No y_true/y_pred columns; skipping confusion matrix plot.")
        return

    apply_plot_style(scale="compact")

    pipelines = [p for p in pipeline_order if p in all_results["pipeline"].unique()]
    n_pipes = len(pipelines)
    if n_pipes == 0:
        return

    # Determine subplot grid: 2 columns for <=4 pipelines, 4 columns for more.
    ncols = 4 if n_pipes > 4 else 2
    nrows = (n_pipes + ncols - 1) // ncols
    fig_width = 3 * ncols
    fig_height = 2.5 * nrows + 0.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
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
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        annotate_heatmap(
            ax, cm_norm, fmt=".2f", raw_counts=cm, threshold=0.6, fontsize=7
        )

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=7)
        ax.set_yticklabels(class_labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        # Pipeline label as subplot annotation (not matplotlib title).
        ax.text(
            0.5,
            1.05,
            PIPELINE_LABELS.get(p, p),
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            fontweight="bold",
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


def compute_error_agreement(all_results, log, pipeline_order=PIPELINE_NAMES):
    """Compute pairwise error agreement between pipelines.

    For each pair of pipelines, measures how often they make the same
    errors on the same samples within shared (repeat, fold) splits.
    This quantifies error correlation and indicates whether ensembling
    the pipelines could improve accuracy.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, y_true, y_pred.
        log (logging.Logger): Logger instance.
        pipeline_order (tuple): Canonical pipeline ordering.

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

    pipelines = [p for p in pipeline_order if p in all_results["pipeline"].unique()]
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
            pipe_wrong_count[i],
            error_rate * 100,
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

    # Scale figure to number of pipelines so cells stay large enough.
    n = len(pipelines)
    cell_size = 0.7
    fig_w = 2 * (n * cell_size + 2.5)
    fig_h = n * cell_size + 2.0
    tick_fs = max(5, 9 - n // 3)
    annot_fs = max(5, 8 - n // 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # Left panel: Jaccard index.
    im1 = ax1.imshow(agreement_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    annotate_heatmap(ax1, agreement_matrix, fontsize=annot_fs)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, fontsize=tick_fs, rotation=45, ha="right")
    ax1.set_yticklabels(labels, fontsize=tick_fs)
    ax1.text(
        0.5,
        1.06,
        "Error overlap (Jaccard index)",
        transform=ax1.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right panel: conditional error probability.
    im2 = ax2.imshow(conditional_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    annotate_heatmap(ax2, conditional_matrix, fontsize=annot_fs)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, fontsize=tick_fs, rotation=45, ha="right")
    ax2.set_yticklabels(labels, fontsize=tick_fs)
    ax2.set_xlabel("Pipeline (wrong?)", fontsize=tick_fs)
    ax2.set_ylabel("Pipeline (given wrong)", fontsize=tick_fs)
    ax2.text(
        0.5,
        1.06,
        "P(column wrong | row wrong)",
        transform=ax2.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = fig_dir / "06_error_agreement.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


"""Hard Sample Diagnostic Analysis"""


def _load_stage2_features(run_dir, log):
    """Load preprocessed feature matrix and clinical labels for hard sample analysis.

    Reads the merged training data and clinical labels, builds a feature matrix
    in (samples x regions) layout, and aligns subtype labels to sample order.

    Args:
        run_dir (Path): Run directory containing preprocessing/data/.
        log (logging.Logger): Logger instance.

    Returns:
        tuple or None: (X, sample_cols, feature_names, sample_labels) where
            X is the feature matrix (n_samples, n_features), sample_cols is
            the list of sample column names, feature_names is a list of
            region identifiers, and sample_labels is an array of subtype
            labels aligned to sample_cols. Returns None if files are missing.
    """
    merged_path = run_dir / "preprocessing" / "data" / "train_merged.tsv"
    clinical_path = CODE_DIR.parent / "data" / "Train_clinical.tsv"

    if not merged_path.exists():
        log.warning(
            "Preprocessed data not found at %s; skipping hard sample analysis.",
            merged_path,
        )
        return None
    if not clinical_path.exists():
        log.warning(
            "Clinical data not found at %s; skipping hard sample analysis.",
            clinical_path,
        )
        return None

    train_df = pd.read_csv(merged_path, sep="\t")
    clinical_df = pd.read_csv(clinical_path, sep="\t")

    # Build feature matrix (samples x regions), same order as the nested CV runner.
    genomic_cols = {"Chromosome", "Start", "End", "Nclone"}
    sample_cols = [c for c in train_df.columns if c not in genomic_cols]
    feature_names = [
        f"chr{row['Chromosome']}_{row['Start']}_{row['End']}"
        for _, row in train_df[["Chromosome", "Start", "End"]].iterrows()
    ]
    X = train_df[sample_cols].values.T  # (n_samples, n_features).

    # Align labels to the sample column order.
    clinical_df = clinical_df.set_index("Sample")
    sample_labels = clinical_df.loc[sample_cols, "Subgroup"].values

    return X, sample_cols, feature_names, sample_labels


def _compute_sample_error_rates(all_results, sample_cols, sample_labels):
    """Compute per-sample error rates across all fold-pipeline evaluations.

    Iterates over fold results, counting how many times each sample appeared
    in a test fold and how many times it was misclassified.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            test_indices, y_true, y_pred.
        sample_cols (list): Sample column names (defines index mapping).
        sample_labels (np.ndarray): Subtype labels aligned to sample_cols.

    Returns:
        pd.DataFrame: Summary with columns (sample, true_label, n_tested,
            n_errors, error_rate).
    """
    n_samples = len(sample_cols)
    n_tested = np.zeros(n_samples, dtype=int)
    n_errors = np.zeros(n_samples, dtype=int)

    for _, row in all_results.iterrows():
        if (
            pd.isna(row.get("test_indices"))
            or pd.isna(row.get("y_true"))
            or pd.isna(row.get("y_pred"))
        ):
            continue

        indices = [int(x) for x in str(row["test_indices"]).split(",")]
        y_true = [s.strip() for s in str(row["y_true"]).split(",")]
        y_pred = [s.strip() for s in str(row["y_pred"]).split(",")]

        if len(indices) != len(y_true) or len(indices) != len(y_pred):
            continue

        for idx, yt, yp in zip(indices, y_true, y_pred):
            n_tested[idx] += 1
            if yt != yp:
                n_errors[idx] += 1

    error_rate = np.where(n_tested > 0, n_errors / n_tested, 0.0)

    return pd.DataFrame(
        {
            "sample": sample_cols,
            "true_label": sample_labels,
            "n_tested": n_tested,
            "n_errors": n_errors,
            "error_rate": error_rate,
        }
    )


def _kw_hard_vs_easy(X, sample_cols, class_df, feature_names, label, log):
    """Run Kruskal-Wallis tests comparing hard vs easy samples within one class.

    Splits samples into hard (above-median error rate) and easy (at-or-below
    median), then tests each genomic feature for differences between groups.
    Applies Bonferroni and Benjamini-Hochberg FDR corrections.

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        sample_cols (list): Sample column names (defines index mapping).
        class_df (pd.DataFrame): Subset of sample_summary for one class,
            with columns (sample, true_label, n_tested, n_errors, error_rate).
        feature_names (list): Region identifiers for logging.
        label (str): Class label (e.g. "HR+" or "Triple Neg").
        log (logging.Logger): Logger instance.

    Returns:
        dict or None: Summary dict with keys n_hard, n_easy, threshold,
            n_sig_uncorrected, n_sig_bonferroni, n_sig_bh_005, n_sig_bh_010,
            n_expected_by_chance, top_features. Returns None if group sizes
            are too small for testing.
    """
    median_er = class_df["error_rate"].median()

    # If median is 0, threshold becomes 0 and "hard" means any error at all.
    threshold = median_er if median_er > 0 else 0.0

    hard_mask = class_df["error_rate"] > threshold
    easy_mask = ~hard_mask
    n_hard = hard_mask.sum()
    n_easy = easy_mask.sum()

    log.info(
        "  %s: median error rate=%.3f, hard (>%.3f)=%d, easy (<=%.3f)=%d",
        label,
        median_er,
        threshold,
        n_hard,
        threshold,
        n_easy,
    )

    if n_hard < 2 or n_easy < 2:
        log.info("    Insufficient group sizes for KW test; skipping.")
        return None

    # Map sample names back to row indices in X.
    hard_names = class_df.loc[hard_mask, "sample"].tolist()
    easy_names = class_df.loc[easy_mask, "sample"].tolist()
    hard_idx = [sample_cols.index(s) for s in hard_names]
    easy_idx = [sample_cols.index(s) for s in easy_names]

    X_hard = X[hard_idx]
    X_easy = X[easy_idx]

    # KW test per feature: does this feature differ between hard and easy samples?
    n_features = X.shape[1]
    kw_h = np.zeros(n_features)
    kw_p = np.ones(n_features)

    for f in range(n_features):
        vals_hard = X_hard[:, f]
        vals_easy = X_easy[:, f]
        # Skip constant features.
        if np.std(np.concatenate([vals_hard, vals_easy])) == 0:
            continue
        try:
            stat_val, p_val = stats.kruskal(vals_hard, vals_easy)
            kw_h[f] = stat_val
            kw_p[f] = p_val
        except ValueError:
            continue

    # Multiple testing correction: Bonferroni and Benjamini-Hochberg FDR.
    kw_p_bonf = np.minimum(kw_p * n_features, 1.0)
    reject_bh, kw_p_bh, _, _ = multipletests(kw_p, alpha=0.05, method="fdr_bh")

    n_sig_raw = int((kw_p < 0.05).sum())
    n_sig_bonf = int((kw_p_bonf < 0.05).sum())
    n_sig_bh05 = int(reject_bh.sum())
    reject_bh10, _, _, _ = multipletests(kw_p, alpha=0.10, method="fdr_bh")
    n_sig_bh10 = int(reject_bh10.sum())

    n_expected_by_chance = n_features * 0.05

    log.info("    KW test (hard vs easy within %s):", label)
    log.info(
        "      Features with uncorrected p < 0.05: %d / %d (%.1f expected by chance)",
        n_sig_raw,
        n_features,
        n_expected_by_chance,
    )
    log.info(
        "      Features with Bonferroni p < 0.05: %d / %d",
        n_sig_bonf,
        n_features,
    )
    log.info(
        "      Features with BH-FDR q < 0.05: %d / %d",
        n_sig_bh05,
        n_features,
    )
    log.info(
        "      Features with BH-FDR q < 0.10: %d / %d",
        n_sig_bh10,
        n_features,
    )

    # Report top features by raw p-value.
    top_idx = np.argsort(kw_p)[:10]
    log.info("      Top 10 features by raw p-value:")
    for ti in top_idx:
        log.info(
            "        %s: H=%.2f, p_raw=%.4e, p_bonf=%.4e, q_BH=%.4e",
            feature_names[ti],
            kw_h[ti],
            kw_p[ti],
            kw_p_bonf[ti],
            kw_p_bh[ti],
        )

    return {
        "n_hard": n_hard,
        "n_easy": n_easy,
        "threshold": float(threshold),
        "n_sig_uncorrected": n_sig_raw,
        "n_sig_bonferroni": n_sig_bonf,
        "n_sig_bh_005": n_sig_bh05,
        "n_sig_bh_010": n_sig_bh10,
        "n_expected_by_chance": round(n_expected_by_chance, 1),
        "top_features": [
            (feature_names[ti], float(kw_h[ti]), float(kw_p[ti]), float(kw_p_bh[ti]))
            for ti in top_idx[:5]
        ],
    }


def _log_hard_sample_conclusions(kw_results_by_class, log):
    """Log interpretive conclusions from the hard-vs-easy KW analysis.

    For each class, determines whether features significantly distinguish
    hard from easy samples and logs the appropriate conclusion about
    classifier ceiling performance.

    Args:
        kw_results_by_class (dict): Mapping from class label to KW result
            dict (or None if skipped).
        log (logging.Logger): Logger instance.
    """
    for label, res in kw_results_by_class.items():
        if res is None:
            continue

        n_uncorr = res["n_sig_uncorrected"]
        n_expected = res["n_expected_by_chance"]
        n_bh05 = res["n_sig_bh_005"]
        n_bh10 = res["n_sig_bh_010"]

        if n_bh05 > 0:
            log.info(
                "CONCLUSION (%s): %d feature(s) significant at BH-FDR q < 0.05. "
                "There is a reproducible biological axis distinguishing hard from "
                "easy samples.",
                label,
                n_bh05,
            )
        elif n_bh10 > 0:
            log.info(
                "CONCLUSION (%s): No features at BH-FDR q < 0.05, but %d at q < 0.10 "
                "(%d uncorrected vs %.1f expected by chance). A weak signal exists "
                "but is not strong enough to exploit with this sample size. "
                "Performance is effectively at ceiling.",
                label,
                n_bh10,
                n_uncorr,
                n_expected,
            )
        elif n_uncorr > n_expected * 2:
            log.info(
                "CONCLUSION (%s): No features survive multiple testing correction, "
                "but %d uncorrected hits vs %.1f expected suggests a diffuse signal "
                "too weak to resolve. Performance is at ceiling for this "
                "feature representation.",
                label,
                n_uncorr,
                n_expected,
            )
        else:
            log.info(
                "CONCLUSION (%s): %d uncorrected hits vs %.1f expected by chance - "
                "pure noise. Hard and easy samples are indistinguishable. "
                "Performance is at ceiling.",
                label,
                n_uncorr,
                n_expected,
            )


def analyse_hard_samples(all_results, run_dir, log, fig_dir, data_dir):
    """Diagnostic: are consistently misclassified samples distinguishable in feature space?

    For each of the 100 training samples, computes the error rate across all
    appearances in test folds (pooled across all pipelines and repeats).
    Within the HR+ and Triple Neg classes (Stage 2 scope), splits samples
    into "hard" (above-median error rate) vs "easy" (at-or-below-median)
    and runs a Kruskal-Wallis test per genomic feature.

    If no features discriminate hard vs easy, the misclassified samples are
    genuinely ambiguous and classifier performance is at ceiling. If features
    do discriminate, there is a biological axis the current features are not
    fully capturing.

    This is a diagnostic question, not a training signal. The hard/easy split
    is never used for model fitting or feature selection.

    Args:
        all_results (pd.DataFrame): Fold-level results with columns
            pipeline, repeat, outer_fold, test_indices, y_true, y_pred.
        run_dir (Path): Run directory containing preprocessing/data/.
        log (logging.Logger): Logger instance.
        fig_dir (Path): Directory to save figures.
        data_dir (Path): Directory to save data outputs.
    """
    # Check required columns.
    required = ["test_indices", "y_true", "y_pred"]
    missing_cols = [c for c in required if c not in all_results.columns]
    if missing_cols:
        log.warning(
            "Missing columns for hard sample analysis: %s; skipping.",
            missing_cols,
        )
        return

    # Load the preprocessed feature matrix and clinical labels.
    result = _load_stage2_features(run_dir, log)
    if result is None:
        return
    X, sample_cols, feature_names, sample_labels = result

    # Compute per-sample error rates across all fold-pipeline evaluations.
    sample_summary = _compute_sample_error_rates(
        all_results,
        sample_cols,
        sample_labels,
    )

    # Log overall per-class statistics.
    log.info("=== Hard Sample Diagnostic Analysis ===")
    log.info(
        "Per-sample error rates pooled across %d fold-pipeline evaluations.",
        len(all_results),
    )
    for label in ["HER2+", "HR+", "Triple Neg"]:
        mask = sample_summary["true_label"] == label
        subset = sample_summary[mask]
        mean_er = subset["error_rate"].mean()
        n_never = (subset["error_rate"] == 0).sum()
        n_any_error = (subset["error_rate"] > 0).sum()
        log.info(
            "  %s (n=%d): mean error rate=%.3f, never wrong=%d, wrong at least once=%d",
            label,
            mask.sum(),
            mean_er,
            n_never,
            n_any_error,
        )

    # Focus on Stage 2 classes since Stage 1 (HER2+) is perfect.
    s2_summary = sample_summary[
        sample_summary["true_label"].isin(["HR+", "Triple Neg"])
    ].copy()
    log.info("Stage 2 samples (HR+ and Triple Neg): %d total.", len(s2_summary))

    # KW analysis: hard vs easy within each class.
    kw_results_by_class = {}
    for label in ["HR+", "Triple Neg"]:
        class_df = s2_summary[s2_summary["true_label"] == label].copy()
        kw_results_by_class[label] = _kw_hard_vs_easy(
            X,
            sample_cols,
            class_df,
            feature_names,
            label,
            log,
        )

    # Log every sample that has any error, sorted by error rate.
    hard_samples = s2_summary[s2_summary["error_rate"] > 0].sort_values(
        "error_rate",
        ascending=False,
    )
    log.info("All Stage 2 samples with error_rate > 0 (%d samples):", len(hard_samples))
    for _, row in hard_samples.iterrows():
        log.info(
            "  %s (%s): error_rate=%.3f (%d / %d evaluations wrong)",
            row["sample"],
            row["true_label"],
            row["error_rate"],
            row["n_errors"],
            row["n_tested"],
        )

    # Save per-sample summary.
    summary_path = data_dir / "hard_sample_summary.csv"
    sample_summary.to_csv(summary_path, index=False)
    log.info("Saved per-sample error summary: %s", summary_path)

    # Generate figure.
    plot_hard_sample_analysis(s2_summary, kw_results_by_class, fig_dir, log)

    # Summary conclusions.
    _log_hard_sample_conclusions(kw_results_by_class, log)


def plot_hard_sample_analysis(s2_summary, kw_results_by_class, fig_dir, log):
    """Per-sample error rate strip plot for Stage 2 classes.

    Shows each HR+ and TN sample as a point, positioned by its error rate
    across all nested CV evaluations. Hard/easy threshold is marked.

    Args:
        s2_summary (pd.DataFrame): Stage 2 sample summary with columns
            sample, true_label, error_rate.
        kw_results_by_class (dict): KW test results per class.
        fig_dir (Path): Directory to save the figure.
        log (logging.Logger): Logger instance.
    """
    from utils.constants import SUBTYPE_COLORS

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    for ax, label in zip(axes, ["HR+", "Triple Neg"]):
        class_df = s2_summary[s2_summary["true_label"] == label].sort_values(
            "error_rate",
            ascending=False,
        )
        n = len(class_df)
        color = SUBTYPE_COLORS.get(label, "#888888")

        ax.barh(
            range(n),
            class_df["error_rate"].values,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_df["sample"].values, fontsize=5)
        ax.set_xlabel("Error rate (fraction of evaluations misclassified)")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        # Annotate with KW result summary.
        res = kw_results_by_class.get(label)
        if res is not None:
            annotation = (
                f"Hard: {res['n_hard']}, Easy: {res['n_easy']}\n"
                f"Uncorr. p<0.05: {res['n_sig_uncorrected']} "
                f"(exp. {res['n_expected_by_chance']})\n"
                f"BH-FDR q<0.10: {res['n_sig_bh_010']}"
            )
            ax.text(
                0.95,
                0.95,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        # Mark the threshold.
        if res is not None and res["threshold"] > 0:
            threshold_y_count = (class_df["error_rate"] > res["threshold"]).sum()
            ax.axhline(
                y=threshold_y_count - 0.5,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

        # Pipeline label as subplot annotation.
        ax.text(
            0.5,
            1.04,
            label,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path = fig_dir / "07_hard_sample_analysis.png"
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
            "configs/<name>.yaml. (default: local)."
        ),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="nested_cv_2x2",
        help=(
            "Phase directory name inside the run directory that contains "
            "the fold result CSVs. Use 'hierarchical_nested_cv' for "
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

    # Determine pipeline ordering based on phase and data content.
    pipeline_order = get_pipeline_order(args.phase, all_results)
    log.info("Pipeline ordering: %s", list(pipeline_order))

    # Save aggregated results.
    all_results_path = data_dir / "all_fold_results.csv"
    all_results.to_csv(all_results_path, index=False)
    log.info("Saved aggregated fold results: %s", all_results_path)

    """Step 2: Compute Per-Repeat Means and Detect Metric"""

    per_repeat = compute_per_repeat_means(all_results)
    log.info("Computed per-repeat means for %d pipeline-repeat pairs.", len(per_repeat))

    # Detect hierarchical mode. Since Stage 1 is perfect (BA=1.0), combined
    # 3-class BA is inflated and masks real differences. Use BA2 (Stage 2
    # only, HR+ vs TN) as the primary comparison metric instead.
    is_hierarchical = "stage2_bal_acc" in all_results.columns
    if is_hierarchical:
        metric = "mean_stage2_bal_acc"
        fold_metric = "stage2_bal_acc"
        metric_label = "Stage 2 balanced accuracy (HR+ vs Triple Negative)"
        log.info("Hierarchical mode detected: using BA2 (Stage 2) as primary metric.")
    else:
        metric = "mean_balanced_accuracy"
        fold_metric = "balanced_accuracy"
        metric_label = "Mean balanced accuracy (across 5 outer folds)"
        log.info("Flat mode: using combined balanced accuracy as primary metric.")

    """Step 3: Summary Statistics"""

    summary = compute_summary_statistics(per_repeat, metric=metric)
    summary_path = data_dir / "summary_statistics.csv"
    summary.to_csv(summary_path)
    log.info("Summary statistics (primary metric: %s):", fold_metric)
    for pipeline in summary.index:
        row = summary.loc[pipeline]
        log.info(
            "  %s: mean=%.4f +/- %.4f, median=%.4f, n_repeats=%d",
            PIPELINE_LABELS.get(pipeline, pipeline),
            row["mean_bal_acc"],
            row["std_bal_acc"],
            row["median_bal_acc"],
            int(row["n_repeats"]),
        )

    # For hierarchical runs, also report combined 3-class BA as secondary diagnostic.
    if is_hierarchical:
        summary_combined = compute_summary_statistics(
            per_repeat,
            metric="mean_balanced_accuracy",
        )
        summary_combined_path = data_dir / "summary_statistics_combined_ba.csv"
        summary_combined.to_csv(summary_combined_path)
        log.info(
            "Combined 3-class BA (secondary diagnostic, inflated by perfect Stage 1):"
        )
        for pipeline in summary.index:
            if pipeline in summary_combined.index:
                row_comb = summary_combined.loc[pipeline]
                row_ba2 = summary.loc[pipeline]
                log.info(
                    "  %s: BA2=%.4f, combined_BA=%.4f",
                    PIPELINE_LABELS.get(pipeline, pipeline),
                    row_ba2["mean_bal_acc"],
                    row_comb["mean_bal_acc"],
                )
        log.info("Saved combined BA summary: %s", summary_combined_path)

    log.info("Saved primary summary statistics: %s", summary_path)

    """Step 3b: Sensitivity Analysis Summary (mislabel exclusion)"""

    # Use whichever metric matches the primary metric for the sensitivity analysis.
    excl_col_name = (
        "stage2_bal_acc_excl" if is_hierarchical else "combined_bal_acc_excl"
    )
    if excl_col_name in all_results.columns:
        excl_col = all_results[excl_col_name].dropna()
        if len(excl_col) > 0:
            log.info(
                "=== Mislabel Exclusion Sensitivity Analysis (metric: %s) ===",
                excl_col_name,
            )
            excl_repeat = all_results.dropna(subset=[excl_col_name]).copy()
            excl_grouped = (
                excl_repeat.groupby("pipeline")
                .agg(
                    mean_ba_excl=(excl_col_name, "mean"),
                    std_ba_excl=(excl_col_name, "std"),
                )
                .sort_values("mean_ba_excl", ascending=False)
            )

            for pipeline in excl_grouped.index:
                row = excl_grouped.loc[pipeline]
                incl_mean = (
                    summary.loc[pipeline, "mean_bal_acc"]
                    if pipeline in summary.index
                    else float("nan")
                )
                delta = row["mean_ba_excl"] - incl_mean
                log.info(
                    "  %s: excl=%.4f +/- %.4f (incl=%.4f, delta=%+.4f)",
                    PIPELINE_LABELS.get(pipeline, pipeline),
                    row["mean_ba_excl"],
                    row["std_ba_excl"],
                    incl_mean,
                    delta,
                )

            excl_path = data_dir / "sensitivity_mislabel_exclusion.csv"
            excl_grouped.to_csv(excl_path)
            log.info("Saved mislabel exclusion summary: %s", excl_path)

    """Step 4: Data Completeness Check"""

    # Warn about incomplete runs.
    expected_pipelines = set(config.get("pipelines", {}).get("names", PIPELINE_NAMES))
    actual_pipelines = set(all_results["pipeline"].unique())
    missing_pipelines = expected_pipelines - actual_pipelines
    if missing_pipelines:
        log.warning(
            "Missing pipelines (incomplete run?): %s",
            sorted(missing_pipelines),
        )

    expected_repeats = config.get("cv", {}).get("n_repeats", None)
    if expected_repeats is not None:
        for p in actual_pipelines:
            actual = all_results.loc[all_results["pipeline"] == p, "repeat"].nunique()
            if actual < expected_repeats:
                log.warning(
                    "Pipeline %s has %d/%d repeats.",
                    p,
                    actual,
                    expected_repeats,
                )

    """Step 5: Statistical Testing (Friedman omnibus)"""

    friedman_stat, friedman_p, pairwise_df_legacy = run_statistical_tests(
        per_repeat,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
    )

    """Step 5a: Pre-registered Primary Comparisons (uncorrected Wilcoxon)"""

    prereg_df = run_preregistered_tests(
        per_repeat,
        summary,
        log,
        metric=metric,
    )
    if prereg_df is not None:
        prereg_path = data_dir / "preregistered_comparisons.csv"
        prereg_df.to_csv(prereg_path, index=False)
        log.info("Saved pre-registered comparisons: %s", prereg_path)

    """Step 5b: Grouped Classifier Test (NMC vs RF, pre-registered)"""

    grouped_result = run_grouped_classifier_test(per_repeat, log, metric=metric)
    if grouped_result is not None:
        grouped_path = data_dir / "grouped_nmc_vs_rf.csv"
        pd.DataFrame([grouped_result]).to_csv(grouped_path, index=False)
        log.info("Saved grouped NMC vs RF test: %s", grouped_path)

    """Step 5c: Exploratory Comparisons (Bonferroni-corrected Wilcoxon)"""

    # Build set of pre-registered pairs to exclude from exploratory tests.
    prereg_pair_set = set()
    if prereg_df is not None:
        for _, row in prereg_df.iterrows():
            prereg_pair_set.add(tuple(sorted((row["pipeline_a"], row["pipeline_b"]))))

    exploratory_df = run_exploratory_tests(
        per_repeat,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
        exclude_pairs=prereg_pair_set,
    )
    if exploratory_df is not None:
        exploratory_path = data_dir / "exploratory_comparisons.csv"
        exploratory_df.to_csv(exploratory_path, index=False)
        log.info("Saved exploratory comparisons: %s", exploratory_path)

    # Combine pre-registered and exploratory results for the violin plot.
    pairwise_parts = [df for df in [prereg_df, exploratory_df] if df is not None]
    pairwise_df = (
        pd.concat(pairwise_parts, ignore_index=True) if pairwise_parts else None
    )

    # Save combined pairwise results.
    if pairwise_df is not None:
        pairwise_path = data_dir / "pairwise_wilcoxon.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        log.info("Saved combined pairwise Wilcoxon tests: %s", pairwise_path)

    """Step 6: Statistical Testing (Nadeau-Bengio Corrected t-test)"""

    nb_df = run_nadeau_bengio_tests(
        all_results,
        config,
        log,
        pipeline_order=pipeline_order,
        fold_metric=fold_metric,
    )
    if nb_df is not None:
        nb_path = data_dir / "pairwise_nadeau_bengio.csv"
        nb_df.to_csv(nb_path, index=False)
        log.info("Saved Nadeau-Bengio tests: %s", nb_path)

    """Step 6b: Paired Bootstrap Confidence Intervals"""

    boot_df = run_bootstrap_ci_tests(
        per_repeat,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
    )
    if boot_df is not None:
        boot_path = data_dir / "pairwise_bootstrap_ci.csv"
        boot_df.to_csv(boot_path, index=False)
        n_dist = boot_df["distinguishable"].sum()
        log.info(
            "Saved bootstrap CIs: %s (%d/%d pairs distinguishable)",
            boot_path,
            n_dist,
            len(boot_df),
        )

    """Step 7: Identify Winner"""

    winner = summary.index[0]
    winner_label = "BA2" if is_hierarchical else "BA"
    log.info(
        "Winning pipeline: %s (mean %s = %.4f)",
        PIPELINE_LABELS.get(winner, winner),
        winner_label,
        summary.loc[winner, "mean_bal_acc"],
    )

    """Step 8: Error Agreement Analysis"""

    agreement_matrix, conditional_matrix, agree_pipelines = compute_error_agreement(
        all_results,
        log,
        pipeline_order=pipeline_order,
    )

    # Save error agreement data.
    if agreement_matrix is not None:
        agree_labels = [PIPELINE_LABELS.get(p, p) for p in agree_pipelines]
        pd.DataFrame(
            agreement_matrix,
            index=agree_labels,
            columns=agree_labels,
        ).to_csv(data_dir / "error_agreement_jaccard.csv")
        pd.DataFrame(
            conditional_matrix,
            index=agree_labels,
            columns=agree_labels,
        ).to_csv(data_dir / "error_agreement_conditional.csv")
        log.info("Saved error agreement matrices.")

    """Step 9: Generate Figures"""

    plot_pipeline_comparison(
        per_repeat,
        pairwise_df,
        fig_dir,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
        metric_label=metric_label,
    )
    plot_interaction(
        per_repeat,
        fig_dir,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
        metric_label=metric_label,
    )
    plot_repeat_convergence(
        per_repeat,
        fig_dir,
        log,
        pipeline_order=pipeline_order,
        metric=metric,
        metric_label=f"Cumulative mean {fold_metric.replace('_', ' ')}",
    )
    plot_feature_importance(
        all_results,
        fig_dir,
        log,
        pipeline_order=pipeline_order,
    )
    plot_confusion_matrices(
        all_results,
        fig_dir,
        log,
        pipeline_order=pipeline_order,
    )
    plot_error_agreement(
        agreement_matrix, conditional_matrix, agree_pipelines, fig_dir, log
    )

    """Step 9b: Hard Sample Diagnostic Analysis"""

    analyse_hard_samples(all_results, run_dir, log, fig_dir, data_dir)

    """Step 10: Save Config Snapshot"""

    save_config(
        run_dir,
        "analyse_nested_cv",
        config_file=config["_config_path"],
        n_fold_files=len(list(nested_cv_data_dir.glob("fold_results_*.csv"))),
        n_total_folds=len(all_results),
        n_pipelines=all_results["pipeline"].nunique(),
        n_repeats=all_results["repeat"].nunique(),
        primary_metric=fold_metric,
        winning_pipeline=winner,
        winning_mean_bal_acc=float(summary.loc[winner, "mean_bal_acc"]),
        friedman_p=float(friedman_p) if friedman_p is not None else None,
    )

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
