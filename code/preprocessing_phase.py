"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: preprocessing_phase.py.
Description:
    Phase 1 preprocessing for the CATS breast cancer subtype classification
    project. Performs label-free region merging of spatially correlated aCGH
    regions into consensus segments and generates handoff files for downstream
    feature selection (Phase 2).

Usage:
    python3 code/preprocessing_phase.py

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback
from rich.progress import track

from utils import (
    DATA_DIR,
    SUBTYPE_COLORS,
    SUBTYPE_ORDER,
    apply_plot_style,
    get_phase_dirs,
    get_sample_columns,
    kruskal_wallis_per_region,
    setup_logging,
)
from utils.statistics import bonferroni_threshold

rich.traceback.install()

# ---------------------------------------------------------------------------
# Phase-specific settings.
# ---------------------------------------------------------------------------
MERGE_THRESHOLD = 0.8

# Initialise output directories and logging.
FIG_DIR, OUT_DIR = get_phase_dirs("preprocessing_phase")
log, console = setup_logging("preprocessing_phase")
apply_plot_style()


"""Data Loading."""


def load_data():
    """Load training CN calls, validation CN calls, and clinical labels.

    Returns:
        tuple: (train_df, val_df, clinical_df).
    """
    train_df = pd.read_csv(DATA_DIR / "Train_call.tsv", sep="\t")
    val_df = pd.read_csv(DATA_DIR / "Validation_call.tsv", sep="\t")
    clinical_df = pd.read_csv(DATA_DIR / "Train_clinical.tsv", sep="\t")
    return train_df, val_df, clinical_df


"""Region Merging."""


def merge_regions(cn_df, sample_cols, threshold=MERGE_THRESHOLD):
    """Merge adjacent correlated regions into consensus segments via greedy one-pass walk.

    For each chromosome, walks adjacent regions in genomic order. Extends a
    chain as long as each new neighbour correlates with the previous region
    at Pearson r > threshold. When a pair fails, the chain is finalised as
    one consensus segment and a new chain starts. No gap bridging.

    Args:
        cn_df (pd.DataFrame): Copy-number data sorted by Chromosome and Start.
        sample_cols (list[str]): Sample column names.
        threshold (float): Pearson r threshold for merging adjacent regions.

    Returns:
        dict: Merge map - keys are consensus segment indices (str for JSON
            compatibility), values are lists of raw region row indices (0-based).
    """
    data = cn_df[sample_cols].values.astype(float)
    chroms = cn_df["Chromosome"].values
    chrom_order = sorted(cn_df["Chromosome"].unique())

    all_chains = []

    for chrom in track(chrom_order, description="Merging regions"):
        chrom_idx = np.where(chroms == chrom)[0]

        if len(chrom_idx) == 0:
            continue

        current_chain = [int(chrom_idx[0])]

        for k in range(1, len(chrom_idx)):
            i_prev = chrom_idx[k - 1]
            i_curr = chrom_idx[k]

            # Correlate current region with its immediate predecessor.
            r = np.corrcoef(data[i_prev], data[i_curr])[0, 1]
            if np.isnan(r):
                r = 0.0

            if r > threshold:
                current_chain.append(int(i_curr))
            else:
                all_chains.append(current_chain)
                current_chain = [int(i_curr)]

        # Finalise last chain on this chromosome.
        all_chains.append(current_chain)

    # Build merge map with string keys for JSON serialisation.
    merge_map = {str(i): chain for i, chain in enumerate(all_chains)}

    return merge_map


def build_consensus_matrix(cn_df, merge_map, sample_cols):
    """Build consensus feature matrix from a merge map.

    Each consensus segment's CN value is the median of its constituent
    regions' CN values per sample. Median is appropriate for ordinal/discrete
    data (-1, 0, 1, 2). Genomic coordinates are derived from the min(Start)
    and max(End) of constituent regions.

    Args:
        cn_df (pd.DataFrame): Original copy-number data.
        merge_map (dict): Maps segment ID (str) to list of raw row indices.
        sample_cols (list[str]): Sample column names.

    Returns:
        pd.DataFrame: Consensus matrix with Chromosome, Start, End, Nclone,
            and sample columns, sorted by genomic position.
    """
    data = cn_df[sample_cols].values.astype(float)
    n_segments = len(merge_map)

    # Pre-allocate arrays for speed.
    chrom_arr = np.empty(n_segments, dtype=int)
    start_arr = np.empty(n_segments, dtype=int)
    end_arr = np.empty(n_segments, dtype=int)
    nclone_arr = np.empty(n_segments, dtype=int)
    cn_matrix = np.empty((n_segments, len(sample_cols)), dtype=float)

    for seg_id_str in track(sorted(merge_map.keys(), key=int),
                            description="Building consensus"):
        seg_id = int(seg_id_str)
        raw_indices = merge_map[seg_id_str]

        constituent = cn_df.iloc[raw_indices]
        chrom_arr[seg_id] = int(constituent["Chromosome"].iloc[0])
        start_arr[seg_id] = int(constituent["Start"].min())
        end_arr[seg_id] = int(constituent["End"].max())
        nclone_arr[seg_id] = int(constituent["Nclone"].sum())

        # Median CN across constituent regions, per sample.
        cn_matrix[seg_id] = np.median(data[raw_indices], axis=0)

    # Build DataFrame.
    consensus_df = pd.DataFrame(cn_matrix, columns=sample_cols)
    consensus_df.insert(0, "Chromosome", chrom_arr)
    consensus_df.insert(1, "Start", start_arr)
    consensus_df.insert(2, "End", end_arr)
    consensus_df.insert(3, "Nclone", nclone_arr)

    # Sort by genomic position (should already be in order from processing).
    consensus_df = consensus_df.sort_values(
        ["Chromosome", "Start"],
    ).reset_index(drop=True)

    return consensus_df


"""Merge-Specific Figures."""


def plot_merge_summary(train_df, merge_map, merged_df):
    """Plot regions per chromosome before/after and segment size distribution.

    Produces figures 01 (grouped bar) and 02 (histogram).

    Args:
        train_df (pd.DataFrame): Raw training data.
        merge_map (dict): Merge map from merge_regions().
        merged_df (pd.DataFrame): Merged consensus data.
    """
    log.info("=" * 60)
    log.info("MERGE SUMMARY")
    log.info("=" * 60)

    chrom_order = sorted(train_df["Chromosome"].unique())
    before_counts = train_df["Chromosome"].value_counts().reindex(
        chrom_order,
    ).values
    after_counts = merged_df["Chromosome"].value_counts().reindex(
        chrom_order,
    ).values

    log.info("Total regions before merging: %d", len(train_df))
    log.info("Total consensus segments after merging: %d", len(merged_df))
    log.info("Overall reduction: %.1f%%",
             (1 - len(merged_df) / len(train_df)) * 100)
    log.info("")
    log.info("Per-chromosome summary:")
    log.info("%5s %8s %8s %10s", "Chr", "Before", "After", "Reduction")
    log.info("-" * 35)
    for chrom, before, after in zip(chrom_order, before_counts, after_counts):
        label = "X" if chrom == 23 else str(chrom)
        ratio = 1 - after / before if before > 0 else 0
        log.info("%5s %8d %8d %9.1f%%", label, before, after, ratio * 100)

    '''Figure 01: Regions per chromosome before and after merging.'''
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(chrom_order))
    width = 0.35
    ax.bar(x - width / 2, before_counts, width, color="#3C5488",
           edgecolor="black", linewidth=0.3,
           label="Raw regions (before merging)")
    ax.bar(x + width / 2, after_counts, width, color="#00A087",
           edgecolor="black", linewidth=0.3,
           label="Consensus segments (after merging)")
    chrom_labels = ["X" if c == 23 else str(c) for c in chrom_order]
    ax.set_xticks(x)
    ax.set_xticklabels(chrom_labels, fontsize=7)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of genomic regions")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "01_regions_per_chromosome.png")
    plt.close(fig)
    log.info("")
    log.info("Saved: 01_regions_per_chromosome.png")

    '''Figure 02: Segment size distribution.'''
    segment_sizes = [len(chain) for chain in merge_map.values()]
    singletons = sum(1 for s in segment_sizes if s == 1)
    log.info("")
    log.info("Segment size statistics:")
    log.info("  Singletons (1 raw region): %d (%.1f%%)",
             singletons, singletons / len(segment_sizes) * 100)
    log.info("  Median segment size: %d raw regions",
             int(np.median(segment_sizes)))
    log.info("  Mean segment size: %.1f raw regions", np.mean(segment_sizes))
    log.info("  Max segment size: %d raw regions", max(segment_sizes))

    fig, ax = plt.subplots(figsize=(6, 4))
    max_size = max(segment_sizes)
    bins = np.arange(0.5, max_size + 1.5, 1)
    ax.hist(segment_sizes, bins=bins, color="#4DBBD5",
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Number of raw regions per consensus segment")
    ax.set_ylabel("Number of consensus segments")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.97, 0.95,
        f"n = {len(segment_sizes)} segments\n"
        f"Singletons: {singletons}"
        f" ({singletons / len(segment_sizes) * 100:.0f}%)\n"
        f"Median: {int(np.median(segment_sizes))} regions\n"
        f"Max: {max_size} regions",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.savefig(FIG_DIR / "02_segment_size_distribution.png")
    plt.close(fig)
    log.info("Saved: 02_segment_size_distribution.png")
    log.info("")


"""Handoff Files."""


def compute_kruskal_wallis(merged_df, clinical_df, sample_cols):
    """Compute Kruskal-Wallis H-test per merged segment.

    Args:
        merged_df (pd.DataFrame): Merged consensus data.
        clinical_df (pd.DataFrame): Clinical labels.
        sample_cols (list[str]): Sample column names.

    Returns:
        tuple: (pvals, h_stats) - arrays of p-values and H-statistics.
    """
    pvals, h_stats = kruskal_wallis_per_region(merged_df, clinical_df, sample_cols)

    n_tests = len(pvals)
    bonf_alpha = bonferroni_threshold(n_tests)
    n_sig = (pvals < bonf_alpha).sum()
    log.info("Kruskal-Wallis: %d / %d segments pass Bonferroni (alpha=%.2e)",
             n_sig, n_tests, bonf_alpha)

    return pvals, h_stats


def save_handoff_files(train_merged, val_merged, merge_map, pvals, h_stats,
                       train_df):
    """Save all handoff files for Phase 2 (feature selection).

    Args:
        train_merged (pd.DataFrame): Merged training data.
        val_merged (pd.DataFrame): Merged validation data.
        merge_map (dict): Merge map (segment ID -> raw indices).
        pvals (np.ndarray): KW p-values per segment.
        h_stats (np.ndarray): KW H-statistics per segment.
        train_df (pd.DataFrame): Raw training data (for per-chromosome summary).
    """
    log.info("=" * 60)
    log.info("SAVING HANDOFF FILES")
    log.info("=" * 60)

    # Merged data matrices.
    train_merged.to_csv(OUT_DIR / "train_merged.tsv", sep="\t", index=False)
    val_merged.to_csv(OUT_DIR / "validation_merged.tsv", sep="\t", index=False)
    log.info("Saved: train_merged.tsv (%d segments x %d samples)",
             len(train_merged), len(get_sample_columns(train_merged)))
    log.info("Saved: validation_merged.tsv (%d segments x %d samples)",
             len(val_merged), len(get_sample_columns(val_merged)))

    # Merge map (JSON).
    with open(OUT_DIR / "merge_map.json", "w") as f:
        json.dump(merge_map, f, indent=2)
    log.info("Saved: merge_map.json (%d consensus segments)", len(merge_map))

    # Segment sizes.
    segment_rows = []
    for seg_id_str in sorted(merge_map.keys(), key=int):
        seg_id = int(seg_id_str)
        raw_indices = merge_map[seg_id_str]
        seg_row = train_merged.iloc[seg_id]
        segment_rows.append({
            "SegmentID": seg_id,
            "Chromosome": int(seg_row["Chromosome"]),
            "Start": int(seg_row["Start"]),
            "End": int(seg_row["End"]),
            "n_raw_regions": len(raw_indices),
        })
    segment_df = pd.DataFrame(segment_rows)
    segment_df.to_csv(OUT_DIR / "segment_sizes.tsv", sep="\t", index=False)
    log.info("Saved: segment_sizes.tsv")

    # Kruskal-Wallis results (two sorted versions).
    n_tests = len(pvals)
    bonf_alpha = bonferroni_threshold(n_tests)
    n_raw_per_seg = [len(merge_map[str(i)]) for i in range(len(merge_map))]

    kw_df = pd.DataFrame({
        "SegmentID": range(len(pvals)),
        "Chromosome": train_merged["Chromosome"].values.astype(int),
        "Start": train_merged["Start"].values.astype(int),
        "End": train_merged["End"].values.astype(int),
        "n_raw_regions": n_raw_per_seg,
        "H_statistic": h_stats,
        "p_value": pvals,
        "bonferroni_significant": pvals < bonf_alpha,
    })

    kw_pos = kw_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)
    kw_pos.to_csv(OUT_DIR / "kruskal_wallis_by_genomic_position.tsv",
                  sep="\t", index=False)
    log.info("Saved: kruskal_wallis_by_genomic_position.tsv")

    kw_pval = kw_df.sort_values("p_value").reset_index(drop=True)
    kw_pval.to_csv(OUT_DIR / "kruskal_wallis_by_pvalue.tsv",
                   sep="\t", index=False)
    log.info("Saved: kruskal_wallis_by_pvalue.tsv")

    # Per-chromosome merge summary.
    chrom_order = sorted(train_df["Chromosome"].unique())
    summary_rows = []
    for chrom in chrom_order:
        before = int((train_df["Chromosome"] == chrom).sum())
        after = int((train_merged["Chromosome"] == chrom).sum())
        ratio = round(after / before, 4) if before > 0 else 0.0
        summary_rows.append({
            "Chromosome": chrom,
            "regions_before": before,
            "regions_after": after,
            "reduction_ratio": ratio,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "per_chromosome_merge_summary.tsv",
                      sep="\t", index=False)
    log.info("Saved: per_chromosome_merge_summary.tsv")
    log.info("")


"""Main."""


def main():
    """Run Phase 1: label-free region merging and handoff file generation."""
    train_df, val_df, clinical_df = load_data()
    train_sample_cols = get_sample_columns(train_df)
    val_sample_cols = get_sample_columns(val_df)

    log.info("=" * 60)
    log.info("PHASE 1: PREPROCESSING (LABEL-FREE REGION MERGING)")
    log.info("=" * 60)
    log.info("Training data: %d regions x %d samples",
             len(train_df), len(train_sample_cols))
    log.info("Validation data: %d regions x %d samples",
             len(val_df), len(val_sample_cols))
    log.info("Merge threshold: Pearson r > %.1f", MERGE_THRESHOLD)
    log.info("")

    # Perform greedy merging on training data (label-free).
    merge_map = merge_regions(train_df, train_sample_cols)
    log.info("Merge complete: %d raw regions -> %d consensus segments",
             len(train_df), len(merge_map))

    # Build consensus matrices for both training and validation data.
    train_merged = build_consensus_matrix(
        train_df, merge_map, train_sample_cols,
    )
    val_merged = build_consensus_matrix(val_df, merge_map, val_sample_cols)
    log.info("Training consensus: %d segments x %d samples",
             len(train_merged), len(train_sample_cols))
    log.info("Validation consensus: %d segments x %d samples",
             len(val_merged), len(val_sample_cols))
    log.info("")

    # Merge-specific figures.
    plot_merge_summary(train_df, merge_map, train_merged)

    # Kruskal-Wallis for handoff.
    pvals, h_stats = compute_kruskal_wallis(
        train_merged, clinical_df, train_sample_cols,
    )

    # Save all handoff files.
    save_handoff_files(
        train_merged, val_merged, merge_map, pvals, h_stats, train_df,
    )

    log.info("=" * 60)
    log.info("PHASE 1 PREPROCESSING COMPLETE.")
    log.info("  Figures: %s", FIG_DIR)
    log.info("  Data:    %s", OUT_DIR)
    log.info("  Log:     %s",
             FIG_DIR.parent.parent / "logs" / "preprocessing_phase.log")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
