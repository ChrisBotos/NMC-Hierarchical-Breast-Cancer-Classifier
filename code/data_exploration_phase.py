"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: data_exploration_phase.py.
Description:
    Phase 0 data exploration for the CATS breast cancer subtype
    classification project. Produces publication-quality figures and
    logs summary statistics to console and log file. Supports running
    on alternative input data (e.g. merged segments) via --input and
    --tag CLI arguments.

Usage:
    python3 code/data_exploration_phase.py
    python3 code/data_exploration_phase.py --input results/data/preprocessing_phase/train_merged.tsv --tag merged

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback
from matplotlib.patches import Patch
from rich.progress import track
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from utils import (
    CN_LABELS,
    DATA_DIR,
    PROJECT_DIR,
    SUBTYPE_COLORS,
    SUBTYPE_ORDER,
    apply_plot_style,
    bonferroni_threshold,
    get_phase_dirs,
    get_sample_columns,
    get_sample_matrix,
    kruskal_wallis_per_region,
    load_gene_map,
    setup_logging,
)

rich.traceback.install()

# Set by setup() before any analysis runs.
FIG_DIR = None
OUT_DIR = None
log = None


def setup(tag=""):
    """Initialise output directories and logging for a given run tag.

    When tag is empty, outputs go to the default locations
    (data_exploration_phase/). When tag is non-empty (e.g. "merged"),
    outputs go to data_exploration_phase_merged/ so that raw and merged
    runs do not overwrite each other.

    Args:
        tag (str): Optional suffix appended to directory and log names.
    """
    global FIG_DIR, OUT_DIR, log

    FIG_DIR, OUT_DIR = get_phase_dirs("data_exploration_phase", tag=tag)
    log, _console = setup_logging("data_exploration_phase", tag=tag)
    apply_plot_style()


"""Data Loading."""


def load_data(input_path=None):
    """Load training CN calls, clinical labels, and gene map.

    Args:
        input_path (str | Path | None): Optional path to an alternative CN
            data file (e.g. merged segments). Defaults to Train_call.tsv.

    Returns:
        tuple: (cn_df, clinical_df, gene_map_df).
    """
    if input_path is not None:
        cn_df = pd.read_csv(input_path, sep="\t")
    else:
        cn_df = pd.read_csv(DATA_DIR / "Train_call.tsv", sep="\t")
    clinical_df = pd.read_csv(DATA_DIR / "Train_clinical.tsv", sep="\t")
    gene_map_df = load_gene_map(DATA_DIR)
    return cn_df, clinical_df, gene_map_df


"""Section 1: Data Validation."""


def validate_data(cn_df, clinical_df):
    """Check data integrity: missing values, unexpected CN values, label consistency.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
    """
    sample_cols = get_sample_columns(cn_df)
    data = cn_df[sample_cols]

    log.info("=" * 60)
    log.info("SECTION 1: DATA VALIDATION")
    log.info("=" * 60)

    # Check for NaN values.
    nan_count = data.isna().sum().sum()
    log.info("Missing values (NaN): %d", nan_count)

    # Check that all values are in {-1, 0, 1, 2}.
    valid_values = {-1, 0, 1, 2}
    unique_values = set(data.values.flatten())
    invalid_values = unique_values - valid_values
    log.info("Unique CN values: %s", sorted(unique_values))
    if invalid_values:
        log.warning("  Invalid values found: %s", invalid_values)
    else:
        log.info("  All values are valid (-1, 0, 1, 2).")

    # Check sample overlap between CN and clinical data.
    cn_samples = set(sample_cols)
    clinical_samples = set(clinical_df["Sample"].values)
    log.info("")
    log.info("Samples in CN data: %d", len(cn_samples))
    log.info("Samples in clinical data: %d", len(clinical_samples))
    log.info("Overlap: %d", len(cn_samples & clinical_samples))
    if cn_samples != clinical_samples:
        log.info("  In CN only: %s", cn_samples - clinical_samples)
        log.info("  In clinical only: %s", clinical_samples - cn_samples)

    # Check genomic coordinate columns.
    log.info("")
    log.info("Genomic regions: %d", len(cn_df))
    log.info("Chromosomes: %s", sorted(cn_df["Chromosome"].unique()))
    log.info("  (23 = X chromosome)")

    # Value distribution across entire dataset.
    log.info("")
    log.info("Overall CN value distribution:")
    flat = data.values.flatten()
    for v in sorted(valid_values):
        count = (flat == v).sum()
        pct = count / len(flat) * 100
        log.info("  %2d (%13s): %8d  (%.1f%%)", v, CN_LABELS[v], count, pct)

    log.info("")


"""Section 2: Class Distribution."""


def plot_class_distribution(clinical_df):
    """Plot subtype class distribution as a bar chart.

    Args:
        clinical_df (pd.DataFrame): Clinical labels.
    """
    log.info("=" * 60)
    log.info("SECTION 2: CLASS DISTRIBUTION")
    log.info("=" * 60)

    counts = clinical_df["Subgroup"].value_counts().reindex(SUBTYPE_ORDER)
    log.info("%s", counts.to_string())
    log.info("")

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(
        SUBTYPE_ORDER,
        counts.values,
        color=[SUBTYPE_COLORS[s] for s in SUBTYPE_ORDER],
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )

    # Add count labels on bars.
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Number of samples (n)")
    ax.set_xlabel("Breast cancer molecular subtype")
    ax.set_ylim(0, counts.max() + 5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "01_class_distribution.png")
    plt.close(fig)
    log.info("Saved: 01_class_distribution.png")
    log.info("")


"""Section 3: Genome-wide Frequency Plots."""


def compute_genome_positions(cn_df):
    """Compute cumulative genome positions for plotting across chromosomes.

    Args:
        cn_df (pd.DataFrame): Copy-number data with Chromosome, Start, End columns.

    Returns:
        tuple: (genome_pos array, chrom_boundaries dict, chrom_centers dict).
    """
    # Build cumulative offsets per chromosome.
    chroms = cn_df["Chromosome"].unique()
    chrom_order = sorted(chroms, key=lambda x: int(x))
    offsets = {}
    cumulative = 0
    chrom_boundaries = {}
    chrom_centers = {}

    for chrom in chrom_order:
        mask = cn_df["Chromosome"] == chrom
        chrom_start = cn_df.loc[mask, "Start"].min()
        chrom_end = cn_df.loc[mask, "End"].max()
        chrom_len = chrom_end - chrom_start
        offsets[chrom] = cumulative - chrom_start
        chrom_boundaries[chrom] = cumulative
        chrom_centers[chrom] = cumulative + chrom_len / 2
        cumulative += chrom_len

    # Compute midpoint genome position for each region.
    midpoints = (cn_df["Start"] + cn_df["End"]) / 2
    genome_pos = midpoints + cn_df["Chromosome"].map(offsets)

    return genome_pos.values, chrom_boundaries, chrom_centers


def plot_genome_frequency(cn_df, clinical_df):
    """Plot genome-wide aberration frequency per subtype.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
    """
    log.info("=" * 60)
    log.info("SECTION 3: GENOME-WIDE FREQUENCY PLOTS")
    log.info("=" * 60)

    sample_cols = get_sample_columns(cn_df)
    genome_pos, chrom_boundaries, chrom_centers = compute_genome_positions(cn_df)

    # Build label lookup.
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, subtype in zip(axes, SUBTYPE_ORDER):
        # Get samples belonging to this subtype.
        subtype_samples = [s for s in sample_cols if label_map.get(s) == subtype]
        subtype_data = cn_df[subtype_samples]
        n = len(subtype_samples)

        # Compute frequency of gains+amplifications and losses.
        gain_freq = ((subtype_data >= 1).sum(axis=1)) / n
        loss_freq = -((subtype_data == -1).sum(axis=1)) / n

        ax.fill_between(genome_pos, 0, gain_freq, color="#E64B35", alpha=0.7, linewidth=0)
        ax.fill_between(genome_pos, 0, loss_freq, color="#3C5488", alpha=0.7, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Fraction of samples")
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.text(
            0.01, 0.95, f"{subtype} (n={n})",
            transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="top", color=SUBTYPE_COLORS[subtype],
        )

        # Chromosome boundaries.
        for i, (chrom, boundary) in enumerate(chrom_boundaries.items()):
            if i > 0:
                ax.axvline(boundary, color="grey", linewidth=0.3, alpha=0.5)

        ax.spines[["top", "right"]].set_visible(False)

    # Chromosome labels on bottom axis.
    chrom_labels = {c: ("X" if c == 23 else str(c)) for c in chrom_centers}
    axes[-1].set_xticks(list(chrom_centers.values()))
    axes[-1].set_xticklabels([chrom_labels[c] for c in chrom_centers], fontsize=7)
    axes[-1].set_xlabel("Genomic position (chromosome)")

    # Legend.
    legend_elements = [
        Patch(facecolor="#E64B35", alpha=0.7, label="Gain / Amplification"),
        Patch(facecolor="#3C5488", alpha=0.7, label="Loss"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=8)

    fig.subplots_adjust(hspace=0.1)
    fig.savefig(FIG_DIR / "02_genome_wide_frequency.png")
    plt.close(fig)
    log.info("Saved: 02_genome_wide_frequency.png")
    log.info("")


"""Section 4: PCA."""


def plot_pca(cn_df, clinical_df):
    """Perform PCA and plot first two components colored by subtype.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.

    Returns:
        tuple: (X_pca, labels, pca) - PCA-transformed matrix, subtype labels, fitted PCA.
    """
    log.info("=" * 60)
    log.info("SECTION 4: PCA")
    log.info("=" * 60)

    X = get_sample_matrix(cn_df)
    sample_cols = get_sample_columns(cn_df)
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))
    labels = [label_map[s] for s in sample_cols]

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    log.info("Variance explained by first 10 PCs:")
    for i, v in enumerate(explained):
        log.info("  PC%d: %.3f (%.1f%%)", i + 1, v, v * 100)
    log.info("  Cumulative (PC1-10): %.3f (%.1f%%)", explained.sum(), explained.sum() * 100)

    # Scree plot + 2D scatter.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Scree plot.
    ax = axes[0]
    ax.bar(range(1, 11), explained * 100, color="#4DBBD5", edgecolor="black", linewidth=0.3)
    ax.plot(range(1, 11), np.cumsum(explained) * 100, "o-", color="#E64B35", markersize=4)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    # Add legend for cumulative line.
    ax.legend(
        [plt.Line2D([0], [0], color="#E64B35", marker="o", markersize=4)],
        ["Cumulative variance"],
        fontsize=7, loc="center right",
    )

    # 2D scatter.
    ax = axes[1]
    for subtype in SUBTYPE_ORDER:
        mask = [l == subtype for l in labels]
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=SUBTYPE_COLORS[subtype], label=subtype,
            s=40, alpha=0.8, edgecolors="black", linewidths=0.3,
        )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "03_pca.png")
    plt.close(fig)
    log.info("Saved: 03_pca.png")

    # Save PCA results: sample coordinates and variance explained.
    pca_df = pd.DataFrame(
        X_pca,
        index=sample_cols,
        columns=[f"PC{i+1}" for i in range(10)],
    )
    pca_df.index.name = "Sample"
    pca_df.to_csv(OUT_DIR / "pca_coordinates.tsv", sep="\t")

    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(10)],
        "variance_explained": explained,
        "cumulative": np.cumsum(explained),
    })
    var_df.to_csv(OUT_DIR / "pca_variance_explained.tsv", sep="\t", index=False)
    log.info("Saved: pca_coordinates.tsv, pca_variance_explained.tsv")
    log.info("")

    return X_pca, labels, pca


"""Section 5: t-SNE."""


def plot_tsne(cn_df, clinical_df):
    """Perform t-SNE at multiple perplexities and plot colored by subtype.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
    """
    log.info("=" * 60)
    log.info("SECTION 5: t-SNE")
    log.info("=" * 60)

    X = get_sample_matrix(cn_df)
    sample_cols = get_sample_columns(cn_df)
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))
    labels = [label_map[s] for s in sample_cols]

    # Try multiple perplexity values.
    perplexities = [5, 15, 30]
    tsne_embeddings = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, perp in zip(axes, perplexities):
        log.info("  Running t-SNE with perplexity=%d ...", perp)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        tsne_embeddings[perp] = X_tsne

        for subtype in SUBTYPE_ORDER:
            mask = [l == subtype for l in labels]
            ax.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                c=SUBTYPE_COLORS[subtype], label=subtype,
                s=40, alpha=0.8, edgecolors="black", linewidths=0.3,
            )
        ax.set_xlabel("t-SNE dimension 1 (arbitrary units)")
        ax.set_ylabel("t-SNE dimension 2 (arbitrary units)")
        ax.text(
            0.02, 0.98, f"perplexity = {perp}",
            transform=ax.transAxes, fontsize=9, va="top",
        )
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(framealpha=0.9, loc="upper right")
    fig.savefig(FIG_DIR / "04_tsne.png")
    plt.close(fig)
    log.info("Saved: 04_tsne.png")

    # Save t-SNE embeddings for each perplexity.
    for perp, X_emb in tsne_embeddings.items():
        tsne_df = pd.DataFrame(
            X_emb, index=sample_cols, columns=["tSNE1", "tSNE2"],
        )
        tsne_df.index.name = "Sample"
        tsne_df.to_csv(OUT_DIR / f"tsne_perp{perp}.tsv", sep="\t")
    log.info("Saved: tsne_perp*.tsv")
    log.info("")


"""Section 6: Hierarchical Clustering."""


def plot_clustering(cn_df, clinical_df):
    """Hierarchical clustering dendrogram with subtype-colored leaves.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
    """
    log.info("=" * 60)
    log.info("SECTION 6: HIERARCHICAL CLUSTERING")
    log.info("=" * 60)

    X = get_sample_matrix(cn_df)
    sample_cols = get_sample_columns(cn_df)
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))

    # Ward linkage with Euclidean distance.
    Z = linkage(X, method="ward", metric="euclidean")

    # Silhouette score for k=3.
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cluster_labels = agg.fit_predict(X)
    sil = silhouette_score(X, cluster_labels)
    log.info("Silhouette score (k=3, Ward): %.3f", sil)

    # Dendrogram.
    fig, ax = plt.subplots(figsize=(14, 5))

    dendrogram(
        Z, ax=ax, leaf_rotation=90, leaf_font_size=6,
        labels=[f"{s} ({label_map[s]})" for s in sample_cols],
        color_threshold=0,
        above_threshold_color="grey",
    )

    # Color the leaf labels.
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        text = lbl.get_text()
        for subtype, color in SUBTYPE_COLORS.items():
            if subtype in text:
                lbl.set_color(color)
                break

    ax.set_ylabel("Ward linkage distance (Euclidean)")
    ax.set_xlabel("Sample (colored by true subtype)")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend with silhouette annotation.
    legend_elements = [Patch(facecolor=c, label=s) for s, c in SUBTYPE_COLORS.items()]
    leg = ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    ax.text(
        0.99, 0.78, f"Silhouette score (k=3) = {sil:.3f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.savefig(FIG_DIR / "05_hierarchical_clustering.png")
    plt.close(fig)
    log.info("Saved: 05_hierarchical_clustering.png")
    log.info("")


"""Section 7: Per-region Statistical Tests."""


def statistical_tests(cn_df, clinical_df, gene_map_df):
    """Run Kruskal-Wallis test per region to find discriminative features.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
        gene_map_df (pd.DataFrame): Gene coordinate mapping.

    Returns:
        tuple: (pvals, h_stats) - arrays of p-values and H-statistics per region.
    """
    log.info("=" * 60)
    log.info("SECTION 7: PER-REGION STATISTICAL TESTS")
    log.info("=" * 60)

    sample_cols = get_sample_columns(cn_df)

    # Kruskal-Wallis test per region.
    pvals, h_stats = kruskal_wallis_per_region(cn_df, clinical_df, sample_cols)

    # Bonferroni correction: 0.05 / number_of_tests.
    n_tests = len(pvals)
    bonf_alpha = bonferroni_threshold(n_tests)
    sig_bonf = pvals < bonf_alpha
    log.info("Bonferroni threshold: 0.05 / %d = %.2e", n_tests, bonf_alpha)
    log.info("Significant regions: %d / %d", sig_bonf.sum(), n_tests)

    # Identify and report the significant regions.
    sig_idx = np.where(sig_bonf)[0]
    sig_idx_sorted = sig_idx[np.argsort(pvals[sig_idx])]

    log.info("")
    log.info("%d regions passing Bonferroni correction:", sig_bonf.sum())
    log.info("%4s %3s %12s %12s %8s %12s %s", "Rank", "Chr", "Start", "End", "H-stat", "p-value", "Gene(s)")
    log.info("-" * 75)

    # Store info for labelling the plot.
    sig_labels = []
    for rank, idx in enumerate(sig_idx_sorted, 1):
        row = cn_df.iloc[idx]
        chrom = int(row["Chromosome"])
        start = int(row["Start"])
        end = int(row["End"])

        # Map to genes.
        gene_hits = gene_map_df[
            (gene_map_df["Chromosome"] == chrom)
            & (gene_map_df["Start"] <= end)
            & (gene_map_df["End"] >= start)
        ]["Gene"].unique()
        genes = ", ".join(gene_hits[:3]) if len(gene_hits) > 0 else "-"
        if len(gene_hits) > 3:
            genes += f" (+{len(gene_hits)-3})"

        # Pick a short label for the plot annotation.
        if len(gene_hits) > 0:
            sig_labels.append((idx, gene_hits[0]))
        else:
            sig_labels.append((idx, f"chr{chrom}:{start}"))

        log.info("%4d %3d %12d %12d %8.1f %12.2e %s", rank, chrom, start, end, h_stats[idx], pvals[idx], genes)

    # Also log the top 20 for reference.
    top_idx = np.argsort(pvals)[:20]
    log.info("")
    log.info("Top 20 most discriminative regions (for reference):")
    log.info("%4s %3s %12s %12s %8s %12s %s", "Rank", "Chr", "Start", "End", "H-stat", "p-value", "Gene(s)")
    log.info("-" * 75)
    for rank, idx in enumerate(top_idx, 1):
        row = cn_df.iloc[idx]
        chrom = int(row["Chromosome"])
        start = int(row["Start"])
        end = int(row["End"])
        gene_hits = gene_map_df[
            (gene_map_df["Chromosome"] == chrom)
            & (gene_map_df["Start"] <= end)
            & (gene_map_df["End"] >= start)
        ]["Gene"].unique()
        genes = ", ".join(gene_hits[:3]) if len(gene_hits) > 0 else "-"
        if len(gene_hits) > 3:
            genes += f" (+{len(gene_hits)-3})"
        log.info("%4d %3d %12d %12d %8.1f %12.2e %s", rank, chrom, start, end, h_stats[idx], pvals[idx], genes)

    # Manhattan-style plot of -log10(p).
    genome_pos, chrom_boundaries, chrom_centers = compute_genome_positions(cn_df)
    neg_log_p = -np.log10(pvals + 1e-300)  # Avoid log(0).
    bonf_thresh = -np.log10(bonf_alpha)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Alternate chromosome colors.
    chroms = cn_df["Chromosome"].values
    for chrom in sorted(cn_df["Chromosome"].unique()):
        mask = chroms == chrom
        color = "#3C5488" if chrom % 2 == 1 else "#4DBBD5"
        ax.scatter(genome_pos[mask], neg_log_p[mask], c=color, s=3, alpha=0.6, rasterized=True)

    # Significance threshold.
    ax.axhline(bonf_thresh, color="red", linestyle="--", linewidth=0.8,
               label=f"Bonferroni threshold (0.05 / {n_tests})")

    # Highlight and label the significant regions.
    for idx, label_text in sig_labels:
        ax.scatter(
            genome_pos[idx], neg_log_p[idx],
            c="red", s=25, zorder=5, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            label_text, (genome_pos[idx], neg_log_p[idx]),
            textcoords="offset points", xytext=(4, 6), fontsize=6,
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    chrom_labels = {c: ("X" if c == 23 else str(c)) for c in chrom_centers}
    ax.set_xticks(list(chrom_centers.values()))
    ax.set_xticklabels([chrom_labels[c] for c in chrom_centers], fontsize=7)
    ax.set_xlabel("Genomic position (chromosome)")
    ax.set_ylabel("Kruskal-Wallis significance: $-\\log_{10}$(p-value)")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "06_manhattan_kruskal.png")
    plt.close(fig)
    log.info("")
    log.info("Saved: 06_manhattan_kruskal.png")

    # Save per-region Kruskal-Wallis results.
    kw_df = pd.DataFrame({
        "Chromosome": cn_df["Chromosome"].values,
        "Start": cn_df["Start"].values,
        "End": cn_df["End"].values,
        "H_statistic": h_stats,
        "p_value": pvals,
        "significant_bonferroni": sig_bonf,
    })
    kw_df.to_csv(OUT_DIR / "kruskal_wallis_per_region.tsv", sep="\t", index=False)
    log.info("Saved: kruskal_wallis_per_region.tsv")
    log.info("")

    return pvals, h_stats


"""Section 8: Correlation vs Genomic Distance."""


def analyze_correlation(cn_df):
    """Analyze how CN correlation between regions decays with genomic distance.

    Compares all same-chromosome region pairs within 200 Mb (not just
    neighbors). Uses the gap between region boundaries (end of region i
    to start of region j) as the distance metric.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
    """
    log.info("=" * 60)
    log.info("SECTION 8: CORRELATION VS GENOMIC DISTANCE")
    log.info("=" * 60)

    sample_cols = get_sample_columns(cn_df)
    data = cn_df[sample_cols].values.astype(float)
    chroms = cn_df["Chromosome"].values
    starts = cn_df["Start"].values
    ends = cn_df["End"].values

    # Collect (distance, correlation) for all same-chromosome pairs within 200 Mb.
    max_dist = 200_000_000
    distances = []
    correlations = []

    chrom_list = sorted(cn_df["Chromosome"].unique())
    for chrom in track(chrom_list, description="Computing correlations"):
        idx = np.where(chroms == chrom)[0]
        for i_pos in range(len(idx)):
            i = idx[i_pos]
            for j_pos in range(i_pos + 1, len(idx)):
                j = idx[j_pos]
                # Gap distance: start of downstream region minus end of upstream region.
                gap = starts[j] - ends[i]
                if gap > max_dist:
                    break  # Regions are sorted by position, so further ones are even farther.
                if gap < 0:
                    gap = 0  # Overlapping regions - treat as distance zero.
                r = np.corrcoef(data[i], data[j])[0, 1]
                if not np.isnan(r):
                    distances.append(gap)
                    correlations.append(r)

    distances = np.array(distances)
    correlations = np.array(correlations)
    log.info("Total same-chromosome pairs (within %.0f Mb): %d", max_dist / 1e6, len(distances))

    # Cross-chromosome baseline: sample random pairs from different chromosomes.
    rng = np.random.RandomState(42)
    n_cross = 10000
    cross_corrs = []
    chrom_indices = {c: np.where(chroms == c)[0] for c in chrom_list}
    cross_count = 0
    while cross_count < n_cross:
        c1, c2 = rng.choice(chrom_list, size=2, replace=False)
        i = rng.choice(chrom_indices[c1])
        j = rng.choice(chrom_indices[c2])
        r = np.corrcoef(data[i], data[j])[0, 1]
        if not np.isnan(r):
            cross_corrs.append(r)
            cross_count += 1
    cross_corrs = np.array(cross_corrs)
    cross_median = np.median(cross_corrs)
    log.info("  Cross-chromosome baseline: median r = %.3f  (n=%d)", cross_median, n_cross)

    # Bin by distance and compute median correlation per bin.
    bin_edges_mb = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    bin_edges_bp = [x * 1_000_000 for x in bin_edges_mb]
    bin_labels = []
    bin_medians = []
    bin_counts = []
    bin_corr_arrays = []  # Store per-bin correlations for significance testing.

    for k in range(len(bin_edges_bp) - 1):
        lo, hi = bin_edges_bp[k], bin_edges_bp[k + 1]
        mask = (distances >= lo) & (distances < hi)
        n = mask.sum()
        if n > 0:
            bin_corrs = correlations[mask]
            med = np.median(bin_corrs)
            label = f"{bin_edges_mb[k]}-{bin_edges_mb[k+1]}"
            bin_labels.append(label)
            bin_medians.append(med)
            bin_counts.append(n)
            bin_corr_arrays.append(bin_corrs)
            # Mann-Whitney U test: is this bin significantly above cross-chromosome?
            u_stat, u_pval = stats.mannwhitneyu(
                bin_corrs, cross_corrs, alternative="greater",
            )
            sig_marker = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
            log.info("  %8s Mb: median r = %.3f  (n=%d)  vs cross-chr: p=%.2e %s",
                     label, med, n, u_pval, sig_marker)

    # Plot: correlation vs distance (scatter + binned medians).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: scatter of all pairs, subsampled for readability.
    ax = axes[0]
    if len(distances) > 8000:
        subsample = rng.choice(len(distances), 8000, replace=False)
    else:
        subsample = np.arange(len(distances))
    ax.scatter(
        distances[subsample] / 1e6, correlations[subsample],
        s=1, alpha=0.1, color="#3C5488", rasterized=True,
    )
    # Cross-chromosome baseline as horizontal line.
    ax.axhline(
        cross_median, color="#E64B35", linestyle="--", linewidth=1,
        label=f"Cross-chromosome median (r={cross_median:.2f})",
    )
    ax.set_xlabel("Gap between region boundaries (Mb)")
    ax.set_ylabel("Pearson correlation (r) between CN profiles")
    ax.set_xlim(-2, 205)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # Right panel: binned median correlation + cross-chromosome bar.
    ax = axes[1]
    # Same-chromosome bins.
    n_bins = len(bin_labels)
    x_pos = list(range(n_bins))
    bars = ax.bar(x_pos, bin_medians, color="#00A087", edgecolor="black", linewidth=0.3)
    # Cross-chromosome bar, separated by a gap.
    cross_x = n_bins + 0.5
    cross_bar = ax.bar(
        cross_x, cross_median, color="#E64B35", edgecolor="black", linewidth=0.3,
    )
    ax.text(cross_x, cross_median + 0.02, f"n={n_cross}",
            ha="center", va="bottom", fontsize=5.5)

    all_x = x_pos + [cross_x]
    all_labels = bin_labels + ["cross-\nchr"]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Gap distance bin (Mb)")
    ax.set_ylabel("Median Pearson correlation (r)")
    ax.set_ylim(0, 1)
    # Annotate counts and significance on same-chromosome bars.
    for bar, n, bin_corrs in zip(bars, bin_counts, bin_corr_arrays):
        _, u_pval = stats.mannwhitneyu(bin_corrs, cross_corrs, alternative="greater")
        sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else "n.s."
        bar_x = bar.get_x() + bar.get_width() / 2
        bar_top = bar.get_height()
        ax.text(bar_x, bar_top + 0.02, f"n={n}", ha="center", va="bottom", fontsize=5.5)
        ax.text(bar_x, bar_top + 0.06, sig, ha="center", va="bottom", fontsize=6,
                fontweight="bold", color="black" if sig != "n.s." else "grey")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "07_correlation_vs_distance.png")
    plt.close(fig)
    log.info("")
    log.info("Saved: 07_correlation_vs_distance.png")

    # Save binned correlation summary.
    corr_df = pd.DataFrame({
        "distance_bin_Mb": bin_labels,
        "median_r": bin_medians,
        "n_pairs": bin_counts,
    })
    # Append cross-chromosome row.
    cross_row = pd.DataFrame({
        "distance_bin_Mb": ["cross_chromosome"],
        "median_r": [cross_median],
        "n_pairs": [n_cross],
    })
    corr_df = pd.concat([corr_df, cross_row], ignore_index=True)
    corr_df.to_csv(OUT_DIR / "correlation_vs_distance.tsv", sep="\t", index=False)
    log.info("Saved: correlation_vs_distance.tsv")
    log.info("")


"""Section 9: Sample Similarity Matrix."""


def plot_similarity_matrix(cn_df, clinical_df):
    """Plot pairwise Hamming distance matrix ordered by subtype.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
    """
    log.info("=" * 60)
    log.info("SECTION 9: SAMPLE SIMILARITY MATRIX")
    log.info("=" * 60)

    sample_cols = get_sample_columns(cn_df)
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))

    # Order samples by subtype.
    ordered_samples = []
    for subtype in SUBTYPE_ORDER:
        ordered_samples.extend([s for s in sample_cols if label_map.get(s) == subtype])

    X = cn_df[ordered_samples].T.values
    dist = pdist(X, metric="hamming")
    dist_mat = squareform(dist)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_mat, cmap="viridis_r", aspect="equal")

    # Subtype separators.
    cumulative = 0
    for subtype in SUBTYPE_ORDER:
        n = sum(1 for s in ordered_samples if label_map.get(s) == subtype)
        cumulative += n
        if subtype != SUBTYPE_ORDER[-1]:
            ax.axhline(cumulative - 0.5, color="white", linewidth=1)
            ax.axvline(cumulative - 0.5, color="white", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Samples (grouped by subtype)")
    ax.set_ylabel("Samples (grouped by subtype)")

    # Add subtype labels.
    cum = 0
    for subtype in SUBTYPE_ORDER:
        n = sum(1 for s in ordered_samples if label_map.get(s) == subtype)
        ax.text(-2, cum + n / 2, subtype, va="center", ha="right",
                fontsize=9, fontweight="bold", color=SUBTYPE_COLORS[subtype])
        cum += n

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Hamming distance (fraction of differing CN calls)")

    fig.savefig(FIG_DIR / "08_similarity_matrix.png")
    plt.close(fig)
    log.info("Saved: 08_similarity_matrix.png")

    # Save Hamming distance matrix.
    dist_df = pd.DataFrame(dist_mat, index=ordered_samples, columns=ordered_samples)
    dist_df.index.name = "Sample"
    dist_df.to_csv(OUT_DIR / "hamming_distance_matrix.tsv", sep="\t")
    log.info("Saved: hamming_distance_matrix.tsv")
    log.info("")


"""Main."""


def main():
    """Run all Phase 0 exploration sections."""
    parser = argparse.ArgumentParser(
        description="Phase 0 data exploration (raw or merged data).",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to alternative CN data file (e.g. merged segments TSV).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix for output directories and log file (e.g. 'merged').",
    )
    args = parser.parse_args()

    setup(tag=args.tag)

    cn_df, clinical_df, gene_map_df = load_data(input_path=args.input)

    # Data basics.
    validate_data(cn_df, clinical_df)
    plot_class_distribution(clinical_df)

    # Genome-wide frequency.
    plot_genome_frequency(cn_df, clinical_df)

    # Dimensionality reduction.
    plot_pca(cn_df, clinical_df)
    plot_tsne(cn_df, clinical_df)

    # Clustering.
    plot_clustering(cn_df, clinical_df)

    # Statistical tests.
    statistical_tests(cn_df, clinical_df, gene_map_df)

    # Feature correlation.
    analyze_correlation(cn_df)

    # Sample similarity.
    plot_similarity_matrix(cn_df, clinical_df)

    log.info("=" * 60)
    log.info("PHASE 0 EXPLORATION COMPLETE.")
    log.info("  Figures: %s", FIG_DIR)
    log.info("  Data:    %s", OUT_DIR)
    suffix = f"_{args.tag}" if args.tag else ""
    log.info("  Log:     %s",
             PROJECT_DIR / "results" / "logs" / f"data_exploration_phase{suffix}.log")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
