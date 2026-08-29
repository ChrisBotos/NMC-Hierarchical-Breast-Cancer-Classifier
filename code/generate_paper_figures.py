"""
Group 9.
Authors:
    Alexandros Michailidis.
    Antonie Wagner.
    Christos Botos.
    Yan Qiao.
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: generate_paper_figures.py.
Description:
    Generates the final publication-quality figures and LaTeX table for the
    research paper. Produces exactly 3 figures + 1 table:
      Figure 1: Methodology flowchart.
      Figure 2: Feature selection stability curve with top regions.
      Figure 3: Multi-panel results (3-class BA violins + confusion matrices).
      Table 1:  Summary statistics LaTeX code.

    Metric glossary (see docs/figure_glossary.md):
      BA   = 3-class balanced accuracy (flat or hierarchical combined).
      BA1  = Stage 1 balanced accuracy (HER2+ vs rest, always 1.0).
      BA2  = Stage 2 balanced accuracy (HR+ vs TN).
      Flat BA and hierarchical combined BA are both 3-class BA and are
      directly comparable on the same axis.

Usage:
    python3 code/generate_paper_figures.py

Dependencies:
    Python >= 3.10.
    matplotlib, pandas, numpy.
"""

from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Resolve paths from script location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "latex_final_report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths.
FLAT_DATA = (
    PROJECT_ROOT
    / "results"
    / "2026-04-22_server_run_flat_nested_CV_2x2"
    / "nested_cv_2x2"
    / "data"
)
HIER_DATA = (
    PROJECT_ROOT
    / "results"
    / "2026-04-25_final_hierarchical"
    / "hierarchical_nested_cv"
    / "data"
)
GENE_MAP_PATH = PROJECT_ROOT / "data" / "BasepairToGeneMap.tsv"

# Global style.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colorblind-safe palette.
BLUE = "#4477AA"
ORANGE = "#EE6677"
GREEN = "#228833"

# Pipeline configuration (kw_nmc_kens excluded from all outputs).
FLAT_PIPES = ["kw_nmc", "kw_rf", "en_nmc", "en_rf"]
HIER_BASE_PIPES = ["kw_nmc", "kw_rf", "en_nmc", "en_rf"]
HIER_PLAT_PIPES = ["kw_nmc_pens", "en_nmc_pens", "nmc_pens_ensemble"]

PIPE_LABELS = {
    "kw_nmc": "KW+NMC",
    "kw_rf": "KW+RF",
    "en_nmc": "EN+NMC",
    "en_rf": "EN+RF",
    "kw_nmc_pens": "KW+NMC(P)",
    "en_nmc_pens": "EN+NMC(P)",
    "nmc_pens_ensemble": "Ensemble",
}
PIPE_COLORS = {
    "kw_nmc": BLUE,
    "kw_rf": ORANGE,
    "en_nmc": BLUE,
    "en_rf": ORANGE,
    "kw_nmc_pens": BLUE,
    "en_nmc_pens": BLUE,
    "nmc_pens_ensemble": BLUE,
}

# Chromosome colors for Figure 2 (colorblind-safe, distinct hues).
CHR_COLORS = {
    "chr5": "#EE6677",  # Red/pink.
    "chr6": "#228833",  # Green.
    "chr12": "#4477AA",  # Blue.
    "chr15": "#CCBB44",  # Yellow.
    "chr16": "#AA3377",  # Purple.
}
CHR_COLOR_DEFAULT = "#BBBBBB"  # Grey for other chromosomes.

# Well-known cancer genes for annotation.
CANCER_GENES = {
    "CDK4",
    "MDM2",
    "ERBB2",
    "ERBB3",
    "TP53",
    "BRCA1",
    "BRCA2",
    "RB1",
    "MYC",
    "CCND1",
    "PTEN",
    "PIK3CA",
    "AKT1",
    "EGFR",
    "FGFR1",
    "FGFR2",
    "ESR1",
    "PGR",
    "GATA3",
    "FOXA1",
    "MAP3K1",
    "CDH1",
    "RUNX1",
    "TBX3",
    "FGF19",
    "FGF3",
    "FGF4",
    "KRAS",
    "NRAS",
    "BRAF",
    "APC",
    "SMAD4",
    "KMT2C",
    "KMT2D",
    "NF1",
    "NF2",
    "CYLD",
    "PALB2",
    "ATM",
    "CHEK2",
    "RAD51C",
    "RAD51D",
    "MLH1",
    "MSH2",
    "NCOR1",
    "MEN1",
    "CDKN2A",
    "CREBBP",
    "EP300",
    "ARID1A",
    "SMARCA4",
    "CTCF",
    "STAG2",
    "BIRC3",
    "BCL2",
    "MCL1",
    "CUL3",
    "NOTCH1",
    "NOTCH2",
    "IRF4",
    "MYB",
    "SHH",
    "WNT",
    "FOXO3",
    "PTCH1",
    "SMO",
    "GLI1",
    "GLI2",
}


"""Data loading"""


def load_flat_results():
    """Load flat 3-class nested CV fold results.

    Returns:
        pd.DataFrame: Combined fold results for the 4 flat pipelines.
    """
    frames = []
    for pipe in FLAT_PIPES:
        for r in range(1, 51):
            fpath = FLAT_DATA / f"fold_results_{pipe}_r{r}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                frames.append(
                    df[
                        [
                            "pipeline",
                            "repeat",
                            "outer_fold",
                            "balanced_accuracy",
                            "n_features_selected",
                            "y_true",
                            "y_pred",
                        ]
                    ]
                )
    return pd.concat(frames, ignore_index=True)


def load_hierarchical_results():
    """Load hierarchical nested CV fold results (base + plateau + ensemble).

    Returns:
        pd.DataFrame: Combined fold results for hierarchical pipelines.
    """
    all_pipes = HIER_BASE_PIPES + HIER_PLAT_PIPES
    frames = []
    for pipe in all_pipes:
        for r in range(1001, 1201):
            fpath = HIER_DATA / f"fold_results_{pipe}_r{r}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                frames.append(
                    df[
                        [
                            "stage2_pipeline",
                            "repeat",
                            "outer_fold",
                            "stage1_bal_acc",
                            "stage2_bal_acc",
                            "combined_bal_acc",
                            "stage2_n_features",
                            "y_true",
                            "y_pred",
                        ]
                    ]
                )
    return pd.concat(frames, ignore_index=True)


def load_gene_map():
    """Load the basepair-to-gene mapping table.

    Returns:
        pd.DataFrame: Gene map with Chromosome, Start, End, Gene columns.
    """
    return pd.read_csv(
        GENE_MAP_PATH,
        sep="\t",
        dtype={"Chromosome": str, "Start": int, "End": int, "Gene": str},
    )


"""Helpers"""


def per_repeat_means(df, ba_col, pipe_col):
    """Compute per-repeat mean BA for each pipeline.

    Args:
        df (pd.DataFrame): Fold-level results.
        ba_col (str): Column name for balanced accuracy.
        pipe_col (str): Column name for pipeline identifier.

    Returns:
        dict: pipeline -> array of per-repeat mean BA values.
    """
    return {
        pipe: grp.groupby("repeat")[ba_col].mean().values
        for pipe, grp in df.groupby(pipe_col)
    }


def per_class_recalls(df, pipe_col):
    """Compute per-class recall from y_true/y_pred strings.

    Args:
        df (pd.DataFrame): Fold-level results with y_true and y_pred columns.
        pipe_col (str): Column name for pipeline identifier.

    Returns:
        dict: pipeline -> {'HER2+': float, 'HR+': float, 'Triple Neg': float}.
    """
    results = {}
    for pipe, grp in df.groupby(pipe_col):
        all_true, all_pred = [], []
        for _, row in grp.iterrows():
            all_true.extend(str(row["y_true"]).split(","))
            all_pred.extend(str(row["y_pred"]).split(","))
        recalls = {}
        for cls in ["HER2+", "HR+", "Triple Neg"]:
            correct = sum(
                1
                for t, p in zip(all_true, all_pred)
                if t.strip() == cls and p.strip() == cls
            )
            total = sum(1 for t in all_true if t.strip() == cls)
            recalls[cls] = correct / total if total > 0 else 0.0
        results[pipe] = recalls
    return results


def per_repeat_class_recalls(df, pipe_col):
    """Compute per-class recall per repeat for SD estimation.

    Args:
        df (pd.DataFrame): Fold-level results with y_true and y_pred columns.
        pipe_col (str): Column name for pipeline identifier.

    Returns:
        dict: pipeline -> {'HER2+': array, 'HR+': array, 'Triple Neg': array}
              where each array contains per-repeat recall values.
    """
    results = {}
    for pipe, grp in df.groupby(pipe_col):
        repeat_recalls = {"HER2+": [], "HR+": [], "Triple Neg": []}
        for repeat_id, rgrp in grp.groupby("repeat"):
            all_true, all_pred = [], []
            for _, row in rgrp.iterrows():
                all_true.extend(str(row["y_true"]).split(","))
                all_pred.extend(str(row["y_pred"]).split(","))
            for cls in ["HER2+", "HR+", "Triple Neg"]:
                correct = sum(
                    1
                    for t, p in zip(all_true, all_pred)
                    if t.strip() == cls and p.strip() == cls
                )
                total = sum(1 for t in all_true if t.strip() == cls)
                repeat_recalls[cls].append(correct / total if total > 0 else 0.0)
        results[pipe] = {cls: np.array(vals) for cls, vals in repeat_recalls.items()}
    return results


def draw_violin(ax, data_list, positions, colors, width=0.65):
    """Draw violin + jittered strip + median line on axes.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        data_list (list): List of arrays.
        positions (list): X positions.
        colors (list): Fill colors.
        width (float): Violin width.
    """
    parts = ax.violinplot(
        data_list,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
        body.set_edgecolor(colors[i])
        body.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, (data, px) in enumerate(zip(data_list, positions)):
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(
            px + jitter,
            data,
            s=5,
            alpha=0.15,
            color=colors[i],
            edgecolors="none",
            zorder=4,
        )
        med = np.median(data)
        ax.plot(
            [px - 0.2, px + 0.2],
            [med, med],
            color=colors[i],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=5,
        )


def shorten_region(name):
    """Shorten a genomic region name for display.

    Args:
        name (str): Full region name like chr17_35076296_35282086.

    Returns:
        str: Shortened name like chr17:35.1-35.3M.
    """
    parts = name.split("_")
    if len(parts) >= 3:
        chrom = parts[0]
        start_mb = int(parts[1]) / 1e6
        end_mb = int(parts[2]) / 1e6
        return f"{chrom}:{start_mb:.1f}-{end_mb:.1f}M"
    return name


def get_chromosome(region_name):
    """Extract chromosome name from a region identifier.

    Args:
        region_name (str): Region name like chr17_35076296_35282086.

    Returns:
        str: Chromosome name like chr17.
    """
    return region_name.split("_")[0]


def find_best_gene(region_name, gene_map_df):
    """Find the most prominent gene overlapping a genomic region.

    Prioritizes well-known cancer genes. If none found, returns the first
    overlapping gene. Returns empty string if no overlap.

    Args:
        region_name (str): Region name like chr12_36739877_67332062.
        gene_map_df (pd.DataFrame): Gene map with Chromosome, Start, End, Gene.

    Returns:
        str: Gene name or empty string.
    """
    parts = region_name.split("_")
    if len(parts) < 3:
        return ""
    chrom_num = parts[0].replace("chr", "")
    start = int(parts[1])
    end = int(parts[2])

    # Filter gene map for overlapping genes on this chromosome.
    mask = (
        (gene_map_df["Chromosome"].astype(str) == chrom_num)
        & (gene_map_df["Start"] < end)
        & (gene_map_df["End"] > start)
    )
    overlapping = gene_map_df.loc[mask, "Gene"].unique()

    if len(overlapping) == 0:
        return ""

    # Prefer well-known cancer genes.
    cancer_hits = [g for g in overlapping if g in CANCER_GENES]
    if cancer_hits:
        return cancer_hits[0]
    return overlapping[0]


# =========================================================================
# FIGURE 1 - Methodology flowchart
# =========================================================================


def generate_fig1():
    """Generate the methodology flowchart (Figure 1, full-width).

    Uses a uniform two-color scheme: light gray for shared steps, light blue
    for the experimental paths. A clean downward flow converges through a
    model selection step before the final model.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 13.0)
    ax.axis("off")

    # Two-color scheme: gray (shared), blue accent (experimental paths).
    gf, ge = "#EDEDED", "#555555"  # Gray fill, gray edge.
    af, ae = "#D6E4F0", "#3A6EA5"  # Accent blue fill, accent blue edge.
    LS = 0.30  # Line spacing in data coords.

    def box(x, y, w, h, fc, ec, lines, fs=7, bold1=True):
        """Draw rounded-rect box with multi-line text."""
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.15",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.0,
                zorder=2,
            )
        )
        cx, cy = x + w / 2, y + h / 2
        top = cy + (len(lines) - 1) * LS / 2
        for i, ln in enumerate(lines):
            fw = "bold" if (i == 0 and bold1) else "normal"
            ax.text(
                cx,
                top - i * LS,
                ln,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=fw,
                zorder=3,
                fontfamily="serif",
            )

    def arrow(x1, y1, x2, y2):
        """Draw straight arrow."""
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->",
                mutation_scale=12,
                connectionstyle="arc3,rad=0",
                color="#333333",
                linewidth=1.0,
                zorder=1,
            )
        )

    def arrow_curved(x1, y1, x2, y2, rad=0.15):
        """Draw curved arrow."""
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->",
                mutation_scale=12,
                connectionstyle=f"arc3,rad={rad}",
                color="#333333",
                linewidth=1.0,
                zorder=1,
            )
        )

    # Dimensions.
    bw = 3.6  # Box width.
    bh = 0.9  # Standard box height.
    bht = 1.2  # Tall box height (3 lines).
    cc = 7.0  # Center x.
    cl = 3.2  # Left column center.
    cr = 10.8  # Right column center.

    # Y positions (top to bottom).
    y_data = 11.6
    y_merge = 10.2
    y_cv = 8.8
    y_flat = 6.8
    y_s1 = 6.8
    y_s2 = 5.1
    y_plat = 3.6
    y_select = 2.0
    y_final = 0.5

    # ---- Shared steps (gray) ----
    box(
        cc - bw / 2,
        y_data,
        bw,
        bh,
        gf,
        ge,
        ["DATA", "100 samples, 2,834 regions, 3 classes"],
    )
    box(
        cc - bw / 2,
        y_merge,
        bw,
        bh,
        gf,
        ge,
        ["REGION MERGING", "Pearson r > 0.8, 273 segments"],
    )
    box(cc - bw / 2, y_cv, bw, bh, gf, ge, ["OUTER CV", "Repeated stratified 5-fold"])
    arrow(cc, y_data, cc, y_merge + bh)
    arrow(cc, y_merge, cc, y_cv + bh)

    # ---- Flat path (accent blue) ----
    box(
        cl - bw / 2,
        y_flat,
        bw,
        bht,
        af,
        ae,
        [
            "FLAT 3-CLASS",
            "[KW / EN] x [NMC / RF]",
            "Inner 5-fold CV tuning",
            "50 repeats x 5 folds = 250",
        ],
        fs=6.5,
    )
    arrow_curved(cc - bw / 2, y_cv + bh * 0.3, cl + bw / 2, y_flat + bht, rad=0.15)

    # ---- Hierarchical path (accent blue) ----
    box(
        cr - bw / 2,
        y_s1,
        bw,
        bh,
        af,
        ae,
        ["STAGE 1: HER2+ vs rest", "Fixed KW+RF, k = 5, BA1 = 1.0"],
        fs=6.5,
    )
    arrow_curved(cc + bw / 2, y_cv + bh * 0.3, cr - bw / 2, y_s1 + bh, rad=-0.15)

    box(
        cr - bw / 2,
        y_s2,
        bw,
        bht,
        af,
        ae,
        [
            "STAGE 2: HR+ vs TN",
            "[KW / EN] x [NMC / RF]",
            "Inner 5-fold CV tuning",
            "200 repeats x 5 folds = 1,000",
        ],
        fs=6.5,
    )
    arrow(cr, y_s1, cr, y_s2 + bht)
    ax.text(
        cr + bw / 2 + 0.15,
        (y_s1 + y_s2 + bht) / 2,
        "non-HER2+\nsamples",
        ha="left",
        va="center",
        fontsize=5.5,
        style="italic",
        color="#555555",
    )

    box(
        cr - bw / 2,
        y_plat,
        bw,
        bh,
        af,
        ae,
        ["PLATEAU ENSEMBLING", "Top configs by inner score"],
    )
    arrow(cr, y_s2, cr, y_plat + bh)

    # ---- Group labels ----
    ax.text(
        cl,
        y_flat + bht + 0.25,
        "Flat experiment",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=ae,
    )
    ax.text(
        cr,
        y_s1 + bh + 0.25,
        "Hierarchical experiment",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=ae,
    )

    # ---- Model selection (gray, convergence point) ----
    box(
        cc - bw / 2,
        y_select,
        bw,
        bh,
        gf,
        ge,
        ["MODEL SELECTION", "Best pipeline by BA2"],
    )
    arrow_curved(cl, y_flat, cc - bw / 2, y_select + bh * 0.5, rad=0.2)
    arrow_curved(cr, y_plat, cc + bw / 2, y_select + bh * 0.5, rad=-0.2)

    # ---- Final model (gray) ----
    box(
        cc - bw / 2,
        y_final,
        bw,
        bh,
        gf,
        ge,
        ["FINAL MODEL", "EN+NMC(P)", "Retrained on 100 samples"],
    )
    arrow(cc, y_select, cc, y_final + bh)
    arrow(cc, y_final, cc, y_final - 0.25)
    ax.text(
        cc,
        y_final - 0.35,
        "Predictions (57 validation samples)",
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#333333",
    )

    # ---- Legend ----
    shared_patch = mpatches.Patch(
        facecolor=gf, edgecolor=ge, linewidth=0.8, label="Shared steps"
    )
    expt_patch = mpatches.Patch(
        facecolor=af, edgecolor=ae, linewidth=0.8, label="Experimental paths"
    )
    ax.legend(
        handles=[shared_patch, expt_patch],
        loc="upper right",
        fontsize=6.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout(pad=0.3)
    fig.savefig(FIGURES_DIR / "fig1_workflow.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / "fig1_workflow.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved fig1_workflow")


# =========================================================================
# FIGURE 3 - Multi-panel: (A) 3-class BA violins, (B) confusion matrices
# =========================================================================


def generate_fig2():
    """Generate the main results figure (Figure 3, full-width, stacked panels).

    Panel A (top, full width): 3-class BA violin comparison (flat BA and
    hierarchical combined BA on the same y-axis).
    Panel B (bottom, full width): Confusion matrices for best flat (KW+RF)
    and best hierarchical (EN+NMC(P)) pipelines, side by side with colorbar.
    Zero cells use regular font weight; non-zero cells use bold.
    """
    flat_df = load_flat_results()
    hier_df = load_hierarchical_results()

    # ---- Compute data for both panels ----
    flat_means = per_repeat_means(flat_df, "balanced_accuracy", "pipeline")
    hier_means = per_repeat_means(hier_df, "combined_bal_acc", "stage2_pipeline")

    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 0.7], hspace=0.55)

    # ---- Panel A: 3-class BA violins (top row, full width) ----
    ax_a = fig.add_subplot(gs[0])

    # Three groups: flat (4), hierarchical base (4), hierarchical plateau (3).
    pos_f = [1, 2, 3, 4]
    pos_hb = [6, 7, 8, 9]
    pos_hp = [11, 12, 13]

    data_all, pos_all, col_all = [], [], []
    for pipes, positions in [
        (FLAT_PIPES, pos_f),
        (HIER_BASE_PIPES, pos_hb),
        (HIER_PLAT_PIPES, pos_hp),
    ]:
        source = flat_means if pipes is FLAT_PIPES else hier_means
        for pipe, px in zip(pipes, positions):
            if pipe in source:
                data_all.append(source[pipe])
                pos_all.append(px)
                col_all.append(PIPE_COLORS[pipe])

    draw_violin(ax_a, data_all, pos_all, col_all)

    # Group separators.
    ax_a.axvline(x=5, color="#CCCCCC", linestyle=":", linewidth=0.7, zorder=0)
    ax_a.axvline(x=10, color="#CCCCCC", linestyle=":", linewidth=0.7, zorder=0)

    lbl_all = []
    for pipes in [FLAT_PIPES, HIER_BASE_PIPES, HIER_PLAT_PIPES]:
        for p in pipes:
            if p in (flat_means if pipes is FLAT_PIPES else hier_means):
                lbl_all.append(PIPE_LABELS[p])
    ax_a.set_xticks(pos_all)
    ax_a.set_xticklabels(lbl_all, rotation=40, ha="right")
    ax_a.set_ylabel("3-class balanced accuracy")

    # Group annotations at top.
    ylim = ax_a.get_ylim()
    top_y = ylim[1] + 0.005
    ax_a.text(
        2.5,
        top_y,
        "Flat",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#444444",
    )
    ax_a.text(
        7.5,
        top_y,
        "Hierarchical\n(base)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#444444",
    )
    ax_a.text(
        12,
        top_y,
        "Hierarchical\n(plateau)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#444444",
    )

    nmc_p = mpatches.Patch(color=BLUE, alpha=0.5, label="NMC-based")
    rf_p = mpatches.Patch(color=ORANGE, alpha=0.5, label="RF-based")
    ax_a.legend(handles=[nmc_p, rf_p], loc="lower left", framealpha=0.9, fontsize=7)
    ax_a.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.text(
        -0.02,
        1.10,
        "(A)",
        transform=ax_a.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    # ---- Panel B: Confusion matrices (bottom row, full width) ----
    gs_bottom = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    ax_cm1 = fig.add_subplot(gs_bottom[0])
    ax_cm2 = fig.add_subplot(gs_bottom[1])
    ax_cb = fig.add_subplot(gs_bottom[2])

    classes = ["HER2+", "HR+", "Triple Neg"]
    short_cls = ["HER2+", "HR+", "TN"]

    def build_cm(df, pipe_col, pipe_name):
        """Build row-normalized confusion matrix from fold results.

        Args:
            df (pd.DataFrame): Fold results.
            pipe_col (str): Pipeline column name.
            pipe_name (str): Pipeline to filter.

        Returns:
            np.ndarray: 3x3 row-normalized confusion matrix.
        """
        sub = df[df[pipe_col] == pipe_name]
        cm = np.zeros((3, 3))
        for _, row in sub.iterrows():
            yt = [t.strip() for t in str(row["y_true"]).split(",")]
            yp = [p.strip() for p in str(row["y_pred"]).split(",")]
            for t, p in zip(yt, yp):
                if t in classes and p in classes:
                    cm[classes.index(t), classes.index(p)] += 1
        rsums = cm.sum(axis=1, keepdims=True)
        rsums[rsums == 0] = 1
        return cm / rsums

    cm_flat = build_cm(flat_df, "pipeline", "kw_rf")
    cm_hier = build_cm(hier_df, "stage2_pipeline", "en_nmc_pens")

    im = None
    for ax, cm, title in [
        (ax_cm1, cm_flat, "Flat: KW+RF"),
        (ax_cm2, cm_hier, "Hierarchical: EN+NMC(P)"),
    ]:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(short_cls, fontsize=9)
        ax.set_yticklabels(short_cls, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        for i in range(3):
            for j in range(3):
                v = cm[i, j]
                c = "white" if v > 0.5 else "black"
                # De-emphasize near-zero cells with regular font weight.
                fw = "normal" if v < 0.01 else "bold"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=c,
                    fontweight=fw,
                )

    # Shared colorbar.
    fig.colorbar(im, cax=ax_cb, label="Recall (row-normalized)")

    ax_cm1.text(
        -0.02,
        1.38,
        "(B)",
        transform=ax_cm1.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    fig.savefig(FIGURES_DIR / "fig3_results.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / "fig3_results.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved fig3_results")


# =========================================================================
# FIGURE 2 - Feature selection stability
# =========================================================================


def generate_fig3():
    """Generate feature selection stability figure (Figure 2, full-width).

    Left: frequency drop-off curve across all features.
    Right: top 15 most frequently selected regions, color-coded by chromosome,
    with gene name annotations where available.
    Data from EN+NMC(P) pipeline (best hierarchical).
    """
    # Load Stage 2 features for en_nmc_pens.
    feat_counter = Counter()
    frames = []
    for r in range(1001, 1201):
        fpath = HIER_DATA / f"fold_results_en_nmc_pens_r{r}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, usecols=["stage2_features"])
            frames.append(df)
    feat_df = pd.concat(frames, ignore_index=True)
    total_folds = len(feat_df)

    for _, row in feat_df.iterrows():
        s = str(row["stage2_features"])
        if s and s != "nan":
            feat_counter.update(f.strip() for f in s.split(",") if f.strip())

    if not feat_counter:
        print("  SKIP: no features found")
        return

    all_feats = feat_counter.most_common()
    all_freqs = [c / total_folds for _, c in all_feats]

    # Load gene map for annotations.
    gene_map_df = load_gene_map()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 3.8), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # Left: drop-off curve.
    ax1.plot(range(len(all_freqs)), all_freqs, color=BLUE, linewidth=1.2)
    ax1.fill_between(range(len(all_freqs)), all_freqs, alpha=0.15, color=BLUE)
    ax1.set_xlabel("Features (ranked by selection frequency)")
    ax1.set_ylabel("Selection frequency\n(fraction of folds)")
    ax1.set_xlim(0, len(all_freqs))
    ax1.set_ylim(0, 1.05)
    n_50 = sum(1 for f in all_freqs if f >= 0.5)
    ax1.axhline(y=0.5, color="#999999", linestyle="--", linewidth=0.7, zorder=0)
    ax1.text(
        len(all_freqs) * 0.95,
        0.52,
        f"{n_50} regions in >50% of folds",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#666666",
    )
    ax1.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.text(
        -0.02,
        1.06,
        "(A)",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    # Right: top 15 regions as horizontal bars, color-coded by chromosome.
    top_n = 15
    top = feat_counter.most_common(top_n)

    # Build labels with gene annotations.
    names = []
    bar_colors = []
    for feat_name, _ in top:
        short = shorten_region(feat_name)
        gene = find_best_gene(feat_name, gene_map_df)
        if gene:
            short = f"{short} ({gene})"
        names.append(short)
        chrom = get_chromosome(feat_name)
        bar_colors.append(CHR_COLORS.get(chrom, CHR_COLOR_DEFAULT))

    freqs = [f[1] / total_folds for f in top]

    y_pos = np.arange(top_n)
    ax2.barh(
        y_pos,
        freqs,
        color=bar_colors,
        alpha=0.8,
        edgecolor=[c for c in bar_colors],
        linewidth=0.3,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=5.0)
    ax2.set_xlabel("Frequency")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.text(
        -0.02,
        1.06,
        "(B)",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    # Chromosome color legend.
    chr_legend_handles = []
    for chrom in ["chr5", "chr6", "chr12", "chr15", "chr16"]:
        chr_legend_handles.append(
            mpatches.Patch(color=CHR_COLORS[chrom], alpha=0.8, label=chrom)
        )
    chr_legend_handles.append(
        mpatches.Patch(color=CHR_COLOR_DEFAULT, alpha=0.8, label="other")
    )
    ax2.legend(
        handles=chr_legend_handles,
        loc="lower right",
        fontsize=5,
        framealpha=0.9,
        ncol=2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.5,
        bbox_to_anchor=(1.55, 0.0),
    )

    plt.tight_layout(pad=1.0)
    fig.savefig(FIGURES_DIR / "fig2_features.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / "fig2_features.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved fig2_features")


# =========================================================================
# TABLE 1 - Summary statistics
# =========================================================================


def generate_table1():
    """Generate LaTeX summary table with pipeline performance metrics.

    Uses consistent terminology: BA (3-class), BA2 (Stage 2), BA1 = 1.0.
    All values reported as mean +/- SD of per-repeat means.
    """
    flat_df = load_flat_results()
    hier_df = load_hierarchical_results()

    flat_rec = per_class_recalls(flat_df, "pipeline")
    hier_rec = per_class_recalls(hier_df, "stage2_pipeline")
    flat_rec_sd = per_repeat_class_recalls(flat_df, "pipeline")
    hier_rec_sd = per_repeat_class_recalls(hier_df, "stage2_pipeline")

    rows = []
    for pipe in FLAT_PIPES:
        sub = flat_df[flat_df["pipeline"] == pipe]
        repeat_ba = sub.groupby("repeat")["balanced_accuracy"].mean()
        ba_mean, ba_sd = repeat_ba.mean(), repeat_ba.std()
        feat = sub["n_features_selected"].median()
        r = flat_rec.get(pipe, {})
        rsd = flat_rec_sd.get(pipe, {})
        rows.append(
            dict(
                group="Flat",
                label=PIPE_LABELS[pipe],
                ba=ba_mean,
                ba_sd=ba_sd,
                ba2=None,
                ba2_sd=None,
                her2=r.get("HER2+", 0),
                her2_sd=rsd.get("HER2+", np.zeros(1)).std(),
                hr=r.get("HR+", 0),
                hr_sd=rsd.get("HR+", np.zeros(1)).std(),
                tn=r.get("Triple Neg", 0),
                tn_sd=rsd.get("Triple Neg", np.zeros(1)).std(),
                feat=feat,
                pipe_key=pipe,
            )
        )

    for pipe in HIER_BASE_PIPES:
        sub = hier_df[hier_df["stage2_pipeline"] == pipe]
        repeat_ba = sub.groupby("repeat")["combined_bal_acc"].mean()
        ba_mean, ba_sd = repeat_ba.mean(), repeat_ba.std()
        repeat_ba2 = sub.groupby("repeat")["stage2_bal_acc"].mean()
        ba2_mean, ba2_sd = repeat_ba2.mean(), repeat_ba2.std()
        feat = sub["stage2_n_features"].median()
        r = hier_rec.get(pipe, {})
        rsd = hier_rec_sd.get(pipe, {})
        rows.append(
            dict(
                group="Hier. (base)",
                label=PIPE_LABELS[pipe],
                ba=ba_mean,
                ba_sd=ba_sd,
                ba2=ba2_mean,
                ba2_sd=ba2_sd,
                her2=r.get("HER2+", 0),
                her2_sd=rsd.get("HER2+", np.zeros(1)).std(),
                hr=r.get("HR+", 0),
                hr_sd=rsd.get("HR+", np.zeros(1)).std(),
                tn=r.get("Triple Neg", 0),
                tn_sd=rsd.get("Triple Neg", np.zeros(1)).std(),
                feat=feat,
                pipe_key=pipe,
            )
        )

    for pipe in HIER_PLAT_PIPES:
        sub = hier_df[hier_df["stage2_pipeline"] == pipe]
        if len(sub) == 0:
            continue
        repeat_ba = sub.groupby("repeat")["combined_bal_acc"].mean()
        ba_mean, ba_sd = repeat_ba.mean(), repeat_ba.std()
        repeat_ba2 = sub.groupby("repeat")["stage2_bal_acc"].mean()
        ba2_mean, ba2_sd = repeat_ba2.mean(), repeat_ba2.std()
        feat = sub["stage2_n_features"].median()
        r = hier_rec.get(pipe, {})
        rsd = hier_rec_sd.get(pipe, {})
        rows.append(
            dict(
                group="Hier. (plateau)",
                label=PIPE_LABELS[pipe],
                ba=ba_mean,
                ba_sd=ba_sd,
                ba2=ba2_mean,
                ba2_sd=ba2_sd,
                her2=r.get("HER2+", 0),
                her2_sd=rsd.get("HER2+", np.zeros(1)).std(),
                hr=r.get("HR+", 0),
                hr_sd=rsd.get("HR+", np.zeros(1)).std(),
                tn=r.get("Triple Neg", 0),
                tn_sd=rsd.get("Triple Neg", np.zeros(1)).std(),
                feat=feat,
                pipe_key=pipe,
            )
        )

    best = {
        "ba": max(r["ba"] for r in rows),
        "ba2": max((r["ba2"] for r in rows if r["ba2"] is not None), default=0),
        "her2": max(r["her2"] for r in rows),
        "hr": max(r["hr"] for r in rows),
        "tn": max(r["tn"] for r in rows),
    }

    def fmt(val, sd, bv, null="--"):
        """Format value as mean +/- SD, bold the mean if best.

        Args:
            val (float or None): Mean value.
            sd (float or None): Standard deviation.
            bv (float): Best value for bolding comparison.
            null (str): String to return if val is None.

        Returns:
            str: Formatted LaTeX string.
        """
        if val is None:
            return null
        mean_str = f"{val:.3f}"
        sd_str = f"{sd:.3f}" if sd is not None else "0.000"
        if abs(val - bv) < 1e-6:
            return f"\\textbf{{{mean_str}}}$\\pm${sd_str}"
        return f"{mean_str}$\\pm${sd_str}"

    def fmt_feat(val, pipe_key):
        """Format the median k column, using -- for Ensemble.

        Args:
            val (float): Median number of features.
            pipe_key (str): Pipeline key.

        Returns:
            str: Formatted string.
        """
        if pipe_key == "nmc_pens_ensemble":
            return "--"
        return f"{val:.0f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Pipeline performance summary. BA is 3-class balanced "
        r"accuracy; BA2 is Stage~2 balanced accuracy (HR+ vs TN). "
        r"BA1\,=\,1.0 for all hierarchical pipelines. "
        r"Values are mean\,$\pm$\,SD of per-repeat means. "
        r"Per-class columns show mean\,$\pm$\,SD of per-repeat recalls. "
        r"Med.\,$k$ is the median number of features selected across outer "
        r"folds. Best mean values per column in bold.}",
        r"\label{tab:results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Experiment & Pipeline & BA & BA2 & HER2+ & HR+ & TN " r"& Med.\,$k$ \\",
        r"\midrule",
    ]

    cur_group = None
    for row in rows:
        if row["group"] != cur_group:
            if cur_group is not None:
                lines.append(r"\midrule")
            cur_group = row["group"]
            grp = row["group"]
        else:
            grp = ""
        lines.append(
            f"  {grp} & {row['label']} & "
            f"{fmt(row['ba'], row['ba_sd'], best['ba'])} & "
            f"{fmt(row['ba2'], row.get('ba2_sd'), best['ba2'])} & "
            f"{fmt(row['her2'], row['her2_sd'], best['her2'])} & "
            f"{fmt(row['hr'], row['hr_sd'], best['hr'])} & "
            f"{fmt(row['tn'], row['tn_sd'], best['tn'])} & "
            f"{fmt_feat(row['feat'], row['pipe_key'])} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    content = "\n".join(lines)
    path = FIGURES_DIR / "table1_results.tex"
    with open(path, "w") as f:
        f.write(content)
    print("  Saved table1_results.tex")
    print("\n" + content)


# =========================================================================
# Main
# =========================================================================


def main():
    """Generate all paper figures and the summary table."""
    print(f"Output: {FIGURES_DIR}\n")

    print("[1/4] Methodology flowchart...")
    generate_fig1()

    print("\n[2/4] Feature selection stability...")
    generate_fig3()

    print("\n[3/4] Results (3-class BA + confusion matrices)...")
    generate_fig2()

    print("\n[4/4] LaTeX summary table...")
    generate_table1()

    print("\nDone. 3 figures + 1 table generated.")


if __name__ == "__main__":
    main()
