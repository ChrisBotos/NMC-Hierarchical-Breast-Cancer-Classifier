"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: generate_paper_figures.py.
Description:
    Generates the final publication-quality figures and LaTeX table for the
    research paper. Produces exactly 3 figures + 1 table:
      Figure 1: Methodology flowchart.
      Figure 2: Multi-panel results (3-class BA violins + confusion matrices).
      Figure 3: Feature selection stability curve with top regions.
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Resolve paths from script location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "latex_draft_report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths.
FLAT_DATA = (PROJECT_ROOT / "results" /
             "2026-04-22_server_run_flat_nested_CV_2x2" / "nested_cv_2x2" / "data")
HIER_DATA = (PROJECT_ROOT / "results" /
             "2026-04-24_server_run_v2" / "hierarchical_nested_cv" / "data")

# Global style.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette.
BLUE = '#4477AA'
ORANGE = '#EE6677'
GREEN = '#228833'

# Pipeline configuration (kw_nmc_kens excluded from all outputs).
FLAT_PIPES = ['kw_nmc', 'kw_rf', 'en_nmc', 'en_rf']
HIER_BASE_PIPES = ['kw_nmc', 'kw_rf', 'en_nmc', 'en_rf']
HIER_PLAT_PIPES = ['kw_nmc_pens', 'en_nmc_pens', 'nmc_ensemble']

PIPE_LABELS = {
    'kw_nmc': 'KW+NMC', 'kw_rf': 'KW+RF',
    'en_nmc': 'EN+NMC', 'en_rf': 'EN+RF',
    'kw_nmc_pens': 'KW+NMC(P)', 'en_nmc_pens': 'EN+NMC(P)',
    'nmc_ensemble': 'Ensemble',
}
PIPE_COLORS = {
    'kw_nmc': BLUE, 'kw_rf': ORANGE,
    'en_nmc': BLUE, 'en_rf': ORANGE,
    'kw_nmc_pens': BLUE, 'en_nmc_pens': BLUE,
    'nmc_ensemble': BLUE,
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
                frames.append(df[['pipeline', 'repeat', 'outer_fold',
                                  'balanced_accuracy', 'n_features_selected',
                                  'y_true', 'y_pred']])
    return pd.concat(frames, ignore_index=True)


def load_hierarchical_results():
    """Load hierarchical nested CV fold results (base + plateau + ensemble).

    Returns:
        pd.DataFrame: Combined fold results for hierarchical pipelines.
    """
    all_pipes = HIER_BASE_PIPES + HIER_PLAT_PIPES
    frames = []
    for pipe in all_pipes:
        for r in range(1, 51):
            fpath = HIER_DATA / f"fold_results_{pipe}_r{r}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                frames.append(df[['stage2_pipeline', 'repeat', 'outer_fold',
                                  'stage1_bal_acc', 'stage2_bal_acc',
                                  'combined_bal_acc', 'stage2_n_features',
                                  'y_true', 'y_pred']])
    return pd.concat(frames, ignore_index=True)


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
    return {pipe: grp.groupby('repeat')[ba_col].mean().values
            for pipe, grp in df.groupby(pipe_col)}


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
            all_true.extend(str(row['y_true']).split(','))
            all_pred.extend(str(row['y_pred']).split(','))
        recalls = {}
        for cls in ['HER2+', 'HR+', 'Triple Neg']:
            correct = sum(1 for t, p in zip(all_true, all_pred)
                         if t.strip() == cls and p.strip() == cls)
            total = sum(1 for t in all_true if t.strip() == cls)
            recalls[cls] = correct / total if total > 0 else 0.0
        results[pipe] = recalls
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
    parts = ax.violinplot(data_list, positions=positions, widths=width,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
        body.set_edgecolor(colors[i])
        body.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, (data, px) in enumerate(zip(data_list, positions)):
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(px + jitter, data, s=5, alpha=0.15,
                   color=colors[i], edgecolors='none', zorder=4)
        med = np.median(data)
        ax.plot([px - 0.2, px + 0.2], [med, med],
                color=colors[i], linewidth=2.0, solid_capstyle='round', zorder=5)


def shorten_region(name):
    """Shorten a genomic region name for display.

    Args:
        name (str): Full region name like chr17_35076296_35282086.

    Returns:
        str: Shortened name like chr17:35.1-35.3M.
    """
    parts = name.split('_')
    if len(parts) >= 3:
        chrom = parts[0]
        start_mb = int(parts[1]) / 1e6
        end_mb = int(parts[2]) / 1e6
        return f'{chrom}:{start_mb:.1f}-{end_mb:.1f}M'
    return name


# =========================================================================
# FIGURE 1 - Methodology flowchart
# =========================================================================

def generate_fig1():
    """Generate the methodology flowchart (Figure 1, full-width)."""
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 11.5)
    ax.axis('off')

    gf, ge = '#E8E8E8', '#666666'   # Grey (shared steps).
    bf, be = '#D4E6F1', '#2874A6'   # Blue (flat path).
    grf, gre = '#D5F5E3', '#1E8449' # Green (hierarchical path).
    goldf, golde = '#FEF9E7', '#B7950B'  # Gold (final model).
    LS = 0.30  # Line spacing in data coords.

    def box(x, y, w, h, fc, ec, lines, fs=7, bold1=True):
        """Draw rounded-rect box with multi-line text."""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.12',
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
        cx, cy = x + w / 2, y + h / 2
        top = cy + (len(lines) - 1) * LS / 2
        for i, ln in enumerate(lines):
            fw = 'bold' if (i == 0 and bold1) else 'normal'
            ax.text(cx, top - i * LS, ln, ha='center', va='center',
                    fontsize=fs, fontweight=fw, zorder=3, fontfamily='serif')

    def arrow(x1, y1, x2, y2, cs='arc3,rad=0'):
        """Draw arrow."""
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='->', mutation_scale=12,
            connectionstyle=cs, color='#333333', linewidth=1.0, zorder=1))

    bw, bh, bht = 3.4, 0.85, 1.15
    cl, cc, cr = 3.3, 7.0, 10.7
    yd, ym, yc = 10.0, 8.6, 7.2
    yb, ys, yp, yfin = 5.5, 3.8, 2.3, 0.7

    # Shared steps.
    box(cc-bw/2, yd, bw, bh, gf, ge,
        ['DATA', '100 samples, 2834 regions, 3 classes'])
    box(cc-bw/2, ym, bw, bh, gf, ge,
        ['REGION MERGING', 'Pearson r > 0.8 --> 273 segments'])
    box(cc-bw/2, yc, bw, bh, gf, ge,
        ['OUTER CV', 'Repeated stratified 5-fold (R repeats)'])
    arrow(cc, yd, cc, ym+bh)
    arrow(cc, ym, cc, yc+bh)

    # Flat path.
    box(cl-bw/2, yb, bw, bht, bf, be,
        ['FLAT 3-CLASS', '[KW / EN] x [NMC / RF]', 'Inner 5-fold CV tuning'])
    arrow(cc-bw/2, yc+bh*0.4, cl+bw/2, yb+bht, cs='arc3,rad=0.15')
    ax.text(cl, yb-0.2, '4 pipelines, 50 repeats x 5 folds = 250',
            ha='center', va='top', fontsize=6, style='italic', color='#555555')

    # Hierarchical Stage 1.
    box(cr-bw/2, yb, bw, bh, grf, gre,
        ['STAGE 1: HER2+ vs rest', 'Fixed KW+RF, k=5, BA1=1.0'])
    arrow(cc+bw/2, yc+bh*0.4, cr-bw/2, yb+bh, cs='arc3,rad=-0.15')

    # Stage 2.
    box(cr-bw/2, ys, bw, bht, grf, gre,
        ['STAGE 2: HR+ vs TN', '[KW / EN] x [NMC / RF]', 'Inner 5-fold CV tuning'])
    arrow(cr, yb, cr, ys+bht)
    ax.text(cr+bw/2+0.15, (yb+ys+bht)/2, 'non-HER2+\nsamples',
            ha='left', va='center', fontsize=5.5, style='italic', color='#555555')
    ax.text(cr, ys-0.2, '4 pipelines, 200 repeats x 5 folds = 1000',
            ha='center', va='top', fontsize=6, style='italic', color='#555555')

    # Plateau.
    box(cr-bw/2, yp, bw, bh, grf, gre,
        ['PLATEAU ENSEMBLING', 'Top configs by inner score'])
    arrow(cr, ys, cr, yp+bh)

    # Final model.
    box(cc-bw/2, yfin, bw, bh, goldf, golde,
        ['FINAL MODEL', 'EN+NMC plateau, all 100 samples'])
    arrow(cl, yb, cc-bw/2, yfin+bh*0.5, cs='arc3,rad=0.2')
    arrow(cr, yp, cc+bw/2, yfin+bh*0.5, cs='arc3,rad=-0.2')
    arrow(cc, yfin, cc, yfin-0.25)
    ax.text(cc, yfin-0.35, 'Predictions (57 validation samples)',
            ha='center', va='top', fontsize=6.5, style='italic', color='#333333')

    # Group labels.
    ax.text(cl, yb+bht+0.25, 'Flat experiment',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=be)
    ax.text(cr, yb+bh+0.25, 'Hierarchical experiment',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=gre)

    plt.tight_layout(pad=0.3)
    fig.savefig(FIGURES_DIR / 'fig1_workflow.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / 'fig1_workflow.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved fig1_workflow")


# =========================================================================
# FIGURE 2 - Multi-panel: (A) 3-class BA violins, (B) confusion matrices
# =========================================================================

def generate_fig2():
    """Generate the main results figure (Figure 2, full-width, stacked panels).

    Panel A (top, full width): 3-class BA violin comparison (flat BA and
    hierarchical combined BA on the same y-axis).
    Panel B (bottom, full width): Confusion matrices for best flat (KW+RF)
    and best hierarchical (EN+NMC(P)) pipelines, side by side with colorbar.
    """
    flat_df = load_flat_results()
    hier_df = load_hierarchical_results()

    # ---- Compute data for both panels ----
    flat_means = per_repeat_means(flat_df, 'balanced_accuracy', 'pipeline')
    hier_means = per_repeat_means(hier_df, 'combined_bal_acc', 'stage2_pipeline')

    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=0.55)

    # ---- Panel A: 3-class BA violins (top row, full width) ----
    ax_a = fig.add_subplot(gs[0])

    # Three groups: flat (4), hierarchical base (4), hierarchical plateau (3).
    pos_f = [1, 2, 3, 4]
    pos_hb = [6, 7, 8, 9]
    pos_hp = [11, 12, 13]

    data_all, pos_all, col_all = [], [], []
    for pipes, positions in [(FLAT_PIPES, pos_f),
                             (HIER_BASE_PIPES, pos_hb),
                             (HIER_PLAT_PIPES, pos_hp)]:
        source = flat_means if pipes is FLAT_PIPES else hier_means
        for pipe, px in zip(pipes, positions):
            if pipe in source:
                data_all.append(source[pipe])
                pos_all.append(px)
                col_all.append(PIPE_COLORS[pipe])

    draw_violin(ax_a, data_all, pos_all, col_all)

    # Group separators.
    ax_a.axvline(x=5, color='#CCCCCC', linestyle=':', linewidth=0.7, zorder=0)
    ax_a.axvline(x=10, color='#CCCCCC', linestyle=':', linewidth=0.7, zorder=0)

    lbl_all = []
    for pipes in [FLAT_PIPES, HIER_BASE_PIPES, HIER_PLAT_PIPES]:
        for p in pipes:
            if p in (flat_means if pipes is FLAT_PIPES else hier_means):
                lbl_all.append(PIPE_LABELS[p])
    ax_a.set_xticks(pos_all)
    ax_a.set_xticklabels(lbl_all, rotation=40, ha='right')
    ax_a.set_ylabel('3-class balanced accuracy')

    # Group annotations at top.
    ylim = ax_a.get_ylim()
    top_y = ylim[1] + 0.005
    ax_a.text(2.5, top_y, 'Flat', ha='center', va='bottom',
              fontsize=8, fontweight='bold', color='#444444')
    ax_a.text(7.5, top_y, 'Hierarchical\n(base)', ha='center', va='bottom',
              fontsize=8, fontweight='bold', color='#444444')
    ax_a.text(12, top_y, 'Hierarchical\n(plateau)', ha='center', va='bottom',
              fontsize=8, fontweight='bold', color='#444444')

    nmc_p = mpatches.Patch(color=BLUE, alpha=0.5, label='NMC-based')
    rf_p = mpatches.Patch(color=ORANGE, alpha=0.5, label='RF-based')
    ax_a.legend(handles=[nmc_p, rf_p], loc='lower left', framealpha=0.9,
                fontsize=7)
    ax_a.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.text(-0.02, 1.10, '(A)', transform=ax_a.transAxes,
              fontsize=11, fontweight='bold', va='top')

    # ---- Panel B: Confusion matrices (bottom row, full width) ----
    gs_bottom = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.4)
    ax_cm1 = fig.add_subplot(gs_bottom[0])
    ax_cm2 = fig.add_subplot(gs_bottom[1])
    ax_cb = fig.add_subplot(gs_bottom[2])

    classes = ['HER2+', 'HR+', 'Triple Neg']
    short_cls = ['HER2+', 'HR+', 'TN']

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
            yt = [t.strip() for t in str(row['y_true']).split(',')]
            yp = [p.strip() for p in str(row['y_pred']).split(',')]
            for t, p in zip(yt, yp):
                if t in classes and p in classes:
                    cm[classes.index(t), classes.index(p)] += 1
        rsums = cm.sum(axis=1, keepdims=True)
        rsums[rsums == 0] = 1
        return cm / rsums

    cm_flat = build_cm(flat_df, 'pipeline', 'kw_rf')
    cm_hier = build_cm(hier_df, 'stage2_pipeline', 'en_nmc_pens')

    im = None
    for ax, cm, title in [(ax_cm1, cm_flat, 'Flat: KW+RF'),
                           (ax_cm2, cm_hier, 'Hierarchical: EN+NMC(P)')]:
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(short_cls, fontsize=9)
        ax.set_yticklabels(short_cls, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        for i in range(3):
            for j in range(3):
                v = cm[i, j]
                c = 'white' if v > 0.5 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=11, color=c, fontweight='bold')

    # Shared colorbar.
    fig.colorbar(im, cax=ax_cb, label='Recall (row-normalized)')

    ax_cm1.text(-0.08, 1.12, '(B)', transform=ax_cm1.transAxes,
                fontsize=11, fontweight='bold', va='top')

    fig.savefig(FIGURES_DIR / 'fig2_results.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / 'fig2_results.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved fig2_results")


# =========================================================================
# FIGURE 3 - Feature selection stability
# =========================================================================

def generate_fig3():
    """Generate feature selection stability figure (Figure 3, single-column).

    Left: frequency drop-off curve across all features.
    Right: top 10 most frequently selected regions.
    Data from EN+NMC(P) pipeline (best hierarchical).
    """
    # Load Stage 2 features for en_nmc_pens.
    feat_counter = Counter()
    frames = []
    for r in range(1, 51):
        fpath = HIER_DATA / f"fold_results_en_nmc_pens_r{r}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, usecols=['stage2_features'])
            frames.append(df)
    feat_df = pd.concat(frames, ignore_index=True)
    total_folds = len(feat_df)

    for _, row in feat_df.iterrows():
        s = str(row['stage2_features'])
        if s and s != 'nan':
            feat_counter.update(f.strip() for f in s.split(',') if f.strip())

    if not feat_counter:
        print("  SKIP: no features found")
        return

    all_feats = feat_counter.most_common()
    all_freqs = [c / total_folds for _, c in all_feats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                    gridspec_kw={'width_ratios': [1.8, 1]})

    # Left: drop-off curve.
    ax1.plot(range(len(all_freqs)), all_freqs, color=BLUE, linewidth=1.2)
    ax1.fill_between(range(len(all_freqs)), all_freqs, alpha=0.15, color=BLUE)
    ax1.set_xlabel('Features (ranked by selection frequency)')
    ax1.set_ylabel('Selection frequency\n(fraction of folds)')
    ax1.set_xlim(0, len(all_freqs))
    ax1.set_ylim(0, 1.05)
    n_50 = sum(1 for f in all_freqs if f >= 0.5)
    ax1.axhline(y=0.5, color='#999999', linestyle='--', linewidth=0.7, zorder=0)
    ax1.text(len(all_freqs) * 0.95, 0.52,
             f'{n_50} regions in >50% of folds',
             ha='right', va='bottom', fontsize=6, color='#666666')
    ax1.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.text(-0.02, 1.06, '(A)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top')

    # Right: top 10 regions as horizontal bars.
    top_n = 10
    top = feat_counter.most_common(top_n)
    names = [shorten_region(f[0]) for f in top]
    freqs = [f[1] / total_folds for f in top]

    y_pos = np.arange(top_n)
    ax2.barh(y_pos, freqs, color=BLUE, alpha=0.7, edgecolor=BLUE, linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=5.5)
    ax2.set_xlabel('Frequency')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.text(-0.02, 1.06, '(B)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top')

    plt.tight_layout(pad=1.0)
    fig.savefig(FIGURES_DIR / 'fig3_features.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / 'fig3_features.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved fig3_features")


# =========================================================================
# TABLE 1 - Summary statistics
# =========================================================================

def generate_table1():
    """Generate LaTeX summary table with pipeline performance metrics.

    Uses consistent terminology: BA (3-class), BA2 (Stage 2), BA1 = 1.0.
    """
    flat_df = load_flat_results()
    hier_df = load_hierarchical_results()

    flat_rec = per_class_recalls(flat_df, 'pipeline')
    hier_rec = per_class_recalls(hier_df, 'stage2_pipeline')

    rows = []
    for pipe in FLAT_PIPES:
        sub = flat_df[flat_df['pipeline'] == pipe]
        ba = sub.groupby('repeat')['balanced_accuracy'].mean().mean()
        feat = sub['n_features_selected'].median()
        r = flat_rec.get(pipe, {})
        rows.append(dict(group='Flat', label=PIPE_LABELS[pipe], ba=ba, ba2=None,
                         her2=r.get('HER2+', 0), hr=r.get('HR+', 0),
                         tn=r.get('Triple Neg', 0), feat=feat))

    for pipe in HIER_BASE_PIPES:
        sub = hier_df[hier_df['stage2_pipeline'] == pipe]
        ba = sub.groupby('repeat')['combined_bal_acc'].mean().mean()
        ba2 = sub.groupby('repeat')['stage2_bal_acc'].mean().mean()
        feat = sub['stage2_n_features'].median()
        r = hier_rec.get(pipe, {})
        rows.append(dict(group='Hier. (base)', label=PIPE_LABELS[pipe],
                         ba=ba, ba2=ba2,
                         her2=r.get('HER2+', 0), hr=r.get('HR+', 0),
                         tn=r.get('Triple Neg', 0), feat=feat))

    for pipe in HIER_PLAT_PIPES:
        sub = hier_df[hier_df['stage2_pipeline'] == pipe]
        if len(sub) == 0:
            continue
        ba = sub.groupby('repeat')['combined_bal_acc'].mean().mean()
        ba2 = sub.groupby('repeat')['stage2_bal_acc'].mean().mean()
        feat = sub['stage2_n_features'].median()
        r = hier_rec.get(pipe, {})
        rows.append(dict(group='Hier. (plateau)', label=PIPE_LABELS[pipe],
                         ba=ba, ba2=ba2,
                         her2=r.get('HER2+', 0), hr=r.get('HR+', 0),
                         tn=r.get('Triple Neg', 0), feat=feat))

    best = {
        'ba': max(r['ba'] for r in rows),
        'ba2': max((r['ba2'] for r in rows if r['ba2'] is not None), default=0),
        'her2': max(r['her2'] for r in rows),
        'hr': max(r['hr'] for r in rows),
        'tn': max(r['tn'] for r in rows),
    }

    def fmt(val, bv, null='--'):
        """Format value, bold if best."""
        if val is None:
            return null
        s = f'{val:.3f}'
        return f'\\textbf{{{s}}}' if abs(val - bv) < 1e-6 else s

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pipeline performance summary. BA is 3-class balanced '
        r'accuracy; BA2 is Stage~2 balanced accuracy (HR+ vs TN). '
        r'BA1\,=\,1.0 for all hierarchical pipelines. Per-class recalls '
        r'pooled over all outer folds. Best values per column in bold.}',
        r'\label{tab:results}',
        r'\small',
        r'\begin{tabular}{llcccccc}',
        r'\toprule',
        r'Experiment & Pipeline & BA & BA2 & HER2+ & HR+ & TN & Feat. \\',
        r'\midrule',
    ]

    cur_group = None
    for row in rows:
        if row['group'] != cur_group:
            if cur_group is not None:
                lines.append(r'\midrule')
            cur_group = row['group']
            grp = row['group']
        else:
            grp = ''
        lines.append(
            f"  {grp} & {row['label']} & "
            f"{fmt(row['ba'], best['ba'])} & "
            f"{fmt(row['ba2'], best['ba2'])} & "
            f"{fmt(row['her2'], best['her2'])} & "
            f"{fmt(row['hr'], best['hr'])} & "
            f"{fmt(row['tn'], best['tn'])} & "
            f"{row['feat']:.0f} \\\\")

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    content = '\n'.join(lines)
    path = FIGURES_DIR / 'table1_results.tex'
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved table1_results.tex")
    print("\n" + content)


# =========================================================================
# Main
# =========================================================================

def main():
    """Generate all paper figures and the summary table."""
    print(f"Output: {FIGURES_DIR}\n")

    print("[1/4] Methodology flowchart...")
    generate_fig1()

    print("\n[2/4] Results (3-class BA + confusion matrices)...")
    generate_fig2()

    print("\n[3/4] Feature selection stability...")
    generate_fig3()

    print("\n[4/4] LaTeX summary table...")
    generate_table1()

    print("\nDone. 3 figures + 1 table generated.")


if __name__ == '__main__':
    main()
