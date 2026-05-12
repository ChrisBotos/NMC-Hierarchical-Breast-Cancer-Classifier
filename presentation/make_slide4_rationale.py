"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: make_slide4_rationale.py.
Description:
    Generates a composite multi-panel figure for presentation slide 4,
    arguing why the submitted method (EN+NMC plateau) outperformed
    alternatives. Panel A shows BA2 distributions per pipeline,
    Panel B shows the 2x2 interaction plot, and Panel C presents
    the key rationale text.

Usage:
    python3 presentation/make_slide4_rationale.py

Dependencies:
    Python >= 3.10.
    matplotlib, pandas, numpy.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# Shared colour palette (consistent across all 4 slides).
C = {
    'stage1': '#4DBBD5',
    'stage2': '#F39B7F',
    'highlight': '#FFD700',
    'highlight_bg': '#FFF8DC',
    'border': '#444444',
    'text': '#1A1A1A',
    'HER2': '#E64B35',
    'HR': '#3C5488',
    'TN': '#00A087',
    'nmc': '#3C5488',
    'rf': '#E64B35',
    'en_sel': '#F39B7F',
    'kw_sel': '#4DBBD5',
    'best': '#FFD700',
}

# Pipeline colours (NMC variants in blue tones, RF in red tones).
PIPE_COLORS = {
    'en_nmc_pens': '#FFD700',
    'nmc_pens_ensemble': '#B8860B',
    'kw_nmc_pens': '#4DBBD5',
    'standalone_en_pens': '#91D1C2',
    'nmc_ensemble': '#8491B4',
    'standalone_en': '#B09C85',
    'kw_nmc': '#3C5488',
    'en_nmc': '#7E6148',
    'kw_rf': '#00A087',
    'en_rf': '#E64B35',
}

# Pipeline display names, mapping internal keys to readable labels.
PIPE_DISPLAY = {
    'en_nmc_pens': 'EN+NMC (plateau) *',
    'nmc_pens_ensemble': 'NMC Pens Ensemble',
    'kw_nmc_pens': 'KW+NMC (plateau)',
    'standalone_en_pens': 'EN (plateau)',
    'nmc_ensemble': 'NMC Ensemble',
    'standalone_en': 'Standalone EN',
    'kw_nmc': 'KW+NMC',
    'en_nmc': 'EN+NMC',
    'kw_rf': 'KW+RF',
    'en_rf': 'EN+RF',
}

# Project root and data paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "2026-04-25_final_hierarchical"
    / "nested_cv_analysis"
    / "data"
)


def load_repeat_ba2():
    """Load all fold results and compute repeat-level BA2 per pipeline.

    Returns:
        pd.DataFrame: Columns [pipeline, repeat, stage2_bal_acc] with
            one row per pipeline-repeat combination (200 per pipeline).
    """
    fpath = RESULTS_DIR / "all_fold_results.csv"
    df = pd.read_csv(fpath, usecols=['pipeline', 'repeat', 'stage2_bal_acc'])
    repeat_ba2 = (
        df.groupby(['pipeline', 'repeat'])['stage2_bal_acc']
        .mean()
        .reset_index()
    )
    return repeat_ba2


def load_summary():
    """Load pre-computed summary statistics.

    Returns:
        pd.DataFrame: One row per pipeline with mean_bal_acc, std_bal_acc,
            etc.
    """
    fpath = RESULTS_DIR / "summary_statistics.csv"
    return pd.read_csv(fpath)


def draw_panel_a(ax, repeat_ba2, summary):
    """Draw horizontal box plots with strip points for BA2 per pipeline.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        repeat_ba2 (pd.DataFrame): Repeat-level BA2 values.
        summary (pd.DataFrame): Summary statistics for ordering.
    """
    # Sort pipelines by mean BA2 (best at top).
    order = summary.sort_values('mean_bal_acc', ascending=True)['pipeline'].tolist()

    positions = list(range(len(order)))
    box_data = []
    for pipe in order:
        vals = repeat_ba2.loc[
            repeat_ba2['pipeline'] == pipe, 'stage2_bal_acc'
        ].values
        box_data.append(vals)

    # Draw box plots.
    bp = ax.boxplot(
        box_data,
        positions=positions,
        vert=False,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='#555555', linewidth=1.0),
        capprops=dict(color='#555555', linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    # Colour each box.
    for i, pipe in enumerate(order):
        color = PIPE_COLORS.get(pipe, '#CCCCCC')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_edgecolor('#333333')
        if pipe == 'en_nmc_pens':
            bp['boxes'][i].set_linewidth(2.0)
            bp['boxes'][i].set_edgecolor('#8B6914')

    # Overlay jittered strip points.
    rng = np.random.default_rng(42)
    for i, pipe in enumerate(order):
        vals = box_data[i]
        jitter = rng.normal(0, 0.12, size=len(vals))
        y_pts = i + jitter
        color = PIPE_COLORS.get(pipe, '#CCCCCC')
        alpha = 0.30 if pipe == 'en_nmc_pens' else 0.12
        size = 8 if pipe == 'en_nmc_pens' else 5
        ax.scatter(
            vals, y_pts,
            color=color, alpha=alpha, s=size,
            edgecolors='none', zorder=1, rasterized=True,
        )

    # Vertical reference lines.
    ax.axvline(0.5, color='#999999', ls='--', lw=1.0, zorder=0, alpha=0.6)
    ax.axvline(0.759, color=C['best'], ls='--', lw=1.5, zorder=0, alpha=0.8)

    # Annotate the best-mean line.
    ax.text(
        0.73, len(order) - 0.7, "BA2 = 0.759",
        fontsize=44, color='#8B6914', fontweight='bold',
        ha='center', va='bottom',
    )
    ax.text(
        0.502, -0.3, 'chance',
        fontsize=11, color='#666666', fontstyle='italic',
        ha='left', va='top',
    )

    # Y-axis labels as display names.
    display_labels = [PIPE_DISPLAY.get(p, p) for p in order]
    ax.set_yticks(positions)
    ax.set_yticklabels(display_labels, fontsize=12)

    # Bold the best pipeline label.
    for label in ax.get_yticklabels():
        if '*' in label.get_text():
            label.set_fontweight('bold')
            label.set_color('#8B6914')

    ax.set_xlabel('Stage 2 Balanced Accuracy (BA2)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(0.48, 0.90)
    ax.set_ylim(-0.7, len(order) + 0.5)

    # Light grid on x-axis.
    ax.grid(axis='x', alpha=0.25, ls='-', color='#CCCCCC', zorder=0)
    ax.set_axisbelow(True)

    # Panel label.
    ax.text(
        0.0, 1.03, 'A', transform=ax.transAxes,
        fontsize=20, fontweight='bold', color=C['text'],
        ha='left', va='bottom',
    )


def draw_panel_b(ax, summary):
    """Draw the 2x2 interaction plot (selector x classifier).

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        summary (pd.DataFrame): Summary statistics with mean and std.
    """
    # Extract the 4 base pipelines.
    base_pipes = {
        'kw_nmc': ('KW', 'NMC'),
        'en_nmc': ('EN', 'NMC'),
        'kw_rf': ('KW', 'RF'),
        'en_rf': ('EN', 'RF'),
    }

    summary_dict = summary.set_index('pipeline').to_dict('index')

    selectors = ['KW', 'EN']
    x_pos = [0, 1]

    # NMC line.
    nmc_means = []
    nmc_stds = []
    for sel in selectors:
        key = f"{sel.lower()}_nmc"
        nmc_means.append(summary_dict[key]['mean_bal_acc'])
        nmc_stds.append(summary_dict[key]['std_bal_acc'])

    # RF line.
    rf_means = []
    rf_stds = []
    for sel in selectors:
        key = f"{sel.lower()}_rf"
        rf_means.append(summary_dict[key]['mean_bal_acc'])
        rf_stds.append(summary_dict[key]['std_bal_acc'])

    # Plot lines with error bars.
    ax.errorbar(
        x_pos, nmc_means, yerr=nmc_stds,
        fmt='o-', color=C['nmc'], linewidth=2.5, markersize=10,
        capsize=6, capthick=1.5, elinewidth=1.5,
        label='NMC', zorder=5,
    )
    ax.errorbar(
        x_pos, rf_means, yerr=rf_stds,
        fmt='s-', color=C['rf'], linewidth=2.5, markersize=10,
        capsize=6, capthick=1.5, elinewidth=1.5,
        label='RF', zorder=5,
    )

    # Standalone EN as a single green triangle at the EN x-position.
    if 'standalone_en' in summary_dict:
        se_mean = summary_dict['standalone_en']['mean_bal_acc']
        se_std = summary_dict['standalone_en']['std_bal_acc']
        ax.errorbar(
            [1], [se_mean], yerr=[se_std],
            fmt='^', color=C['TN'], markersize=11, markeredgecolor='#006650',
            capsize=5, capthick=1.2, elinewidth=1.2,
            label='Standalone EN', zorder=6,
        )
        ax.annotate(
            f'{se_mean:.3f}',
            (1, se_mean),
            textcoords='offset points', xytext=(14, 10),
            fontsize=10, color=C['TN'], fontweight='bold',
        )

    # Add value annotations next to each point.
    for i, sel in enumerate(selectors):
        # At EN position (i=1), push NMC annotation down to avoid
        # overlap with the standalone EN triangle annotation above.
        nmc_dy = 8 if i == 0 else -12
        ax.annotate(
            f'{nmc_means[i]:.3f}',
            (x_pos[i], nmc_means[i]),
            textcoords='offset points', xytext=(12, nmc_dy),
            fontsize=11, color=C['nmc'], fontweight='bold',
        )
        ax.annotate(
            f'{rf_means[i]:.3f}',
            (x_pos[i], rf_means[i]),
            textcoords='offset points', xytext=(12, -14),
            fontsize=11, color=C['rf'], fontweight='bold',
        )

    # Axis formatting.
    ax.set_xticks(x_pos)
    ax.set_xticklabels(selectors, fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Selector', fontsize=14)
    ax.set_ylabel('Mean BA2', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(-0.4, 1.4)

    # Extend y-limits to accommodate error bars and annotations.
    all_vals = nmc_means + rf_means
    all_stds = nmc_stds + rf_stds
    y_min = min(v - s for v, s in zip(all_vals, all_stds)) - 0.02
    y_max = max(v + s for v, s in zip(all_vals, all_stds)) + 0.03
    ax.set_ylim(y_min, y_max)

    # Legend.
    ax.legend(
        fontsize=12, loc='lower left',
        framealpha=0.9, edgecolor='#CCCCCC',
    )

    # Annotation of significance.
    ax.text(
        0.5, 0.02, r'NMC > RF  ($p < 10^{-16}$)',
        transform=ax.transAxes, fontsize=11,
        ha='center', va='bottom',
        color=C['text'], fontstyle='italic',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#F0F0F0',
            edgecolor='#CCCCCC',
            alpha=0.9,
        ),
    )

    ax.grid(axis='y', alpha=0.25, ls='-', color='#CCCCCC', zorder=0)
    ax.set_axisbelow(True)

    # Panel label.
    ax.text(
        -0.12, 1.06, 'B', transform=ax.transAxes,
        fontsize=20, fontweight='bold', color=C['text'],
        ha='left', va='bottom',
    )


def draw_panel_c(ax):
    """Draw the key rationale text panel with clean formatting.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
    """
    ax.axis('off')

    # Background box.
    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.03',
        facecolor='#F8F8F4',
        edgecolor='#CCCCCC',
        linewidth=1.2,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(bg)

    # Section header.
    ax.text(
        0.50, 0.95, 'Why EN+NMC (plateau)?',
        transform=ax.transAxes, fontsize=19, fontweight='bold',
        ha='center', va='top', color=C['text'], zorder=5,
    )

    # Bullet points with compact layout.
    bullets = [
        (
            '1.',
            'Simple classifier wins.',
            'NMC > RF across all selectors (p < 10$^{-16}$).\n'
            'Fewer parameters resist overfitting (n = 72).'
        ),
        (
            '2.',
            'Plateau exposes true EN+NMC quality.',
            'Base EN+NMC (0.729) < standalone EN (0.738): inner\n'
            'CV noisy at small n. Plateau pools top configs $\\rightarrow$ 0.759.'
        ),
        (
            '3.',
            'EN+NMC(P) outperforms standalone EN(P).',
            'Best model overall (0.759 vs 0.741): separating\n'
            'selection from classification yields higher BA2.'
        ),
        (
            '4.',
            'Aligns with Wessels et al. (2005).',
            'Simple classifiers with univariate filtering match\n'
            'or exceed complex pipelines on small-n discrete data.'
        ),
    ]

    y_start = 0.82
    y_step = 0.22

    for i, (num, title, body) in enumerate(bullets):
        y = y_start - i * y_step

        # Number in gold.
        ax.text(
            0.07, y, num,
            transform=ax.transAxes, fontsize=15, fontweight='bold',
            ha='center', va='top', color='#5A4010', zorder=5,
        )

        # Bold title.
        ax.text(
            0.12, y, title,
            transform=ax.transAxes, fontsize=14.5, fontweight='bold',
            ha='left', va='top', color=C['text'], zorder=5,
        )

        # Body text.
        ax.text(
            0.12, y - 0.075, body,
            transform=ax.transAxes, fontsize=12,
            ha='left', va='top', color='#2A2A2A', zorder=5,
            linespacing=1.35,
        )

    # Panel label.
    ax.text(
        -0.12, 1.06, 'C', transform=ax.transAxes,
        fontsize=20, fontweight='bold', color=C['text'],
        ha='left', va='bottom',
    )


def main():
    """Generate the slide 4 rationale composite figure."""
    # Load data.
    repeat_ba2 = load_repeat_ba2()
    summary = load_summary()

    # Create figure with gridspec layout.
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor('white')

    # Panel A takes left ~48%, panels B and C share the right ~52%.
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.0, 0.9],
        height_ratios=[0.75, 1.30],
        wspace=0.30,
        hspace=0.30,
        left=0.08, right=0.97,
        top=0.95, bottom=0.05,
    )

    # Panel A spans both rows on the left.
    ax_a = fig.add_subplot(gs[:, 0])
    draw_panel_a(ax_a, repeat_ba2, summary)

    # Panel B is top-right.
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b, summary)

    # Panel C is bottom-right.
    ax_c = fig.add_subplot(gs[1, 1])
    draw_panel_c(ax_c)

    # Save output.
    out_path = Path(__file__).resolve().parent / "slide_4_rationale.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
