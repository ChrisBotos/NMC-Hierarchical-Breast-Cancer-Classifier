"""Matplotlib style configuration and reusable plotting helpers."""

import matplotlib.pyplot as plt

from utils.statistics import format_p_value


def apply_plot_style(scale="normal"):
    """Apply project-wide matplotlib rcParams.

    Args:
        scale (str): "normal" for standard single-panel figures,
            "compact" for multi-panel comparison figures with smaller fonts.
    """
    if scale == "compact":
        params = {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.family": "sans-serif",
        }
    else:
        params = {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
        }
    plt.rcParams.update(params)


def annotate_heatmap(ax, data, fmt=".2f", raw_counts=None, threshold=0.6, fontsize=9):
    """Annotate cells of an imshow heatmap with text values.

    Args:
        ax (matplotlib.axes.Axes): Axes containing the imshow plot.
        data (np.ndarray): 2-D array of values to display.
        fmt (str): Format string for each value (e.g. ".2f").
        raw_counts (np.ndarray or None): If provided, displayed in
            parentheses below the formatted value (e.g. for confusion
            matrices showing both normalised and raw counts).
        threshold (float): Values above this use white text for contrast.
        fontsize (int): Font size for the annotations.
    """
    nrows, ncols = data.shape
    for i in range(nrows):
        for j in range(ncols):
            text_color = "white" if data[i, j] > threshold else "black"
            if raw_counts is not None:
                label = f"{data[i, j]:{fmt}}\n({raw_counts[i, j]})"
            else:
                label = f"{data[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def draw_significance_brackets(ax, pairwise_df, group_positions, data_by_group):
    """Draw significance brackets with p-values above a violin/box plot.

    Sorts brackets by span width so narrower brackets are drawn lower.
    Uses U-shaped brackets with formatted p-values centred above.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        pairwise_df (pd.DataFrame): Pairwise test results with columns:
            pipeline_a, pipeline_b, p_corrected, significant.
        group_positions (list[str]): Ordered group names matching the
            x-axis positions (0, 1, 2, ...).
        data_by_group (list[np.ndarray]): Data arrays for each group,
            used to compute bracket y-position.
    """
    if pairwise_df is None or pairwise_df.empty:
        return

    y_max = max(v.max() for v in data_by_group if len(v) > 0)
    y_range = y_max - min(v.min() for v in data_by_group if len(v) > 0)
    bracket_height = y_range * 0.05
    y_offset = y_max + y_range * 0.08

    # Sort by span width so narrower brackets are drawn lower.
    pw_sorted = pairwise_df.copy()
    pw_sorted["span"] = pw_sorted.apply(
        lambda r: abs(
            group_positions.index(r["pipeline_a"])
            - group_positions.index(r["pipeline_b"])
        ),
        axis=1,
    )
    pw_sorted = pw_sorted.sort_values("span").reset_index(drop=True)

    for i, row in pw_sorted.iterrows():
        x1 = group_positions.index(row["pipeline_a"])
        x2 = group_positions.index(row["pipeline_b"])
        y_bar = y_offset + i * bracket_height * 2.5

        is_sig = row["significant"]
        bracket_color = "black" if is_sig else "grey"

        ax.plot(
            [x1, x1, x2, x2],
            [y_bar, y_bar + bracket_height, y_bar + bracket_height, y_bar],
            color=bracket_color,
            linewidth=0.8,
        )
        p_text = format_p_value(row["p_corrected"])
        if is_sig:
            p_text += " *"
        ax.text(
            (x1 + x2) / 2,
            y_bar + bracket_height,
            p_text,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=bracket_color,
        )
