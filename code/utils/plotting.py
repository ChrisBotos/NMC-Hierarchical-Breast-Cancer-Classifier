"""Matplotlib style configuration."""

import matplotlib.pyplot as plt


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
