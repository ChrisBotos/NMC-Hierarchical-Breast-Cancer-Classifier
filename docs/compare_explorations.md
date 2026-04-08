# Comparison of Exploration Runs

Figure produced by `code/compare_explorations.py`. The script loads pre-computed results from any two `data_exploration_phase` output directories and generates a side-by-side comparison figure.

## Usage

```bash
# Default: raw vs merged
python3 code/compare_explorations.py \
    --label-a "Raw (2834 regions)" --label-b "Merged (273 segments)"

# Any two exploration runs
python3 code/compare_explorations.py \
    --dir-a results/data/data_exploration_phase \
    --dir-b results/data/data_exploration_phase_merged \
    --label-a "Raw" --label-b "Merged" --tag raw_vs_merged
```

Required files in each directory: `pca_coordinates.tsv`, `pca_variance_explained.tsv`, `tsne_perp30.tsv`, `kruskal_wallis_per_region.tsv`.

---

## Figure 01 - Comparison (`01_comparison.png`)

Six-panel figure (2 rows x 3 columns) comparing two exploration runs.

### Panel (a) - PCA: Dataset A
- **Axes:** PC1 vs PC2, colored by subtype.

### Panel (b) - PCA: Dataset B
- **Axes:** PC1 vs PC2, colored by subtype.

### Panel (c) - Cumulative Variance
- **Axes:** Number of PCs vs cumulative variance explained (%).
- Blue line = dataset A, green line = dataset B.

### Panel (d) - t-SNE: Dataset A
- t-SNE embedding (perplexity=30).

### Panel (e) - t-SNE: Dataset B
- t-SNE embedding (perplexity=30).

### Panel (f) - Summary Metrics
- Grouped bars comparing two metrics:
  - **Cumulative variance (PC1-10, %)**: How much signal is captured in the first 10 components.
  - **Bonferroni-significant (%)**: Fraction of features passing Bonferroni-corrected KW test.
- Text annotation below shows raw feature counts and Bonferroni counts.

### Default comparison (raw vs merged)
- PCA structure preserved: PC1 explains 16.1% (raw) vs 15.4% (merged).
- t-SNE cluster structure visually unchanged.
- Cumulative variance: 47.2% (raw) vs 46.0% (merged) in first 10 PCs.
- Bonferroni-significant: 6/2834 (0.2%) raw vs 3/273 (1.1%) merged.
