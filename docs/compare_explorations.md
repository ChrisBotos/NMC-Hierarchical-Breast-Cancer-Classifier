# Comparison: Raw vs Merged Data Exploration

Single figure produced by `code/compare_explorations.py`.

---

## Figure 01 — Raw vs Merged Comparison (`01_raw_vs_merged_comparison.png`)

Six-panel figure (2 rows x 3 columns) comparing exploration analyses on raw data (2834 regions) vs merged data (273 consensus segments).

### Panel (a) — PCA: Raw Data
- **Axes:** PC1 vs PC2 from PCA on all 2834 raw features.
- PC1 explains 16.1% of variance, PC2 explains 6.2%.

### Panel (b) — PCA: Merged Data
- **Axes:** PC1 vs PC2 from PCA on 273 merged segments.
- PC1 explains 15.4% of variance, PC2 explains 5.1%.
- HER2+ separation on PC1 is preserved; HR+ and Triple Neg still overlap.

### Panel (c) — Cumulative Variance
- **Axes:** Number of PCs vs cumulative variance explained (%).
- Blue line = raw, green line = merged.
- After merging, 10 PCs capture 46.0% of total variance (vs 22.3% for raw). Removing redundant features concentrates variance into fewer components.

### Panel (d) — t-SNE: Raw Data
- t-SNE embedding (perplexity=30) on raw features.

### Panel (e) — t-SNE: Merged Data
- t-SNE embedding (perplexity=30) on merged segments.
- Cluster structure is preserved: HER2+ separates, HR+ and Triple Neg remain intermixed.

### Panel (f) — Summary Metrics
- Grouped bars comparing raw vs merged on three metrics:
  - **Silhouette score** (k=3 Ward): 0.092 (raw) vs 0.025 (merged). Lower after merging -- expected because the median-based consensus blurs discrete CN values. Merging is designed to help supervised classification, not unsupervised clustering.
  - **Cumulative variance (PC1-10)**: 22.3% (raw) vs 46.0% (merged). Higher after merging -- signal is concentrated after redundancy removal.
  - **Median adjacent correlation**: 0.925 (raw) vs 0.590 (merged). Successfully reduced spatial redundancy.
- Annotation: Bonferroni-significant features: raw = 6/2834, merged = 3/273.
