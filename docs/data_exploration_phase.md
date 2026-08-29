# Phase 0: Data Exploration

Summary of all figures produced by `code/data_exploration_phase.py`.

---

## Figure 01 - Class Distribution (`01_class_distribution.png`)

- **X-axis:** Breast cancer molecular subtype (HER2+, HR+, Triple Neg).
- **Y-axis:** Number of samples in each class.
- **Conclusion:** Classes are roughly balanced (32 / 36 / 32). No severe imbalance that would require resampling or class weighting.

## Figure 02 - Genome-wide Aberration Frequency (`02_genome_wide_frequency.png`)

- **X-axis:** Genomic position across all chromosomes (1-X).
- **Y-axis:** Fraction of samples with gain/amplification (upward, red) or loss (downward, blue), per subtype.
- **Conclusion:** Subtypes have distinct copy-number landscapes. HER2+ shows a sharp amplification peak on chr17q (the ERBB2 locus). Triple Negative has broader gains on 1q and losses on 5q. HR+ shows moderate 1q gain and 16q loss.

## Figure 03 - PCA (`03_pca.png`)

- **Left panel - Scree plot.** X-axis: principal component index (1-10). Y-axis: percentage of total variance explained. Red line shows cumulative variance.
- **Right panel - PC1 vs PC2 scatter.** Each point is a sample, colored by subtype.
- **Conclusion:** PC1 (16.1%) and PC2 (6.2%) together capture 22.3% of variance. HER2+ partially separates on PC1, but HR+ and Triple Neg overlap. The data is high-dimensional with no single dominant axis.

## Figure 04 - t-SNE (`04_tsne.png`)

- **X-axis / Y-axis:** t-SNE dimensions 1 and 2 (arbitrary units, non-linear embedding). Three panels with perplexity = 5, 15, 30.
- **Conclusion:** HER2+ forms a recognisable cluster at all perplexities. HR+ and Triple Neg remain intermixed regardless of perplexity, confirming they are harder to separate in raw feature space.

## Figure 05 - Hierarchical Clustering (`05_hierarchical_clustering.png`)

- **X-axis:** Individual samples (leaf labels colored by true subtype).
- **Y-axis:** Ward linkage distance (Euclidean distance between cluster centroids).
- **Conclusion:** Silhouette score = 0.092 (very low), meaning 3 clusters do not emerge naturally from unsupervised clustering on all 2834 features. Feature selection will be necessary before classification.

## Figure 06 - Manhattan Plot of Kruskal-Wallis Tests (`06_manhattan_kruskal.png`)

- **X-axis:** Genomic position across all chromosomes.
- **Y-axis:** -log10(p-value) from Kruskal-Wallis test comparing CN distributions across the 3 subtypes for each region.
- **Red dashed line:** Bonferroni-corrected significance threshold (0.05 / 2834 = 1.76 × 10⁻⁵). Red dots with gene labels mark the 6 regions that pass this threshold.
- **Conclusion:** 6 regions pass Bonferroni correction, all on chromosomes 12 and 17. The top hit (chr17:35.1-35.3 Mb, H=73.9, p=9.0 × 10⁻¹⁷) contains the ERBB2/HER2 amplicon locus - a biologically expected positive control.

## Figure 07 - Correlation vs Genomic Distance (`07_correlation_vs_distance.png`)

- **Left panel - Scatter.** X-axis: gap between region boundaries (Mb). Y-axis: Pearson correlation (r) between CN profiles of two regions across 100 samples. Each dot is one region pair. Red dashed line shows the cross-chromosome baseline (median r = 0.15).
- **Right panel - Binned medians.** X-axis: distance bins (Mb) plus a cross-chromosome reference bar (red). Y-axis: median Pearson r within each bin. Bar labels show pair counts.
- **Conclusion:** Nearby regions are highly correlated (median r = 0.92 within 0.1 Mb) and correlation decays with distance, approaching the cross-chromosome baseline (r = 0.15) at 100-200 Mb. The non-zero baseline reflects genome-wide CN patterns shared across chromosomes (e.g., whole-genome instability). This spatial autocorrelation means neighboring features carry redundant information - feature selection should account for this by either filtering correlated neighbors or selecting representative regions.

## Figure 08 - Sample Similarity Matrix (`08_similarity_matrix.png`)

- **Rows / Columns:** 100 samples, ordered by subtype (HER2+, HR+, Triple Neg). White lines separate subtype blocks.
- **Color:** Hamming distance (fraction of regions where two samples have different CN calls). Darker = more similar.
- **Conclusion:** The HER2+ block (top-left) is visually darker (more similar internally) than the other two. HR+ and Triple Neg blocks have more within-group heterogeneity and some cross-group similarity, consistent with their overlap in PCA/t-SNE.
