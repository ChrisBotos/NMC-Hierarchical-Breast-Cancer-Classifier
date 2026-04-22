# Nested CV Analysis Phase

Phase documentation for `analyse_nested_cv.py`. This script aggregates fold-level results from the 2x2 nested CV experimental design, computes summary statistics, runs statistical comparisons, and produces figures.

---

## Figures

### 01_pipeline_comparison.png

**Description:** Box plot comparing balanced accuracy distributions across the four pipelines (KW + NMC, KW + RF, EN + NMC, EN + RF). Each data point represents the mean balanced accuracy across 5 outer folds for a single repeat. Individual repeat scores are overlaid as jittered strip points. Significance brackets annotate pairwise comparisons that remain significant after Bonferroni correction.

**Axes:**
- X-axis: Pipeline (categorical).
- Y-axis: Mean balanced accuracy (across 5 outer folds).

**Interpretation:** Allows visual comparison of central tendency, spread, and overlap between pipelines. Significance brackets indicate which pairwise differences survive correction for multiple comparisons.

### 02_interaction_plot.png

**Description:** Interaction plot showing the 2x2 factorial structure of the experimental design. Two lines (NMC = simple classifier, RF = complex classifier) are plotted across the two feature selection methods (Kruskal-Wallis, Elastic Net). Error bars represent 95% confidence intervals computed from the repeated CV means.

**Axes:**
- X-axis: Feature selection method (Kruskal-Wallis, Elastic Net).
- Y-axis: Mean balanced accuracy.

**Interpretation:** Directly visualises main effects and interaction. Parallel lines indicate no interaction (classifier and feature selection effects are additive). Crossing lines indicate interaction (the best classifier depends on the feature selection method). This figure ties directly to the Wessels et al. research question about whether simple classifiers outperform complex ones on high-dimensional genomic data.

---

## Key Outputs

- `all_fold_results.csv` - Aggregated fold-level results from all pipeline-repeat jobs.
- `summary_statistics.csv` - Per-pipeline mean, std, median, min, max balanced accuracy and AUROC.
- `pairwise_tests.csv` - Pairwise Wilcoxon signed-rank test results with Bonferroni correction (6 comparisons). Only produced if the Friedman omnibus test is significant.

## Statistical Methods

1. **Friedman test** - Non-parametric repeated-measures test across 4 pipelines. Each repeat provides one paired observation per pipeline.
2. **Pairwise Wilcoxon signed-rank tests** - Post-hoc pairwise comparisons with Bonferroni correction (alpha = 0.05 / 6 = 0.0083).
