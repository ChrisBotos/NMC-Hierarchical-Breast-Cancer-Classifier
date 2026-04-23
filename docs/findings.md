# Findings: Raw vs Merged Data

## Merging summary

- 2834 raw regions merged into 273 consensus segments (90.4% reduction).
- Merging is label-free (Pearson r between adjacent regions, no class labels used), so no information leakage.
- Median segment contains 3 raw regions; largest contains 98.
- 93 segments (34.1%) are singletons, meaning they represent genuine CN breakpoints that could not be merged.

## Signal preservation

- PCA structure is preserved: PC1 explains 16.1% (raw) vs 15.4% (merged) — almost identical.
- HER2+ separation on PC1 is maintained; HR+ and Triple Neg still overlap in both cases.
- t-SNE cluster structure is visually the same before and after merging.
- The top KW hit (ERBB2/TCAP locus on chr17, H=73.9, p=9e-17) is the #1 hit in both raw and merged — the strongest biological signal is fully retained.
- All 3 Bonferroni-significant merged segments are on chr17, matching the raw chr17 hits.

## What improved

- Cumulative variance in first 10 PCs doubled: 22.3% (raw) to 46.0% (merged) — signal is concentrated into fewer components because redundant features were removed.
- Median adjacent correlation dropped from 0.925 to 0.590 — spatial redundancy is substantially reduced.
- Correlation at 100-200 Mb distance dropped to non-significant vs cross-chromosome baseline (p=1.0), showing that long-range redundancy was eliminated.

## What got slightly worse (and why it does not matter)

- Silhouette score (k=3 Ward) dropped from 0.092 to 0.025 — both are very low, meaning unsupervised clustering was never separating subtypes well.
- The drop is expected: median-based consensus blurs the discrete CN values (-1, 0, 1, 2) into half-integer values (-0.5, 0.5, 1.5), which spreads within-cluster distances. This affects Euclidean-based unsupervised clustering but does not hurt supervised classification.
- Bonferroni-significant features dropped from 6/2834 to 3/273, but 3/273 is proportionally higher (1.1% vs 0.2%). The Bonferroni correction is simply less harsh with fewer tests.

## Why r > 0.8 was chosen

- The raw data correlation analysis (Phase 0, Section 8) shows that adjacent regions within 0.1 Mb have median r = 0.92, and regions within 0.5 Mb still have median r = 0.83.
- An r > 0.8 threshold sits just below the natural correlation of physically adjacent same-CN-event regions.
- It merges truly redundant neighbours (same copy-number event spanning multiple probes) while preserving genuine breakpoints where CN state changes.
- Higher thresholds (e.g. 0.9) would be too conservative and leave most redundancy in place. Lower thresholds (e.g. 0.6) would start merging biologically distinct regions that happen to be moderately correlated.
- The 0.8 threshold is inspired by the CGHregions approach (van de Wiel & van Wieringen), which uses a similar correlation-based merging strategy for aCGH data.

## Why adjacent-only merging, not all-pairwise (for Methods/Discussion)

- Adjacent merging only considers physically neighbouring regions on the same chromosome. Each merged segment stays a contiguous genomic block with real Start/End coordinates and gene annotations.
- All-pairwise merging (r between every pair, cluster regardless of genomic distance) would merge regions on different chromosomes that happen to correlate — e.g. chr1q gain and chr16q loss co-occur in HR+ but are biologically independent events.
- Adjacent correlation reflects probe-level redundancy (same CN event spanning multiple probes). Distant/cross-chromosome correlation reflects co-occurring but distinct events. Merging the latter destroys information.
- Interpretability: the project requires identifying a "best single biomarker region" with genomic coordinates and gene annotations. All-pairwise would produce abstract feature clusters with no locus.
- Label-free safety: adjacent merging uses only spatial correlation structure. All-pairwise would implicitly encode subtype-specific co-occurrence patterns into the feature set before classification — a subtle form of information leakage even without directly using labels.
- Supported by the data: Phase 0 correlation analysis showed median r = 0.92 within 0.1 Mb, decaying to r = 0.15 (cross-chromosome baseline) at 100-200 Mb. High-correlation neighbours are genuinely redundant; moderate-correlation distant pairs are not.

## Bottom line

- Merging removed ~90% of features while preserving all major biological signals (HER2 amplicon, subtype separation in PCA/t-SNE, top KW hits).
- The only casualty is unsupervised clustering quality, which was poor to begin with (silhouette 0.09) and is not the goal of this project.
- The merged data is a better starting point for supervised classification: fewer features, less redundancy, same discriminative signal.

---

# Findings: Nested CV 2x2 Experiment (50 repeats, server run)

## Pipeline performance

| Pipeline | Mean BA | Std | Median | Mean features selected |
|----------|---------|-----|--------|----------------------|
| KW + RF | 0.791 | 0.025 | 0.793 | 64.8 |
| EN + RF | 0.770 | 0.029 | 0.774 | 31.9 |
| EN + NMC | 0.746 | 0.032 | 0.750 | 15.6 |
| KW + NMC | 0.631 | 0.031 | 0.633 | 25.6 |

## Statistical testing

- Friedman test: chi2=117.24, p < 0.000001 - significant differences exist.
- Wilcoxon signed-rank (Bonferroni): all 6 pairs significant. But this test is anti-conservative for repeated CV because it treats overlapping folds as independent.
- Nadeau-Bengio corrected t-test (the appropriate test): only KW+NMC vs KW+RF is significant (p=0.018). The top 3 pipelines (KW+RF, EN+RF, EN+NMC) are not statistically distinguishable after correction.
- The violin plot uses Nadeau-Bengio p-values, not Wilcoxon, to avoid misleading significance brackets.

## Classifier matters more than feature selector

- Swapping NMC for RF improves KW from 0.631 to 0.791 (+16 points).
- Swapping KW for EN only moves RF from 0.791 to 0.770 (-2 points).
- RF benefits from large feature sets (median k=75); NMC needs aggressive filtering (median k=15).

## The HR+ vs Triple Negative problem

- All pipelines classify HER2+ easily (high recall). The accuracy bottleneck is HR+ vs TN confusion.
- TN recall is 0.62-0.64 across all pipelines - substantial room to improve.
- The confusion matrices show this is a shared structural failure, not pipeline-specific.

## Error agreement analysis

- Pairwise error overlap (Jaccard index) is highest between pipelines sharing the same classifier: KW+RF vs EN+RF = 0.542.
- Cross-classifier pairs are more complementary: KW+NMC vs EN+RF = 0.357 (lowest).
- Error diversity is driven by the classifier, not the feature selector.
- When KW+RF is wrong, EN+NMC is also wrong 65.4% of the time - still high overlap.
- Flat majority-vote ensemble ceiling is modest: ~2-4 BA points recovery. The shared HR+/TN errors are correlated across all pipelines.

## Flat ensembling ruled out

- The error correlation data shows that the HR+/TN confusion is systematic and shared.
- Flat ensembling cannot solve a structural failure mode - it only helps with random disagreements.
- The most complementary competitive pair (KW+RF + EN+NMC, Jaccard 0.420) still shares most errors.

## HR+ vs TN-specific KW ranking reveals hidden signal

- A KW ranking computed on only HR+ and TN samples (excluding HER2+) surfaces completely different features.
- 3-class ranking top 30: dominated by chr17 (11/30 features, all near ERBB2).
- HR+ vs TN ranking top 30: zero chr17 features. Instead dominated by chr12q (23/30) and chr5q (7/30).
- Only 13/30 features overlap between the two rankings.
- 17 features surfaced by the 2-class ranking were ranked 31st-78th in the 3-class ranking - buried by the HER2 signal.
- This confirms the hierarchical classifier hypothesis: a Stage 2 classifier using a dedicated HR+/TN feature ranking should access discriminative signal that the 3-class approach misses.

## Strategic conclusion

- The 2x2 experiment answered the Wessels et al. research question: univariate filtering works (KW >= EN), but the complex classifier wins (RF >> NMC) on discrete aCGH data.
- The path forward is a hierarchical classifier: Stage 1 (HER2+ vs rest, trivial) then Stage 2 (HR+ vs TN, with dedicated feature ranking on chr12q/chr5q signal).
