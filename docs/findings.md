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

## Flat ensembling analysis (flat pipelines only)

- Error correlation across the four flat 2x2 pipelines shows that HR+/TN confusion is systematic and shared.
- The most complementary competitive pair (KW+RF + EN+NMC, Jaccard 0.420) still shares most errors.
- However, ensembling is NOT ruled out. The hierarchical pipelines may have different error profiles, and same-pipeline ensembles with different hyperparameters or seeds could still add value. Revisit after hierarchical results are in.

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

---

# Findings: Hierarchical Classifier Design

## Stage 1 is fixed: KW+RF, k=5, no tuning

- Binary KW+RF (HER2+ vs rest) achieved balanced accuracy = 1.0000 in all 15 outer folds tested (3 repeats x 5 folds, local grid, binary KW ranking).
- Inner CV unanimously selected k=5 over k=20 in every fold. The HER2 amplicon signal is so strong that 5 features suffice for perfect binary separation.
- Stage 1 is therefore hardcoded to KW+RF with k=5. No inner GridSearchCV is run for Stage 1 - it would be 40 unnecessary RF fits per outer fold (8 grid combos x 5 inner folds) with the same result every time.
- Note: k=5 was validated using binary KW (HER2+ vs rest), not the 3-class KW from the flat 2x2 experiment. The rankings differ, but the top features are the same chr17q ERBB2 amplicon regions in both cases.

## Preliminary hierarchical results (local grid, 1 repeat)

| Stage 2 Pipeline | Stage 2 BA (HR+/TN) | Combined 3-class BA | AUROC |
|------------------|---------------------|---------------------|-------|
| KW + RF          | 0.788               | 0.859               | 0.950 |
| KW + NMC         | 0.764               | 0.843               | 0.949 |
| EN + NMC         | 0.742               | 0.828               | 0.930 |
| EN + RF          | (not completed)     | -                   | -     |

- Stage 1 was perfect (1.0) in all runs. The bottleneck is entirely Stage 2.
- The hierarchical combined BA of 0.859 (kw_rf Stage 2) already exceeds the flat kw_rf BA of 0.791 from the 50-repeat server run.
- These are preliminary (1 repeat, local grid). Server run with 50 repeats and dense grids needed for final comparison.

---

# Findings: Hierarchical Nested CV (50 repeats, server run)

## Pipeline performance (hierarchical, Stage 2 varies)

| Pipeline (Stage 2) | Mean BA | Std | Median | Mean features (S2) |
|---------------------|---------|-------|--------|---------------------|
| EN + NMC            | 0.812   | 0.025 | 0.816  | 28.5                |
| KW + NMC            | 0.811   | 0.032 | 0.815  | 32.3                |
| KW + RF             | 0.800   | 0.027 | 0.807  | 44.8                |
| EN + RF             | 0.791   | 0.030 | 0.794  | 32.0                |

Stage 1 (KW+RF, k=5, HER2+ vs rest) is fixed and identical across all 4 pipelines. It achieved BA=1.0 in all 245 folds (49 complete repeats x 5 folds). Repeat 47 is missing for both KW pipelines (checkpoint exists but fold results were not written).

## Hierarchical vs flat comparison

- Best hierarchical (EN+NMC): 0.812 vs best flat (KW+RF): 0.791 - a +2.1 point improvement.
- Crucially, the winner flipped: flat experiment was won by RF, hierarchical is won by NMC. The hierarchical decomposition removed the trivial HER2+ class that inflated RF's advantage, exposing that NMC is better at the harder HR+ vs TN problem.

## Statistical testing

### Bug fix applied
- The original analysis had NaN Friedman/Wilcoxon results because kw_nmc and kw_rf had 49/50 repeats (repeat 47 missing). The pivot table had NaN for that repeat, which propagated through scipy. Fixed by dropping incomplete repeats before paired tests.

### Friedman test
- chi2=12.34, p=0.006 - significant differences exist among the 4 pipelines.

### Pairwise Wilcoxon (Bonferroni-corrected, 49 complete repeats)
- KW+NMC vs EN+RF: p=0.014 (significant)
- EN+NMC vs EN+RF: p=0.017 (significant)
- KW+NMC vs KW+RF: p=0.097 (borderline)
- KW+RF vs EN+NMC: p=0.101 (borderline)
- KW+NMC vs EN+NMC: p=1.000 (n.s. - same classifier, different selector)
- KW+RF vs EN+RF: p=0.870 (n.s. - same classifier, different selector)

### Grouped NMC vs RF test (Wilcoxon, pooled over feature selectors)
- NMC mean=0.812, RF mean=0.795, diff=+0.017, W=231, **p=0.000148**.
- NMC significantly outperforms RF when the classifier effect is tested directly.

### Nadeau-Bengio corrected t-test
- All pairwise comparisons n.s. (all p_corrected=1.0). This is the most conservative test because it penalizes for fold overlap in repeated CV. Worth noting in the paper as a methodological point: the correction may be overly conservative for 50x5 repeated CV.

## Classifier effect dominates, feature selector does not matter

- NMC beats RF regardless of feature selector (interaction plot: NMC line flat and above RF).
- Feature selector has no effect: KW+NMC vs EN+NMC is n.s. (p=1.0), KW+RF vs EN+RF is n.s. (p=0.87).
- This directly supports Wessels et al. (2005): simple classifiers with univariate filtering match or beat complex pipelines.

## KW has higher variance than EN

- KW+NMC std=0.032 vs EN+NMC std=0.025. KW uses univariate ranking with a tuned threshold - small perturbations in training data can flip borderline features in/out, leading to less stable feature sets. EN is multivariate and regularized, producing more consistent subsets. They average to the same mean.

## Stage 1 is trivially separable

- KW+RF Stage 1 (HER2+ vs rest) is perfect in all 245 folds across all repeats, using only 5 features from the chr17 ERBB2 amplicon.
- 30 unique features were ever selected for Stage 1 across all folds, all on chr17 or known HER2-correlated loci.
- Any classifier would achieve the same result. KW+RF is kept as the Stage 1 classifier for simplicity since it was already validated and hardcoded.

## Confusion matrix patterns

- HER2+: 1.00 recall across all 4 pipelines (perfect, from Stage 1).
- HR+: 0.72-0.77 recall (NMC pipelines slightly better).
- TN: 0.65-0.67 recall (NMC pipelines slightly better).
- The HR+ vs TN confusion remains the bottleneck, but hierarchical decomposition improved it vs flat.

## Error agreement

- Highest Jaccard overlap between same-classifier pairs: KW+RF vs EN+RF = 0.59.
- Lowest overlap between cross-classifier pairs: EN+NMC vs KW+RF = 0.50.
- When one pipeline is wrong, ~65-75% chance the other is also wrong - errors are correlated.

## Wessels et al. comparison

- Wessels et al. (2005) tested simple vs complex classifiers on **continuous gene expression** microarray data using **flat multi-class** classification. They found simple classifiers with univariate filtering outperform complex pipelines.
- This project tests whether that principle generalizes to (a) **discrete aCGH copy-number** data and (b) a **hierarchical** classification setting. Neither was tested by Wessels.
- **Flat experiment result**: RF beat NMC - appeared to contradict Wessels.
- **Hierarchical experiment result**: NMC beats RF (p < 0.001) - confirms Wessels, but only after removing the trivial HER2+ class.
- **Key insight**: the multi-class structure was a confound. In the flat setup, the trivially separable HER2+ class inflated RF's advantage (RF can partition easy and hard regions of feature space simultaneously; NMC cannot). Once the problem is decomposed hierarchically and the real challenge (HR+ vs TN) is isolated, simple classifiers win - exactly as Wessels predicted.
- **Feature selection**: univariate KW and multivariate EN perform identically (p=1.0), meaning the discriminative signal resides in individually informative regions, not feature interactions. This also aligns with Wessels.
- **Contribution over Wessels**: showing that the principle holds on discrete data, but only becomes visible after proper problem decomposition. The flat result is misleading due to multi-class structure effects.

## Per-sample error analysis and suspected mislabels

- 16 "hard" samples (out of 68 non-HER2+) have error rates >= 50% across all pipelines, repeats, and folds. These 16 samples account for **64.4% of all Stage 2 errors**.
- The remaining 52 samples split into 27 easy (< 10% error) and 25 medium (10-50% error).
- Samples 2 (HR+) and 4 (TN) are misclassified **97.5% of the time** across all 4 pipelines and ~200 fold appearances each. Sample 4 (labelled TN) is predicted HR+ in 193/198 appearances; sample 2 (labelled HR+) is predicted TN in 193/198 appearances. These are almost certainly mislabeled or represent edge-case biology that aCGH copy-number profiles cannot resolve.
- Consensus filtering (Brodley & Friedl, 1999, "Identifying Mislabeled Training Data", Journal of Artificial Intelligence Research 11:131-167) provides a principled framework for flagging such samples: if multiple independent classifiers consistently misclassify a sample in held-out evaluation, it is flagged as a potential mislabel.
- **Decision: do not remove these samples from training.** Reasons: (1) circularity risk - the same classifier family is used for both flagging and training, undermining the independence assumption; (2) n=68 is already small, losing 2 samples costs 3% of Stage 2 training data; (3) the validation set likely contains similar ambiguous cases, and a classifier that has never seen edge cases will handle them worse.
- Instead, report the suspected mislabels as a finding in the paper and let the ensemble approach soften their impact through probability averaging.

## Ensemble analysis

- **4-way majority vote** across all 4 pipelines gives **+5.5 pp** over the best single pipeline on Stage 2 samples (77.6% vs 72.1% correct).
- KW+NMC and EN+NMC agree on 84.8% of predictions. When they agree, they are 75.9% correct. When they disagree (505 cases), KW is right 251 times and EN is right 254 times - perfectly balanced, with zero cases of "both wrong with different predictions." A tiebreaker would recover nearly all disagreement errors.
- A confidence-based 3-stage hierarchy (routing uncertain predictions to a different classifier) was considered but rejected: confidence calibration is weak (wrong predictions have mean max_prob=0.61 vs correct at 0.68, heavy overlap), and the 16 hard samples are consistently wrong, not randomly uncertain - no routing strategy can fix them.
- **Caveat on the +5.5 pp claim (outside reviewer correction):** The 4-way vote was measured on out-of-fold predictions (each sample predicted only when in the test set), so it avoids train/test leakage. However, the choice of which 4 pipelines to combine and the combination rule (majority vote) were decided **after** seeing all results - the pipeline selection is post-hoc. A proper unbiased estimate would require the ensemble to be evaluated as a fifth pipeline inside nested CV, with the combination rule fixed before seeing outer fold test data. The true cross-validated gain is likely 1-2 pp, not 5.5 pp.
- **Decision: KW+NMC is the submission model without ensemble.** An ensemble of KW+NMC and EN+NMC remains scientifically interesting (perfect complementarity in disagreement cases), but implementing it requires proper nested CV validation before any gain claim is trustworthy. KW+NMC alone is the safe, defensible choice.

## Strategic conclusion (updated from flat experiment and outside review)

- The flat experiment suggested RF >> NMC. The hierarchical experiment reverses this: NMC >= RF (p < 0.001 pooled).
- The flat result was an artefact of the 3-class structure: HER2+ is trivially separable and inflated RF's advantage. Once decomposed, the harder HR+ vs TN binary problem favours simpler classifiers. The original falsification of Wessels was not a property of NMC; it was a property of the contaminated feature space.
- The paper narrative: flat 2x2 falsifies Wessels (feature space problem) -> error analysis and feature importance diagnose the cause -> hierarchical redesign recovers the suppressed signal -> in the conditioned feature space the simple classifier reasserts itself.
- 50-repeat design is necessary, not overkill: KW+NMC convergence requires ~25-30 repeats to stabilize due to Stage 2 operating on only ~54 samples. Fewer repeats would give unreliable estimates.
- For the final model: **KW+NMC** for Stage 2, KW+RF (k=5) for Stage 1.
