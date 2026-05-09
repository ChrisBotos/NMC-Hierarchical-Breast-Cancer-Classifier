# Findings - TB-Project (CATS)

Curated log of notable findings, in reverse-chronological order.

---

### 2026-05-09 - Why NMC beats logistic regression: constrained linear classifier on discrete data

**Category:** discussion_point
**Status:** confirmed

**Summary:**
NMC is itself a linear classifier (its decision boundary is the perpendicular bisector between class centroids). It beats ElasticNet logistic regression not because it is non-linear or fundamentally different, but because it is a MORE CONSTRAINED linear classifier with fewer free parameters. In the small-sample discrete-feature regime (n=72, p=50, features in {-1,0,1,2}), this constraint prevents overfitting.

**Key argument:**
- Logistic regression with k features has k free parameters (one coefficient per feature), each independently optimized.
- NMC estimates 2 centroids for binary classification. The decision boundary is fully determined as the perpendicular bisector - its orientation cannot be freely adjusted.
- With ~72 Stage 2 samples and top_k=50 features, logistic regression has enough freedom to overfit even with L1/L2 regularization. NMC cannot overfit as easily because the boundary shape is geometrically locked.
- For discrete copy-number data (-1, 0, 1, 2), all features live on the same scale. NMC's implicit equal-variance assumption (perpendicular bisector) is approximately satisfied. This constraint does not hurt - it helps.
- "Average copy-number state per class per region" (what centroids compute) is a biologically meaningful quantity. A linear logistic boundary has no such natural interpretation for discrete ordinal data.

**EN_NMC vs standalone EN:**
- The ElasticNet in EN_NMC uses the exact same model (LogisticRegression with elasticnet penalty) as standalone EN. Feature selection is identical.
- The difference is purely in classification: EN_NMC discards the logistic coefficients after selection and trains NMC from scratch on the selected features.
- EN is a better selector than classifier for this data. NMC is a better classifier given the selected features. Separating the two roles outperforms a single model doing both.

**Implications for the paper discussion:**
This is the bias-variance tradeoff in action. NMC wins not through model complexity but through principled constraint. This extends Wessels et al.'s finding: simple classifiers win because their inductive bias (here: centroid-based, equal-variance) happens to match the structure of discrete genomic data. The lesson is not "always use simple models" but "match model assumptions to data structure."

---

### 2026-04-25 - Stage 1 threshold bug fix eliminates pathological tiebreaker artifacts

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** anomaly
**Status:** confirmed

**Summary:**
Inner-CV-tuned Stage 1 threshold could select pathological values because the problem is trivially separable (ROC is a step function). Fixing threshold at 0.5 eliminates the artifact.

**Key evidence:**
- For a trivially separable binary problem, the ROC curve is a step function: every threshold between the most-positive "rest" score and the least-positive "HER2+" score achieves perfect classification.
- GridSearchCV or threshold-sweep logic that picks among ties has no principled basis for choosing within this flat region.
- Depending on implementation, it may select a threshold near the extreme of the tied range, fragile to slight score perturbations in test data.
- Fix: Stage 1 threshold fixed at 0.5 (natural midpoint for probability outputs).

**Implications:**
Do not tune hyperparameters that the data cannot inform. When inner CV produces ties over a wide threshold range, fix the parameter at the natural decision boundary rather than tuning.

---

### 2026-04-25 - Feature engineering experiments: three-state encoding and arm-level features both fail

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** negative_result
**Status:** confirmed

**Summary:**
Two feature engineering approaches were tested and both degraded performance. Three-state encoding (collapsing CN=2 to CN=1) lost 6.3 pp BA. Arm-level mean CN features collapsed Stage 1 from 1.00 to 0.62 BA. Region-level resolution after r>0.8 merging is the correct granularity.

**Key evidence:**
- Three-state encoding: np.sign on merged region values. Result: -6.3 pp BA drop (0.812 to ~0.75), variance increased from 0.055 to 0.090. CN=2 carries genuine discriminative signal in Stage 2 on chr12q and chr5q.
- Arm-level mean CN features (~44 features from 273 merged regions). Stage 1 BA collapsed to 0.617 (chr17q arm averaging dilutes focal ERBB2 amplicon). Stage 2 BA dropped to ~0.75 (discriminative signal on chr12q and chr5q is focal, not broad).
- Merging threshold sweep: r=0.8 is optimal. Raw data is significantly worse (-6.9 pp, p=0.008). r=0.7 and r=0.9 are ~2 pp lower but n.s. The merging threshold is not a sensitive hyperparameter.

**Implications:**
The discriminative signal in aCGH copy-number profiles is focal, not broad. Arm-level averaging destroys it. All representations must preserve the full ordinal scale including CN=2. Region-level merging at r>0.8 is the sweet spot.

---

### 2026-04-25 - Final hierarchical nested CV (200 repeats): EN+NMC(plateau) wins, NMC beats RF

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** discovery
**Status:** confirmed

**Summary:**
200-repeat hierarchical nested CV with 10 pipeline variants confirms EN+NMC(plateau) as the best pipeline (BA2=0.759). NMC significantly outperforms RF (p=1e-16 Wilcoxon pooled). This reverses the flat experiment where RF won, confirming the hierarchical decomposition changes the classifier comparison.

**Key evidence:**
- Top 3 pipelines: EN+NMC(pens) BA2=0.759, NMC Pens Ensemble BA2=0.757, KW+NMC(pens) BA2=0.750.
- Worst: EN+RF BA2=0.691. NMC beats RF by +2.7 pp pooled (p=1e-16).
- Convergence: cumulative mean BA2 stabilises by ~75 repeats; 200 is well into plateau.
- Nadeau-Bengio corrected t-test: all pairwise comparisons n.s. (p_corrected=1.0) due to variance inflation.
- Bootstrap CIs: 38/45 pairs distinguishable. Two indistinguishable clusters: mid-tier NMC (BA2 0.736-0.740) and top-tier plateau (EN+NMC pens vs NMC Pens Ensemble).
- HER2+ recall: 1.00 across all variants (Stage 1 perfect). HR+ recall: 0.73-0.78. TN recall: 0.66-0.74.

**Implications:**
Submit EN+NMC(plateau) as the final model. True test set BA2 probably 0.75-0.76, combined 3-class BA 0.82-0.85. For the paper: report Nadeau-Bengio for formal claims, bootstrap CIs as supplementary distinguishability.

---

### 2026-04-25 - Plateau ensembling ranking flip: EN+NMC(pens) beats KW+NMC(pens) despite weaker base

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** discovery
**Status:** confirmed

**Summary:**
EN+NMC (base BA2=0.729) gains +0.030 from plateau ensembling to reach 0.759, overtaking KW+NMC (base 0.736, gain +0.014 to 0.750). The gain scales with hyperparameter grid size (EN+NMC has 800 combos vs KW+NMC's 32), consistent with inner CV underfitting rescue.

**Key evidence:**
- Gain scales with grid size: EN+NMC +0.030 (800 combos), KW+NMC +0.014 (32 combos), Standalone EN +0.004 (50 combos).
- Inner CV optimism gap for EN+NMC single-best: +0.084 (inner CV overestimates). Inner-test correlation: r=-0.378 (negative - noise-driven selection).
- 303 distinct top-1 hyperparameter combos across 1000 folds. Selection is near-random within the plateau.
- Leave-one-repeat-out plateau overlap: 14.9/15 configs match full plateau. Direct leakage from any individual repeat is negligible.
- Decomposition: ~0 pp leakage, ~2 pp hyperparameter stabilisation, ~1 pp ensemble averaging.

**Implications:**
Plateau ensembling is a legitimate technique for large-grid pipelines with noisy inner CV. The gain is real, not overfitting. True deployment BA2 for EN+NMC(pens) is probably ~0.755.

---

### 2026-04-25 - Performance ceiling at BA2~0.76 set by hard/mislabelled samples

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** discovery
**Status:** confirmed

**Summary:**
10 samples are misclassified in >80% of evaluations across ALL pipeline variants. KW test for hard vs easy within each class finds 0 significant features. Performance is at ceiling for this feature representation.

**Key evidence:**
- Top 3 hardest: Array.22 (TN, 96.5%), Array.67 (HR+, 96.2%), Array.23 (HR+, 95.6%).
- Excluding 3 suspected mislabels (Array.67, Array.22, Array.113) improves BA2 by +3.0-3.4 pp uniformly across all pipelines.
- Ceiling estimates: remove 4 hardest -> BA2~0.81; remove 8 -> BA2~0.86; remove 10 -> BA2~0.89.
- No KW feature at any significance level distinguishes hard from easy samples. The feature representation has no signal left to exploit.

**Implications:**
The 0.76 BA2 ceiling is data-limited, not classifier-limited. Removing mislabels from training would not help for validation (validation may have its own mislabels, and the decision boundary learned with mislabels is more robust).

---

### 2026-04-25 - Stage 1 fixed at KW+RF k=5: perfect separation of HER2+ vs rest

**Source run:** `results/2026-04-25_smoke_test_final/hierarchical_nested_cv/`
**Category:** validation
**Status:** confirmed

**Summary:**
Binary KW+RF (HER2+ vs rest) achieved BA=1.0000 in all 15 outer folds tested (3 repeats x 5 folds). Inner CV unanimously selected k=5 over k=20 in every fold. Stage 1 is hardcoded with no tuning needed.

**Key evidence:**
- BA=1.0 across all folds. The HER2 amplicon signal (ERBB2, chr17q) is so strong that 5 features suffice.
- k=5 validated using binary KW ranking (HER2+ vs rest), not 3-class ranking.
- No inner GridSearchCV needed - would be 40 unnecessary RF fits per outer fold.

**Implications:**
Stage 1 is a solved problem. All effort should focus on Stage 2 (HR+ vs TN). The "4 pipelines" in the hierarchical experiment refer to Stage 2 only.

---

### 2026-04-24 - Flat nested CV 2x2 (50 repeats): RF beats NMC, classifier matters more than feature selector

**Source run:** `results/2026-04-24_server_run_v2/nested_cv_2x2/`
**Category:** discovery
**Status:** superseded

**Summary:**
In the flat 3-class setting, KW+RF achieves the highest mean BA (0.791). Classifier choice matters more than feature selector: NMC->RF improves KW by +16 pp, while KW->EN moves RF by only -2 pp. All pipelines struggle with HR+ vs TN confusion.

**Key evidence:**
- Pipeline ranking: KW+RF 0.791, EN+RF 0.770, EN+NMC 0.746, KW+NMC 0.631.
- Friedman test: chi2=117.24, p<0.000001.
- Nadeau-Bengio corrected: only KW+NMC vs KW+RF significant (p=0.018). Top 3 not distinguishable.
- TN recall: 0.62-0.64 across all pipelines - shared structural failure.
- Error diversity driven by classifier, not feature selector (Jaccard: same-classifier pairs ~0.54, cross-classifier ~0.36).

**Implications:**
Superseded by hierarchical results which reverse the NMC vs RF finding. The flat result (RF > NMC) was an artifact of the trivially separable HER2+ class inflating RF's advantage. Led to the hierarchical classifier architecture.

---

### 2026-04-24 - HR+ vs TN-specific KW ranking reveals hidden signal on chr12q and chr5q

**Source run:** `results/2026-04-24_server_run_v2/nested_cv_2x2/`
**Category:** discovery
**Status:** confirmed

**Summary:**
KW ranking on only HR+ and TN samples surfaces completely different features than 3-class ranking. 3-class top 30: dominated by chr17 (11/30). HR+/TN top 30: zero chr17 features, instead chr12q (23/30) and chr5q (7/30). Only 13/30 overlap.

**Key evidence:**
- 17 features surfaced by 2-class ranking were ranked 31st-78th in 3-class ranking - buried by HER2 signal.
- This confirms the hierarchical classifier hypothesis: Stage 2 with dedicated HR+/TN feature ranking accesses discriminative signal the 3-class approach misses.

**Implications:**
Validates the hierarchical architecture. The path forward is Stage 1 (HER2+ vs rest, trivial) then Stage 2 (HR+ vs TN, with dedicated feature ranking).

---

### 2026-04-23 - Region merging (r>0.8) removes 90% of features while preserving all major signals

**Source run:** `results/2026-04-23_default_run/compare_explorations/`
**Category:** validation
**Status:** confirmed

**Summary:**
2834 raw regions merged into 273 consensus segments (90.4% reduction). PCA structure preserved, top KW hit (ERBB2/TCAP, chr17) remains #1, cumulative variance in first 10 PCs doubled from 22.3% to 46.0%.

**Key evidence:**
- PCA: PC1 explains 16.1% (raw) vs 15.4% (merged). HER2+ separation maintained.
- Median adjacent correlation dropped from 0.925 to 0.590.
- Silhouette score dropped (0.092 to 0.025) but was already poor - expected from median-consensus blurring discrete CN values.
- r>0.8 threshold sits just below natural adjacent-region correlation (~0.83 at 0.5 Mb).
- Adjacent-only merging preserves contiguous genomic blocks. All-pairwise would merge biologically independent events.

**Implications:**
Merged data is the correct starting point for supervised classification: fewer features, less redundancy, same discriminative signal. r=0.8 is robust (not sensitive to exact threshold).

---

### 2026-04-24 - Wessels et al. comparison: simple classifiers win after proper problem decomposition

**Source run:** `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/`
**Category:** discovery
**Status:** confirmed

**Summary:**
In the hierarchical setting, NMC (simple) significantly beats RF (complex) at p=1e-16. Feature selector does not matter (KW vs EN n.s.). This directly supports Wessels et al. (2005): simple classifiers with univariate filtering match or beat complex pipelines - but only once the problem is properly decomposed.

**Key evidence:**
- Flat experiment: RF > NMC (artifact of trivially separable HER2+).
- Hierarchical experiment: NMC > RF by +2.7 pp pooled, p=1e-16 (Wilcoxon).
- KW+NMC vs EN+NMC: p=0.055, not significant after correction.
- The reversal occurs because the hierarchical decomposition removes the HER2+ class from Stage 2, and NMC handles the remaining HR+/TN binary problem better than RF.

**Implications:**
Answers the research question: Wessels' principle holds for discrete aCGH data, but feature redundancy and multi-class structure must be addressed through preprocessing (merging) and problem decomposition (hierarchical classification) before the simple-classifier advantage emerges.
