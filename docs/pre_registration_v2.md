# Pre-registration: Hierarchical Nested CV v2

**Date:** 2026-04-24
**Status:** Pre-registered before any v2 code is written or executed.

---

## 1. Motivation

The v1 hierarchical experiment (50 repeats x 5 folds x 4 pipelines) established that NMC significantly outperforms RF for the Stage 2 (HR+ vs TN) problem (pooled Wilcoxon p=0.000148), reversing the flat experiment finding. Three principled extensions are now applied uniformly to all pipelines:

1. **Cost-sensitive NMC** - inverse class-frequency log-weight bias in the softmax probability computation, equivalent to adjusting class priors. Addresses the HR+/TN class imbalance (36 vs 32 in Stage 2 training).
2. **Stage 1 threshold calibration** - inner CV over probability thresholds for the HER2+ vs rest decision. Expected to be degenerate (all thresholds yield BA=1.0) but provides a principled safety net for the competition test set.
3. **K-ensemble and pipeline-ensemble variants** - averaging predicted probabilities across multiple k values or across pipelines, tested as pre-registered pipeline variants.

---

## 2. Pipeline Variants (7 total)

| # | CLI name | Stage 2 behavior | GridSearchCV? |
|---|----------|-------------------|---------------|
| 1 | `kw_nmc` | KW + NMC (cost-sensitive, class_weight='balanced') | Yes, full k grid |
| 2 | `en_nmc` | EN + NMC (cost-sensitive, class_weight='balanced') | Yes, full EN grid |
| 3 | `kw_rf` | KW + RF (class_weight='balanced', as before) | Yes, full grid |
| 4 | `en_rf` | EN + RF (class_weight='balanced', as before) | Yes, full grid |
| 5 | `kw_nmc_kens` | Average predict_proba over k in {15, 20, 30, 50} (cost-sensitive NMC) | No |
| 6 | `nmc_ensemble` | Average probabilities from pipeline 5 (k-ensemble) and pipeline 2 (EN+NMC) | Mixed |
| 7 | `kw_nmc_kgrid` | KW + NMC (cost-sensitive), k restricted to {15, 20, 30, 50} | Yes, restricted grid |

**Stage 1 is identical across all 7 variants:** KW+RF, k=5, fixed, no inner CV tuning.

All 7 variants receive Stage 1 threshold calibration.

### Justification for each variant

- **Pipelines 1-4 (baselines):** Direct v2 counterparts of the v1 baselines, with the addition of cost-sensitive NMC and threshold calibration. These allow assessing the effect of cost-sensitivity vs v1.
- **Pipeline 5 (k-ensemble):** Averaging over multiple k values avoids committing to a single feature count and may stabilize predictions. The k values {15, 20, 30, 50} span the range where v1 inner CV most frequently selected optimal k.
- **Pipeline 6 (NMC ensemble):** Combines KW-selected and EN-selected features via probability averaging. v1 error agreement showed KW+NMC and EN+NMC are complementary (when they disagree, each is right ~50% of the time).
- **Pipeline 7 (k-grid):** Like pipeline 1 but with the k grid restricted to the same 4 values used in the k-ensemble, enabling a fair comparison of "ensemble averaging" vs "CV-selected best k" from the same candidate set.

### Pipeline 6 (EN k-ensemble) - skipped

EN k-ensemble (analogous to pipeline 5 but with EN+NMC for each k) is excluded because EN's `top_k` interacts with `C` and `l1_ratio` - there is no single "natural" EN pipeline at a fixed k, making the ensemble definition ambiguous.

---

## 3. Pre-registered Decisions

### 3.1 Cost-sensitive NMC

- **Method:** Log-weight bias added to negative distances before softmax in predict_proba. Weight for class c = log(n_total / (n_classes * n_c)), where n_c is the count of class c in the training fold.
- **Equivalent to:** Bayesian classification with adjusted class priors (inverse class frequency).
- **NOT tuned:** Weights are computed deterministically from training fold class counts. No hyperparameter search over weight values.
- **Backward compatibility:** When class_weight=None, behavior is identical to the v1 NearestCentroidWithProba (verified by mandatory test before any v2 run).

### 3.2 Stage 1 threshold calibration

- **Threshold range:** [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90] (17 values).
- **Method:** For each outer fold, fit Stage 1 KW+RF on the training set. Run inner CV (5-fold, stratified on binary HER2+ vs rest labels) over all threshold candidates. Select the threshold with the highest mean balanced accuracy across inner folds.
- **Tie-breaking:** When multiple thresholds achieve the same best BA (expected when Stage 1 is perfect), default to 0.5.
- **Rationale:** Stage 1 achieved BA=1.0 in all 245 v1 folds. Calibration will be degenerate in most folds, defaulting to 0.5. Kept because: (a) principled, (b) cheap (~2.5s extra per task), (c) the competition test set may contain borderline HER2+ cases.

### 3.3 K-ensemble (pipeline 5)

- **K values:** {15, 20, 30, 50}.
- **Method:** For each k, fit a complete KW+NMC pipeline (StandardScaler -> KruskalWallisSelector(k) -> NearestCentroidWithProba(class_weight='balanced')) on the Stage 2 training subset. Average predict_proba across all 4 pipelines. Final prediction = argmax of averaged probabilities.
- **No inner CV:** The k values are fixed, not selected. This is a pre-registered ensemble, not a tuning procedure.

### 3.4 Pipeline ensemble (pipeline 6)

- **Components:** Pipeline 5 (k-ensemble, KW-based) + Pipeline 2 (EN+NMC, with GridSearchCV).
- **Method:** Run both components within the same outer fold. Average their predict_proba outputs. Final prediction = argmax.
- **Equal weighting:** No learned weights. Simple average.
- **EN+NMC component:** Uses full GridSearchCV with the en_nmc grid (same as pipeline 2).

### 3.5 Suspected mislabels

- **Samples:** Array.67 (index 2, HR+) and Array.22 (index 4, TN).
- **Evidence:** 97.5% error rate across all 4 v1 pipelines and ~200 fold appearances each (from hard_sample_summary.csv).
- **Treatment:** NOT removed from training. Instead, all metrics are computed both with and without these samples in the test fold. The "excl" metrics provide a sensitivity analysis.
- **Runner verification:** At startup, the runner asserts that sample names at config indices match config names. If the assertion fails, the runner aborts.

---

## 4. Evaluation Protocol

### 4.1 Cross-validation scheme

- **Outer CV:** 5-fold stratified k-fold, stratified on original 3-class labels.
- **Inner CV:** 5-fold stratified k-fold (for GridSearchCV pipelines and threshold calibration).
- **Repeats:** 50 (different random seeds).
- **Total tasks:** 7 pipelines x 50 repeats = 350 SLURM array tasks.

### 4.2 Metrics

**Headline metric:** Combined 3-class balanced accuracy (via hierarchical routing).

**Supporting metrics:**
- Stage 1 binary balanced accuracy (HER2+ vs rest).
- Stage 2 binary balanced accuracy (HR+ vs TN, on true HR+/TN test samples only).
- Macro-averaged AUROC via Bayesian probability decomposition.
- All metrics computed both with and without suspected mislabels.

### 4.3 Per-fold output columns (CSV)

Superset of v1 columns, adding:
- `stage1_threshold` (float) - calibrated threshold, or 0.5 if all thresholds tied.
- `combined_bal_acc_excl` (float) - combined BA excluding suspected mislabels from test fold.
- `stage2_bal_acc_excl` (float) - Stage 2 BA excluding suspected mislabels.
- `n_excluded` (int) - number of mislabel samples in this test fold.
- `en_converged` (bool or null) - whether SAGA converged for EN-containing pipelines.

---

## 5. Statistical Testing

### 5.1 Primary comparison (confirmatory)

**Pipeline 5 (kw_nmc_kens) vs Pipeline 1 (kw_nmc)** - Wilcoxon signed-rank test on per-repeat mean balanced accuracy, alpha=0.05, no correction.

This tests the pre-registered hypothesis that k-ensemble averaging improves over single-k CV selection.

### 5.2 All other pairwise comparisons (exploratory)

C(7, 2) = 21 pairwise Wilcoxon signed-rank tests. Bonferroni correction: alpha/21 = 0.0024. These are exploratory and will be reported as such.

### 5.3 Additional tests

- Friedman test as omnibus test for any differences among 7 pipelines.
- Grouped NMC vs RF test (pooled over feature selectors, only on baseline pipelines 1-4).
- Nadeau-Bengio corrected resampled t-test as the most conservative pairwise test.

---

## 6. Sensitivity Analysis

### 6.1 Mislabel exclusion

For each fold, metrics are computed with and without the suspected mislabel samples (Array.67, Array.22). The analysis script will report:
- Mean and std of `combined_bal_acc_excl` alongside `combined_bal_acc`.
- Whether pipeline rankings change under exclusion.

### 6.2 Probability scale monitoring (pipeline 6)

For the pipeline ensemble (pipeline 6), per-fold logging of:
- Mean max_prob for the k-ensemble component.
- Mean max_prob for the EN+NMC component.

If one component dominates (e.g., mean max_prob > 0.9 vs < 0.6), this is reported as a finding. The pre-registered combination rule (simple averaging) is applied regardless.

---

## 7. Implementation Notes

### 7.1 New runner, not modified v1

The v2 runner (`hierarchical_nested_cv_v2_runner.py`) is a new script, not a modification of the v1 runner. Reasons:
- V1 is stable and produces the baseline results we keep.
- V2 has fundamentally different logic (threshold calibration, k-ensemble dispatch, pipeline ensemble) that would make v1 illegible if mixed in.

### 7.2 Pipeline name reuse

The 4 baseline pipeline names (kw_nmc, kw_rf, en_nmc, en_rf) are reused in v2. They live in a different phase directory (`hierarchical_nested_cv_v2`), so no collision. V2 baselines differ from v1 due to threshold calibration + cost-sensitive NMC.

### 7.3 V1 vs v2 baseline distinction in the paper

- Table 1: v1 flat results (standalone).
- Table 2: v1 hierarchical results (standalone).
- Table 3: v2 hierarchical results (all 7 pipelines).
- Narrative may compare v1 to v2 baselines to show the effect of cost-sensitivity.

---

## 8. Expected Outcomes

Based on v1 results and theoretical reasoning:

1. **Cost-sensitive NMC should help Stage 2:** The 36:32 HR+:TN imbalance is mild, but NMC is particularly sensitive to class imbalance because it uses centroid distances without prior adjustment. Even a small rebalancing should improve TN recall.
2. **Threshold calibration should be degenerate:** Stage 1 was perfect in 245/245 v1 folds. All thresholds should tie, defaulting to 0.5.
3. **K-ensemble should match or slightly beat single-k CV:** Averaging over multiple reasonable k values smooths out variance from the k selection step.
4. **Pipeline ensemble should be the best performer:** v1 error agreement showed KW+NMC and EN+NMC are complementary in their disagreement cases.
5. **All NMC variants should beat RF variants:** Confirmed by v1 (p=0.000148) and expected to hold with cost-sensitivity.

---

*This document was written before any v2 code was implemented or executed. All decisions are fixed and will not be changed based on v2 results.*
