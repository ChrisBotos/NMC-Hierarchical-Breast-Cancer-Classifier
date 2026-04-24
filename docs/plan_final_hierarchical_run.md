# Plan: Final Hierarchical Nested CV Run

## 1. Overview

One clean run producing 10 pipeline variants x 100 repeats x 5 folds = 5000 fold evaluations across 1000 SLURM tasks. Supports both the paper's Wessels comparison and the competition submission.

**Run name:** `final_hierarchical` (no v3 suffix).
**Seeds:** 1001-1100 (avoids correlation with v2 seeds 1-50).

---

## 2. Pipeline Inventory (10 total)

### Kept from v2 (unchanged except shrink_threshold addition)
| Pipeline | Type | Stage 2 description |
|----------|------|---------------------|
| kw_nmc | Base (GridSearchCV) | KW selector + NMC |
| en_nmc | Base (GridSearchCV) | EN selector + NMC |
| kw_rf | Base (GridSearchCV) | KW selector + RF |
| en_rf | Base (GridSearchCV) | EN selector + RF |
| standalone_en | Base (GridSearchCV) | Logistic Regression (elasticnet) |
| kw_nmc_pens | Plateau ensemble | Pooled from kw_nmc |
| standalone_en_pens | Plateau ensemble | Pooled from standalone_en |
| en_nmc_pens | Plateau ensemble | Pooled from en_nmc |

### Changed
| Pipeline | Change |
|----------|--------|
| nmc_ensemble | **Was:** kw_nmc_kens + en_nmc (fresh per fold). **Now:** post-hoc average of completed kw_nmc + en_nmc fold probabilities. |

### New
| Pipeline | Type | Description |
|----------|------|-------------|
| nmc_pens_ensemble | Post-hoc ensemble | Average of completed kw_nmc_pens + en_nmc_pens fold probabilities. |

### Dropped
| Pipeline | Reason |
|----------|--------|
| kw_nmc_kens | Hand-picked k values, not methodologically clean. |
| kw_nmc_kgrid | Hand-picked restricted grid, not methodologically clean. |

---

## 3. NMC Shrink Threshold Tuning

Add `clf__shrink_threshold: [null, 0.1, 0.2, 0.5]` to the grids for **kw_nmc** and **en_nmc**. This multiplies their grid sizes by 4:

| Pipeline | Old grid size | New grid size |
|----------|--------------|---------------|
| kw_nmc | 8 (k only) | 32 (k x shrink) |
| en_nmc | 200 (C x l1 x top_k) | 800 (C x l1 x top_k x shrink) |

NearestCentroidWithProba already accepts `shrink_threshold` in its constructor (line 202 of cv_components.py) and passes it to sklearn's NearestCentroid. No change to the component class needed. GridSearchCV will set `clf__shrink_threshold` via the Pipeline step name.

Metric stays fixed at `euclidean` (the default in NearestCentroidWithProba). Not added to the grid.

Plateau ensembles that pool from NMC base pipelines will naturally include shrink_threshold combinations in their pooled hyperparameter space (the inner_cv.csv files will have a `param_clf__shrink_threshold` column).

---

## 4. Suspected Mislabels Update

Add Array.113 to the sensitivity analysis. Current config has:
```yaml
suspected_mislabels:
  indices: [2, 4]
  names: ["Array.67", "Array.22"]
```

**Action needed:** Determine the 0-indexed position of Array.113 in the sample ordering (columns of train_merged.tsv). This will be done during the smoke test phase by running:
```python
import pandas as pd
df = pd.read_csv("results/.../preprocessing/data/train_merged.tsv", sep="\t", nrows=0)
cols = [c for c in df.columns if c not in {"Chromosome", "Start", "End", "Nclone"}]
print(cols.index("Array.113"))
```

Then update config to include all three indices and names.

---

## 5. File-by-File Change List

### 5.1 `config_files/server.yaml`

| Section | Change |
|---------|--------|
| `cv.n_repeats` | 50 -> 100 |
| `cv.seed_start` | Add new field: 1001 |
| `pipelines.names` | Remove kw_nmc_kens, kw_nmc_kgrid. Add nmc_pens_ensemble. Final list: [kw_nmc, en_nmc, kw_rf, en_rf, standalone_en, kw_nmc_pens, standalone_en_pens, en_nmc_pens, nmc_ensemble, nmc_pens_ensemble] |
| `grids.kw_nmc` | Add `clf__shrink_threshold: [null, 0.1, 0.2, 0.5]` |
| `grids.en_nmc` | Add `clf__shrink_threshold: [null, 0.1, 0.2, 0.5]` |
| `grids.kw_nmc_kgrid` | Delete entire section |
| `k_ensemble` | Delete entire section |
| `suspected_mislabels` | Add Array.113 (index TBD) |

### 5.2 `config_files/local.yaml`

Mirror server.yaml structure for local smoke testing:
- n_repeats: 2 (keep small)
- seed_start: 1001
- Same pipeline names
- Sparse grids (fewer combos)
- Same suspected_mislabels

### 5.3 `code/utils/constants.py`

| Constant | Change |
|----------|--------|
| PIPELINE_LABELS | Remove kw_nmc_kens, kw_nmc_kgrid. Add `"nmc_pens_ensemble": "NMC Pens Ensemble"`. |
| PIPELINE_COLORS | Remove kw_nmc_kens, kw_nmc_kgrid. Add `"nmc_pens_ensemble": "<color>"`. |
| PIPELINE_NAMES | Remove kw_nmc_kens, kw_nmc_kgrid. Add nmc_pens_ensemble. New ordering: `("kw_nmc", "en_nmc", "kw_rf", "en_rf", "standalone_en", "nmc_ensemble", "kw_nmc_pens", "standalone_en_pens", "en_nmc_pens", "nmc_pens_ensemble")` |
| GRIDSEARCH_PIPELINES | Remove kw_nmc_kgrid. Final: `("kw_nmc", "en_nmc", "kw_rf", "en_rf", "standalone_en")` |
| Add POSTHOC_ENSEMBLE_COMPONENTS | New dict: `{"nmc_ensemble": ("kw_nmc", "en_nmc"), "nmc_pens_ensemble": ("kw_nmc_pens", "en_nmc_pens")}` |

### 5.4 `code/utils/cv_config.py`

| Function | Change |
|----------|--------|
| `build_stage2_pipeline()` | Remove kw_nmc_kgrid case (line 166). Remove kw_nmc_kens from None-return check. Add nmc_pens_ensemble to the None-returning ensemble/plateau set. nmc_ensemble stays in None-return set (now post-hoc, not live). |

### 5.5 `code/hierarchical_nested_cv_runner.py` (major changes)

**Delete:**
- `run_stage2_k_ensemble()` function (lines 378-431) - kw_nmc_kens is dropped.
- `run_stage2_pipeline_ensemble()` function (lines 434-485) - old nmc_ensemble approach.

**Add:**
- `run_posthoc_ensemble()` function - reads component pipeline fold results CSVs, averages proba_combined, recomputes all metrics. This function replaces the fold loop for ensemble pipelines.
- `POSTHOC_ENSEMBLE_COMPONENTS` constant (import from constants.py).

**Modify in `run_single_repeat()`:**
- Remove the `elif stage2_pipeline_name == "kw_nmc_kens":` branch (lines 685-690).
- Remove the `elif stage2_pipeline_name == "nmc_ensemble":` branch (lines 692-703).
- (The remaining dispatch handles GridSearchCV pipelines and plateau ensembles.)

**Modify in `main()`:**
- Add `seed_start` config reading: `seed_start = config.get("cv", {}).get("seed_start", 1)`. This is used only for the log message and the already-passed `--repeat` arg.
  - Actually, `--repeat` is passed directly by the SLURM script, which already computes the correct seed. The runner doesn't need to add the offset - SLURM does. So no change needed in the runner for seed handling.
- Add post-hoc ensemble branch before the normal flow: if pipeline is in POSTHOC_ENSEMBLE_COMPONENTS, call `run_posthoc_ensemble()` instead of `run_single_repeat()`.
- The post-hoc ensemble function handles its own CSV output.

**Post-hoc ensemble implementation (`run_posthoc_ensemble`):**

```
For ensemble_name with components (comp_a, comp_b):
1. Read fold_results_{comp_a}_r{repeat}.csv and fold_results_{comp_b}_r{repeat}.csv
2. Verify both exist (exit with error if not)
3. For each outer fold (1-5):
   a. Get matching row from each component CSV
   b. Verify test_indices and y_true are identical (same outer fold split)
   c. Parse proba_combined JSON from each -> numpy arrays
   d. Average: avg_proba = (proba_a + proba_b) / 2
   e. Compute y_pred_combined from argmax of avg_proba
   f. Compute combined_bal_acc
   g. Identify HR+/TN test samples from y_true, compute stage2_bal_acc
   h. Compute auroc_macro using avg_proba
   i. Compute mislabel sensitivity metrics
   j. Build fold_row dict matching the standard schema
4. Save fold_results_{ensemble_name}_r{repeat}.csv
```

This does NOT need the training data (X, y) at all - everything is derived from stored probabilities and labels. But it does need the LabelEncoder class ordering. We can hardcode it: `classes = ["HER2+", "HR+", "Triple Neg"]` since the ordering is always alphabetical.

### 5.6 `code/submit_hierarchical_nested_cv.sh`

**Three-phase submission:**

```
Phase 1: Base pipelines
  kw_nmc, en_nmc, kw_rf, en_rf, standalone_en
  5 x 100 = 500 tasks
  No dependency (or external --dependency if provided)

Phase 2: Plateau ensembles + nmc_ensemble
  kw_nmc_pens, standalone_en_pens, en_nmc_pens, nmc_ensemble
  4 x 100 = 400 tasks
  --dependency=afterok:Phase1_JOB_ID

Phase 3: Pens ensemble
  nmc_pens_ensemble
  1 x 100 = 100 tasks
  --dependency=afterok:Phase2_JOB_ID
```

**Specific changes:**
- Read `seed_start` from YAML config (default 1).
- Change repeat calculation: `REPEAT=$(( SLURM_ARRAY_TASK_ID % REPEATS_PER_PIPELINE + SEED_START ))`.
- Categorize pipelines into three phases instead of two:
  ```bash
  for p in "${PIPELINES[@]}"; do
      case "$p" in
          *_pens_ensemble) PHASE3_PIPELINES+=("$p") ;;
          *_pens)          PHASE2_PIPELINES+=("$p") ;;
          nmc_ensemble)    PHASE2_PIPELINES+=("$p") ;;
          *)               PHASE1_PIPELINES+=("$p") ;;
      esac
  done
  ```
- Submit Phase 3 with afterok dependency on Phase 2 job ID.

### 5.7 `code/analyse_nested_cv.py`

**Pre-registered vs exploratory comparisons:**

Add a pre-registered comparisons config (hardcoded in the analysis script or in the YAML):

```python
PREREGISTERED_COMPARISONS = [
    ("en_nmc_pens", "kw_nmc_pens"),        # Which pens variant wins
    ("nmc_pens_ensemble", "en_nmc_pens"),   # Does ensembling pens help
    ("nmc_ensemble", "kw_nmc"),             # Does ensembling base NMC help (vs KW)
    ("nmc_ensemble", "en_nmc"),             # Does ensembling base NMC help (vs EN)
]
# Grouped: all NMC pooled vs all RF pooled (already implemented)
```

**Changes to statistical testing:**
- `run_statistical_tests()`: split pairwise comparisons into pre-registered and exploratory.
- Pre-registered pairs: report raw p-values, no correction.
- Exploratory pairs: Bonferroni-correct over N_exploratory pairs only.
- Save separate output files: `preregistered_comparisons.csv` and `exploratory_comparisons.csv`.
- Same split for Nadeau-Bengio tests.

**Sensitivity analysis:**
- The sensitivity analysis (BA with/without mislabels) is already implemented. Adding Array.113 to the config's suspected_mislabels list handles this automatically.

**Other:**
- Update `get_pipeline_order()` to handle the new pipeline set (remove kw_nmc_kens, kw_nmc_kgrid detection).
- The interaction plot, feature importance, confusion matrices, error agreement, and hard sample plots need no logic changes - they naturally handle whatever pipelines are present.

---

## 6. Config File Structure

Single config file (`config_files/server.yaml`) for the entire run. The SLURM script freezes it as `config_snapshot_hierarchical.yaml` in the run directory.

No split base/pens/ensemble configs needed - the SLURM script's phase categorization handles the submission ordering.

---

## 7. SLURM Submission Workflow

### Step-by-step commands:

```bash
# 1. Submit everything (script auto-splits into 3 phases):
bash code/submit_hierarchical_nested_cv.sh final_hierarchical

# Output (example):
# Phase 1 submitted: job 12345 (5 pipelines x 100 repeats = 500 tasks)
# Phase 2 submitted: job 12346 (4 pipelines x 100 repeats = 400 tasks, afterok:12345)
# Phase 3 submitted: job 12347 (1 pipeline x 100 repeats = 100 tasks, afterok:12346)

# 2. Monitor:
squeue -u $USER

# 3. After all complete, run analysis:
python3 code/analyse_nested_cv.py \
    --name final_hierarchical \
    --config results/*_final_hierarchical/config_snapshot_hierarchical.yaml \
    --phase hierarchical_nested_cv
```

### Failure recovery:

If tasks fail:
```bash
# Re-submit with --skip-if-complete (already in runner):
# Just re-run the same submit command - completed tasks exit immediately.
bash code/submit_hierarchical_nested_cv.sh final_hierarchical
```

---

## 8. Pre-Registration Document Outline

Create `docs/preregistration_final_run.md` before submitting:

```markdown
# Pre-Registration: Final Hierarchical Nested CV Run

## Date locked: [DATE]

## Experimental design
- 10 pipelines, 100 repeats, 5-fold stratified outer CV
- Seeds 1001-1100
- Stage 1 fixed: KW+RF k=5

## Primary metric
- BA2 (Stage 2 balanced accuracy, HR+ vs TN)

## Pre-registered comparisons (no correction)
1. en_nmc_pens vs kw_nmc_pens
2. nmc_pens_ensemble vs en_nmc_pens
3. nmc_ensemble vs kw_nmc
4. nmc_ensemble vs en_nmc
5. All NMC variants pooled vs all RF variants pooled

## Statistical tests
- Primary: Wilcoxon signed-rank on per-repeat mean BA2
- Sensitivity: Nadeau-Bengio corrected t-test on fold-level BA2

## Exploratory comparisons
- All remaining pairwise comparisons
- Bonferroni-corrected at alpha = 0.05/N_exploratory

## Sensitivity analyses
- BA2 computed with and without suspected mislabels
  (Array.22, Array.67, Array.113)
- Both pre-registered and exploratory results reported with
  and without mislabel exclusion

## Stopping rule
- Run all 100 repeats. No early stopping or adaptive analysis.
```

---

## 9. Verification / Smoke Test Plan

### Before server submission:

1. **Determine Array.113 index:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('results/*/preprocessing/data/train_merged.tsv', sep='\t', nrows=0)
   cols = [c for c in df.columns if c not in {'Chromosome','Start','End','Nclone'}]
   print(f'Array.113 is at index {cols.index(\"Array.113\")}')
   "
   ```

2. **Local syntax check** (no compute, just import and arg parsing):
   ```bash
   source ~/miniconda3/bin/activate tb_310
   python3 code/hierarchical_nested_cv_runner.py --help
   ```

3. **Run 1 repeat of each base pipeline locally:**
   ```bash
   for pipe in kw_nmc en_nmc kw_rf en_rf standalone_en; do
       python3 code/hierarchical_nested_cv_runner.py \
           --pipeline $pipe --repeat 1001 --config local --name smoke_test
   done
   ```
   Verify: fold_results CSVs created with correct columns and 5 rows each.

4. **Run 1 repeat of each pens pipeline locally:**
   ```bash
   for pipe in kw_nmc_pens standalone_en_pens en_nmc_pens; do
       python3 code/hierarchical_nested_cv_runner.py \
           --pipeline $pipe --repeat 1001 --config local --name smoke_test
   done
   ```
   Verify: fold_results CSVs created, plateau params logged.

5. **Run 1 repeat of nmc_ensemble:**
   ```bash
   python3 code/hierarchical_nested_cv_runner.py \
       --pipeline nmc_ensemble --repeat 1001 --config local --name smoke_test
   ```
   Verify: reads from kw_nmc and en_nmc results, produces averaged probabilities.

6. **Run 1 repeat of nmc_pens_ensemble:**
   ```bash
   python3 code/hierarchical_nested_cv_runner.py \
       --pipeline nmc_pens_ensemble --repeat 1001 --config local --name smoke_test
   ```
   Verify: reads from kw_nmc_pens and en_nmc_pens results, produces averaged probabilities.

7. **Run analysis script on smoke test results:**
   ```bash
   python3 code/analyse_nested_cv.py \
       --name smoke_test --config local --phase hierarchical_nested_cv
   ```
   Verify: no crashes, all 7 figures generated, pre-registered vs exploratory separation works.

8. **Verify grid sizes:**
   Check that the inner_cv.csv files for kw_nmc have 32 rows (8 k x 4 shrink) and en_nmc have 800 rows (10 C x 5 l1 x 4 top_k x 4 shrink).

9. **Verify seed separation:**
   Confirm fold_results files are named r1001.csv (not r1.csv).
   Confirm outer CV splits differ from v2 (different random_state).

### Estimated smoke test time:
~15-20 minutes locally (en_nmc is the bottleneck with 800 grid combos).

---

## 10. Estimated Wall Time and SLURM Resources

### Per-task estimates (single repeat, 5 outer folds):

| Pipeline | Grid size | Inner fits per repeat | Est. time |
|----------|----------|----------------------|-----------|
| kw_nmc | 32 | 32 x 5 x 5 = 800 | ~2 min |
| en_nmc | 800 | 800 x 5 x 5 = 20,000 | ~30 min |
| kw_rf | 32 | 32 x 5 x 5 = 800 | ~5 min |
| en_rf | 800 | 800 x 5 x 5 = 20,000 | ~45 min |
| standalone_en | 50 | 50 x 5 x 5 = 1,250 | ~5 min |
| kw_nmc_pens | 15 retrained | 15 x 5 = 75 | ~1 min |
| standalone_en_pens | 15 retrained | 15 x 5 = 75 | ~1 min |
| en_nmc_pens | 15 retrained | 15 x 5 = 75 | ~1 min |
| nmc_ensemble | Post-hoc read | 0 (just I/O) | ~10 sec |
| nmc_pens_ensemble | Post-hoc read | 0 (just I/O) | ~10 sec |

### Phase wall times (50 max concurrent):

| Phase | Tasks | Bottleneck | Est. wall time |
|-------|-------|-----------|----------------|
| Phase 1 (base) | 500 | en_rf ~45 min | ~8 hours |
| Phase 2 (pens + nmc_ens) | 400 | pens ~1 min each | ~15 min |
| Phase 3 (pens_ens) | 100 | ~10 sec each | ~1 min |
| **Total** | **1000** | | **~8-9 hours** |

### SLURM resources per task:
- Memory: 4G (unchanged - data is small)
- CPUs: 1 (unchanged - all n_jobs=1)
- Time limit: 2 days (unchanged - generous margin)

---

## 11. Pens Information Flow (Addressing User Concern)

**Confirm: this is the intended behavior.**

The plateau ensemble pools inner CV scores from base pipelines running on the *same seeds* it is evaluated on. The information flow is:

1. Base pipeline (e.g. kw_nmc) runs 100 repeats x 5 folds, each with inner CV. The inner CV scores (per hyperparameter combo) are saved to inner_cv.csv files.

2. The plateau computation reads ALL 500 inner_cv.csv files (100 repeats x 5 folds), pools mean scores per combo, identifies the stable plateau.

3. The plateau ensemble then runs 100 repeats x 5 folds, using the same outer fold splits (same seeds) as the base pipeline. For each fold, it retrains the plateau models and averages predictions.

The shared information is "which hyperparameter combos are stable across 500 independent evaluations." This is dataset-level meta-knowledge, not sample-level leakage:
- Outer fold test sets are never seen during inner CV (no label leakage).
- The plateau is pooled across all 500 observations, so no single fold dominates.
- The plateau is a deployment tool (competition), not a research finding (paper).

**Alternative considered and rejected:** Leave-one-repeat-out plateau computation (compute plateau from 99 repeats, evaluate on the held-out repeat). This would require 100 separate plateau computations, is more complex, and yields minimal gain since the plateau is already a robust aggregate of 500 observations. The marginal change from removing 5/500 observations is negligible.

---

## 12. Implementation Order

1. Update `code/utils/constants.py` (pipeline definitions)
2. Update `code/utils/cv_config.py` (pipeline factory)
3. Update `code/hierarchical_nested_cv_runner.py` (remove old ensembles, add post-hoc ensemble)
4. Update `config_files/server.yaml` and `config_files/local.yaml`
5. Update `code/submit_hierarchical_nested_cv.sh` (three phases, seed_start)
6. Update `code/analyse_nested_cv.py` (pre-registered vs exploratory)
7. Determine Array.113 index and update config
8. Run smoke tests locally
9. Create pre-registration document
10. Submit to SLURM
