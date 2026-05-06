# Figure and Paper Glossary

Canonical definitions for all metrics and pipeline names used in figures, tables, and text.

## Metrics

| Symbol | Full Name | Definition | Notes |
|--------|-----------|------------|-------|
| **BA** | Combined balanced accuracy | 3-class BA across HER2+, HR+, TN (hierarchical pipeline). Mean of per-class recalls. | Inflated because BA1 = 1.0; use for fair comparison with flat BA. |
| **Flat BA** | Flat balanced accuracy | 3-class BA from the flat (non-hierarchical) experiment. | Directly comparable to BA. |
| **BA1** | Stage 1 balanced accuracy | BA on the HER2+ vs rest binary problem. | Always 1.0 in all tested folds. Never write "BA = 1.0" - always "BA1 = 1.0". |
| **BA2** | Stage 2 balanced accuracy | BA on the HR+ vs TN binary problem (Stage 2 only). | Primary metric for comparing hierarchical pipelines (since BA1 is constant). |
| **AUROC** | Macro-averaged AUROC | Area under ROC, macro-averaged across classes. | |

## Pipeline Names

### Flat experiment (3-class, single-stage)

| Code | Label | Feature selection | Classifier |
|------|-------|-------------------|------------|
| `kw_nmc` | KW+NMC | Kruskal-Wallis | Nearest Mean Centroid |
| `kw_rf` | KW+RF | Kruskal-Wallis | Random Forest |
| `en_nmc` | EN+NMC | Elastic Net | Nearest Mean Centroid |
| `en_rf` | EN+RF | Elastic Net | Random Forest |

### Hierarchical experiment (Stage 2 only - Stage 1 is always fixed KW+RF, k=5)

| Code | Label | Stage 2 FS | Stage 2 Classifier | Variant |
|------|-------|------------|-------------------|---------|
| `kw_nmc` | KW+NMC | Kruskal-Wallis | NMC | Base |
| `kw_rf` | KW+RF | Kruskal-Wallis | RF | Base |
| `en_nmc` | EN+NMC | Elastic Net | NMC | Base |
| `en_rf` | EN+RF | Elastic Net | RF | Base |
| `kw_nmc_pens` | KW+NMC(P) | Kruskal-Wallis | NMC plateau ensemble | Plateau |
| `en_nmc_pens` | EN+NMC(P) | Elastic Net | NMC plateau ensemble | Plateau |
| `nmc_ensemble` | Ensemble | Combined | NMC ensemble (kens+en_nmc) | Ensemble |

### Excluded from figures

| Code | Reason |
|------|--------|
| `kw_nmc_kens` | Internal variant, not reported. |
| `kw_nmc_kgrid` | Grid search variant, not reported. |
| `standalone_en` | Standalone EN baseline, not in main comparison. |
| `standalone_en_pens` | Standalone EN plateau, not in main comparison. |

## Column Mapping

| CSV column (hierarchical) | CSV column (flat) | Glossary term |
|---------------------------|-------------------|---------------|
| `combined_bal_acc` | `balanced_accuracy` | BA / Flat BA |
| `stage1_bal_acc` | -- | BA1 |
| `stage2_bal_acc` | -- | BA2 |
| `stage2_n_features` | `n_features_selected` | Feature count |

## Key Rules

1. **Never write "BA = 1.0"** - always "BA1 = 1.0" to distinguish from combined BA.
2. **Flat BA vs BA** is the fair cross-experiment comparison (both are 3-class).
3. **BA2** is the primary metric for comparing hierarchical pipelines to each other.
4. Pipeline labels (KW+NMC etc.) refer to **Stage 2 only** in the hierarchical experiment.
