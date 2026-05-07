# GLOSSARY.md - TB-Project (CATS)

**Last Updated:** 2026-05-07

## Parameters

| Term | Definition | Units | Code Variable | Notes |
|------|-----------|-------|---------------|-------|
| Random seed | Global seed for reproducibility | - | `RANDOM_SEED` | Fixed at 42 |
| n_repeats | Number of repeated CV iterations | - | `n_repeats` | 50 for server runs, 3 for local |
| n_outer_folds | Number of outer CV folds | - | `n_outer_folds` | 5 |
| n_inner_folds | Number of inner CV folds | - | `n_inner_folds` | 5 |
| Merging threshold | Pearson r threshold for region merging | - | `r_threshold` | 0.8 |
| k (KW) | Number of top features selected by Kruskal-Wallis | - | `k` | Tuned via inner CV |
| Stage 1 k | Fixed feature count for Stage 1 KW selector | - | - | Fixed at 5 |
| figure_dpi | Resolution for saved figures | DPI | `dpi` | >= 300 |
| max_plateau_size | Maximum models in a plateau ensemble | - | `MAX_PLATEAU_SIZE` | 15 |

## Derived Quantities

| Quantity | Definition | Formula / Source |
|----------|-----------|-----------------|
| Balanced accuracy (BA) | Mean per-class recall | `sklearn.metrics.balanced_accuracy_score` |
| BA2 | Stage 2 balanced accuracy (HR+ vs TN only) | `stage2_bal_acc` in fold results |
| Combined BA | 3-class BA across the full hierarchical classifier | Weighted by samples routed to each stage |
| Silhouette score | Mean silhouette coefficient for clustering quality | `sklearn.metrics.silhouette_score` |

## Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| aCGH | Array comparative genomic hybridisation |
| BA | Balanced accuracy |
| BA2 | Stage 2 balanced accuracy |
| CATS | Classification Assessment of Tumor Subtypes |
| CN | Copy number |
| CV | Cross-validation |
| EN | Elastic Net (multivariate feature selector) |
| ER | Estrogen receptor |
| HER2 | Human epidermal growth factor receptor 2 |
| HR | Hormone receptor |
| KW | Kruskal-Wallis (univariate feature selector) |
| NMC | Nearest Mean Centroid (classifier) |
| PCA | Principal component analysis |
| PR | Progesterone receptor |
| RF | Random Forest (classifier) |
| TN | Triple Negative |
| t-SNE | t-distributed stochastic neighbour embedding |

## Conventions

| Convention | Description |
|-----------|-------------|
| Pipeline | One of the four sklearn `Pipeline` objects from the 2x2 design: `kw_nmc`, `kw_rf`, `en_nmc`, `en_rf`. Each combines a feature selector with a classifier. Pipeline labels refer to Stage 2 only in the hierarchical context. |
| Workflow | The full multi-phase orchestrator (`run_full_workflow.py`) that runs exploration, preprocessing, and comparison phases sequentially under a single named run. |
| Phase | A discrete stage of the project (Phase 0: exploration, Phase 1: preprocessing, Phase 2-3: nested CV, Phase 4: final prediction). |
| Run | A self-contained experiment directory under `results/` (e.g., `2026-04-24_server_run_v2/`). All phases write into the same run directory. |
| 2x2 design | The experimental grid crossing two feature selection methods (KW univariate, EN multivariate) with two classifiers (NMC, RF), producing 4 pipelines. |
| Nested CV | Outer CV estimates generalisation performance; inner CV tunes hyperparameters. Repeated with multiple seeds for stability. |
| Region merging | Label-free preprocessing that collapses spatially adjacent, highly correlated (r > 0.8) genomic regions into consensus segments (~2834 to ~273). |
| Hierarchical classifier | Two-stage architecture: Stage 1 (HER2+ vs rest, fixed KW+RF k=5) then Stage 2 (HR+ vs TN, variable pipeline). |
| Plateau ensemble | Post-hoc ensemble that pools models within 1 SE of the best inner CV score, averaging their predictions. |
| Date prefix | Run directories are prefixed with `YYYY-MM-DD` when created. |
| Phase replacement | If a phase subdirectory already exists within a run, it is deleted and recreated. |
