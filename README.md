# TB-Project - Classification Assessment of Tumor Subtypes (CATS)

Breast cancer molecular subtype classification from array CGH (aCGH) copy-number data.

## Overview

This project trains a machine learning classifier to predict three breast cancer molecular subtypes - HER2+, HR+, and Triple Negative - from segmented copy-number profiles (100 training samples, 57 unlabelled validation samples). It investigates whether simple classifiers with univariate feature filtering outperform complex pipelines on discrete aCGH data, following the methodology of Wessels et al. (2005).

## Glossary

| Term | Meaning |
|------|---------|
| **Pipeline** | One of the four sklearn `Pipeline` objects from the 2x2 experimental design. Each combines a feature selector with a classifier. The four pipelines are listed below. |
| **Workflow** | The multi-phase orchestrator (`run_full_workflow.py`) that runs exploration, preprocessing, and comparison phases sequentially under a single named run. |
| **Phase** | A discrete stage of the project (see Phases below). |
| **Run** | A self-contained experiment directory under `results/` (e.g., `2026-04-20_my_experiment/`). All phases write into the same run directory. |
| **2x2 design** | The experimental grid crossing 2 feature selection methods with 2 classifiers, producing 4 pipelines. |
| **Nested CV** | Outer CV estimates generalisation; inner CV tunes hyperparameters. Repeated with multiple seeds. |
| **Region merging** | Label-free preprocessing that collapses adjacent correlated genomic regions (~2834 to ~273 segments). |

### The 4 pipelines

| Pipeline | Feature selection | Classifier |
|----------|------------------|------------|
| `kw_nmc` | Kruskal-Wallis (univariate) | Nearest Mean Centroid (simple) |
| `kw_rf` | Kruskal-Wallis (univariate) | Random Forest (complex) |
| `en_nmc` | Elastic Net (multivariate) | Nearest Mean Centroid (simple) |
| `en_rf` | Elastic Net (multivariate) | Random Forest (complex) |

### Phases

| Phase | Name                             | Script                      |
|-------|----------------------------------|-----------------------------|
| 0     | Exploration                      | `data_exploration_phase.py` |
| 1     | Preprocessing (region merging)   | `preprocessing_phase.py`    |
| -     | Raw vs merged comparison         | `compare_explorations.py`   |
| 2     | Nested CV (runs all 4 pipelines) | `nested_cv_2x2_runner.py`   |
| 3     | CV (runs 1 best pipeline)        | TBD                         |
| 4     | Final prediction                 | TBD                         |

The **workflow** (`run_full_workflow.py`) orchestrates phases 0, 1, 0-merged, and comparison sequentially.

## Quick start

```bash
# Activate the conda environment.
source ~/miniconda3/bin/activate tb_310

# Run the full workflow (phases 0-1 + comparison).
python3 code/run_full_workflow.py --name my_experiment

# Run nested CV locally (all 4 pipelines).
bash code/run_local.sh

# Or submit to SLURM cluster.
bash code/submit_nested_cv.sh

# Analyse nested CV results.
python3 code/analyse_nested_cv.py --name my_experiment
```

## Project structure

```
TB-Project/
  data/                    Input data (gitignored)
  results/                 Named run directories with all outputs
  model/                   Submission model (model.pkl + run_model.py)
  code/                    All scripts and utilities
  docs/                    Phase documentation
  reference_documents/     Assignment instructions and references
  configs/            YAML configs for local and server runs
```

See `CLAUDE.md` for full rules, directory layout, and coding conventions.

## Environment

Python 3.10 via conda (`tb_310`). Key packages: scikit-learn, pandas, numpy, scipy, matplotlib, rich.

## Authors (Group 9)

- Alexandros Michailidis (2903034)
- Antonie Wagner (2903383)
- Christos Botos (2878553)
- Yan Qiao (2874296)

MSc Computer Science and Bioinformatics, VU Amsterdam.
