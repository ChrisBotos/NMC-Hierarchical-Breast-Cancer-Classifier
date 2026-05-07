# STATUS.md - TB-Project (CATS)

**Last Updated:** 2026-05-07 13:00
**Current Phase:** Final deliverables preparation (paper, presentation).

## Active Work

| Work ID | Task ID | Description | Agent | Started | Blocking |
|---------|---------|-------------|-------|---------|----------|

## Recently Completed

| Work ID | Task ID | Description | Completed | Outcome |
|---------|---------|-------------|-----------|---------|
| W-001 | SETUP-001 | Repository structure alignment to standard conventions | 2026-05-07 | pyproject.toml, Makefile, .pre-commit-config.yaml, GLOSSARY.md, CITATIONS.yaml, TODO.md, STATUS.md, tests/ (33 passing), interesting_results/findings.md, configs/ rename. All 33 tests pass. |
| - | - | Hierarchical nested CV (200 repeats, 10 pipelines) | 2026-04-25 | Results in `results/2026-04-25_final_hierarchical/hierarchical_nested_cv/` |
| - | - | Final model training and prediction generation | 2026-04-25 | `results/prediction.txt`, `model/model.pkl` |
| - | - | Draft report figures and tables | 2026-04-25 | `results/2026-04-24_server_run_v2/hierarchical_nested_cv/figures/` |
| - | - | Analysis of nested CV results (stats, plots) | 2026-04-25 | `results/2026-04-24_server_run_v2/hierarchical_nested_cv/` |

## Blocked Items

| Item | Blocked By | Since | Impact |
|------|-----------|-------|--------|
| - | - | - | - |

## System Health

- Last test run: 2026-05-07 - PASS (33/33 tests).
- Known issues: None critical. Pipeline runs successfully end-to-end.
- Environment: conda `tb_310`, Python 3.10, all dependencies installed.

## Agent Coordination Notes

- 2026-05-07: Repository standardisation completed. Changes: config_files/ renamed to configs/, all references updated across 10+ files. docs/findings.md migrated to interesting_results/findings.md with standard schema. New files: pyproject.toml, Makefile, .pre-commit-config.yaml, GLOSSARY.md, CITATIONS.yaml, TODO.md, STATUS.md. New directories: tests/, scripts/, notebooks/, interesting_results/, data/raw|interim|processed|external/. Code/utils enhanced: paths.py (12 new constants), logging_setup.py (git hash, verbose, handler cleanup), config_loader.py (JSON support, merge_cli_overrides). All existing scripts work unchanged.
