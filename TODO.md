# TODO.md - TB-Project (CATS)

**Last Updated:** 2026-05-07 12:00

## Critical (P0)

- (none)

## High Priority (P1)

- [ ] `DOC-001` Finalise research paper (5-6 pages, Bioinformatics style). **Owner:** team.
- [ ] `DOC-002` Prepare presentation (4 slides, 8 minutes). **Owner:** team.

## Medium (P2)

- [ ] `SETUP-001` Repository structure alignment to standard conventions. **Owner:** claude. **Due:** 2026-05-07.
- [ ] `TEST-001` Expand test coverage for core utilities (paths, logging, config, CV components). **Owner:** unassigned.
- [ ] `DOC-003` Update CLAUDE.md to reference new directory layout (configs/, interesting_results/, tests/). **Owner:** unassigned.

## Low (P3)

- [ ] `REFACTOR-001` Add `merge_cli_overrides()` to config_loader for CLI/YAML config merging. **Owner:** unassigned.
- [ ] `REFACTOR-002` Add git hash logging to `setup_logging()` first log message. **Owner:** unassigned.
- [ ] `REFACTOR-003` Add `generate_run_report()` utility for auto-generated run summaries. **Owner:** unassigned.
- [ ] `TEST-002` Add tests for CV components (KruskalWallisSelector, ElasticNetSelector, NearestCentroidWithProba). **Owner:** unassigned.
- [ ] `SETUP-002` Install and configure pre-commit hooks in dev environment. **Owner:** unassigned.

## Completed

- [x] `SETUP-001` Repository structure alignment (directories, tooling files, tracking). **Done:** 2026-05-07. **Artifacts:** pyproject.toml, Makefile, .pre-commit-config.yaml, GLOSSARY.md, CITATIONS.yaml, TODO.md, STATUS.md, tests/, interesting_results/.
