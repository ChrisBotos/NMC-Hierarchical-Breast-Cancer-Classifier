# Phase 1: Preprocessing (Label-Free Region Merging)

Summary of all figures produced by `code/preprocessing_phase.py` and key numerical results.

---

## Figure 01 - Regions per Chromosome Before and After Merging (`01_regions_per_chromosome.png`)

- **X-axis:** Chromosome (1-22, X).
- **Y-axis:** Number of genomic regions.
- **Blue bars:** Raw regions before merging. **Green bars:** Consensus segments after merging.
- **Conclusion:** All chromosomes show substantial reduction. The overall count drops from 2834 raw regions to 273 consensus segments (90.4% reduction). Chromosome 18 has the highest reduction (98.2%, from 56 to 1 segment), while the X chromosome has the lowest (71.6%, from 74 to 21), indicating more heterogeneous CN profiles on chrX.

## Figure 02 - Segment Size Distribution (`02_segment_size_distribution.png`)

- **X-axis:** Number of raw regions per consensus segment.
- **Y-axis:** Number of consensus segments with that size.
- **Conclusion:** 93 segments (34.1%) are singletons (1 raw region) representing genuine CN breakpoints. The median segment contains 3 raw regions, and the largest contains 98. The distribution is right-skewed: most segments are small, but a few large segments correspond to broad CN events spanning many array probes.

---

## Key Numerical Results

| Metric | Value |
|--------|-------|
| Raw regions | 2834 |
| Consensus segments | 273 |
| Dimensionality reduction | 90.4% |
| Pre-merge median adjacent r | 0.925 |
| Post-merge median adjacent r | 0.590 |
| Singleton segments | 93 (34.1%) |
| Bonferroni-significant segments (KW) | 3 / 273 |

---

## Handoff Summary

All output files are in `results/data/preprocessing_phase/`.

| File | Description |
|------|-------------|
| `train_merged.tsv` | 273 segments x 100 samples (CN values are medians of constituent regions) |
| `validation_merged.tsv` | 273 segments x 57 samples (same merge map applied) |
| `merge_map.json` | Maps segment ID (0-272) to constituent raw region row indices |
| `kruskal_wallis_by_pvalue.tsv` | KW test results sorted by p-value ascending |
| `kruskal_wallis_by_genomic_position.tsv` | Same data sorted by genomic position |
| `per_chromosome_merge_summary.tsv` | Per-chromosome before/after region counts |
| `segment_sizes.tsv` | Size of each consensus segment (n_raw_regions) |

See `docs/compare_explorations.md` for the raw-vs-merged comparison figure.
