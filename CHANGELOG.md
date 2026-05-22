# Changelog

All notable changes to pyena will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-01

### Added
- rENA-exact adjacency vector construction (`compute_all_avs`, `vector_to_ut`,
  `ref_window_df`) reverse-engineered from rENA 0.3.1's C++ source.
- Sphere normalization (`sphere_normalize`) and SVD projection (`svd_project`).
- rENA-exact means rotation (`means_rotation`, `orthogonal_svd`) reproducing
  rENA's `ena.rotate.by.mean` at machine epsilon.
- Two-group statistical comparison (`ena_group_comparison`) with Welch's t,
  Mann-Whitney U, and permutation test, plus diagnostic indicators.
- Four reproducibility indicators (`reproducibility_metrics`): centroid
  dispersion, pairwise distance, 95% confidence ellipse area, convex hull area.
- ENA-space visualization (`plot_reproducibility`).
- `ENA` scikit-learn-style estimator orchestrating the full pipeline.
- 101 automated tests covering reference regression, mathematical properties,
  and external cross-checks against scipy and rENA.

### Validation
- Adjacency vectors: 900/900 cells match rENA 0.3.1 (max abs diff = 0).
- SVD coordinates: 180/180 values match rENA (max abs diff = 0).
- Means rotation: 180/180 values match rENA (max abs diff < 4e-16).
