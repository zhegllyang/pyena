# pyena

[![PyPI](https://img.shields.io/pypi/v/pyena.svg)](https://pypi.org/project/pyena/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)](#testing)
[![DOI](https://zenodo.org/badge/1246415240.svg)](https://doi.org/10.5281/zenodo.20339527)

**pyena** is a Python implementation of Epistemic Network Analysis (ENA),
numerically verified against the reference R package
[`rENA`](https://cran.r-project.org/package=rENA) 0.3.1.

## Features

- **Verified adjacency vectors** — reverse-engineered from rENA's C++ source
  (`vector_to_ut` + `ref_window_df`), using the cross-speaker moving stanza window. Verified exactly against rENA on nine datasets (RS.data + eight synthetic).
- **SVD projection + means rotation** — both stages match rENA outputs at
  machine epsilon on RS.data (RMSE < 1e-16).
- **Group comparison** — Welch's t, Mann-Whitney U, and permutation test in one
  call, with automatic small-sample diagnostics.
- **Reproducibility metrics** — four indicators (centroid dispersion, pairwise
  distance, 95% confidence ellipse area, convex hull area) for quantifying the
  spread of repeated analyses under nominally identical conditions.
- **scikit-learn-style API** — single `ENA` estimator that orchestrates the
  full pipeline.

## Installation

```bash
pip install pyena
```

Requires Python 3.10+.

### Development install

```bash
git clone https://github.com/zhegllyang/pyena.git
cd pyena
pip install -e ".[dev]"
```

## Quickstart

```python
import pandas as pd
from pyena import ENA

df = pd.read_csv("your_long_format_utterances.csv")

ena = ENA(
    codes=["Data", "Theory", "Question", "Example", "Critique"],
    unit_col="unit",
    conversation_col="conversation",
    window_size=4,
    rotation="means",
    mr_groups=("A", "B"),
)
ena.fit(df)

ena.plot()
result = ena.compare(axis="x")
metrics = ena.reproducibility()
```

## Validation against rENA 0.3.1

| Stage              | Test data                                   | Match           | Max abs diff |
|--------------------|---------------------------------------------|-----------------|--------------|
| Adjacency vectors  | nine datasets (RS.data + 8 synthetic)       | all cells       | 0            |
| SVD coordinates    | RS.data (48 units × 2 axes)                 | 96 / 96         | < 1e-15      |
| Means rotation     | RS.data (48 units × 2 axes)                 | 96 / 96         | < 1e-15      |
| Node placement     | RS.data (6 codes × 2 axes)                  | 12 / 12         | < 1e-13      |

See `tests/test_adjacency.py`, `tests/test_projection.py`, and
`tests/test_crossspeaker.py` for the regression tests, and the R scripts in
`validation_rsdata/` for regenerating the rENA reference outputs.

## Testing

```bash
pytest
```

108 tests cover reference regression, mathematical properties of each
algorithm, and external cross-checks against scipy (Welch, Mann-Whitney) and
the rENA reference outputs.

## Citation

If you use pyena in your research, please cite:

```bibtex
@software{song_pyena_2026,
  author    = {Song, JongHwi},
  title     = {{pyena: Python implementation of Epistemic Network Analysis, verified against rENA}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.2.0},
  doi       = {10.5281/zenodo.20339527},
  url       = {https://github.com/zhegllyang/pyena}
}
```

A SoftwareX paper describing pyena is in preparation.

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Acknowledgements

pyena reimplements the algorithms of [rENA](https://cran.r-project.org/package=rENA)
0.3.1 by Marquart, Swiecki, Collier, Eagan, Woodward, and Shaffer. The
reimplementation was done independently from the publicly available C++ and R
source code.
