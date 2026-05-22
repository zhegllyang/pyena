# pyena

[![PyPI](https://img.shields.io/pypi/v/pyena.svg)](https://pypi.org/project/pyena/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-101%20passing-brightgreen.svg)](#testing)
[![DOI](https://zenodo.org/badge/1246415240.svg)](https://doi.org/10.5281/zenodo.20339527)

**pyena** is a Python implementation of rENA-exact Epistemic Network Analysis
(ENA), numerically validated bit-for-bit against the reference R package
[`rENA`](https://cran.r-project.org/package=rENA) 0.3.1.

## Features

- **rENA-exact adjacency vectors** — reverse-engineered from rENA's C++ source
  (`vector_to_ut` + `ref_window_df`). Verified at 900/900 cells, max abs diff = 0.
- **SVD projection + means rotation** — both stages match rENA outputs at
  machine epsilon (180/180 values each).
- **Group comparison** — Welch's t, Mann-Whitney U, and permutation test in one
  call, with automatic small-sample diagnostics.
- **Reproducibility metrics** — four indicators (centroid dispersion, pairwise
  distance, 95% confidence ellipse area, convex hull area) designed for LLM
  dialogue-path analysis.
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
| Adjacency vectors  | 90 units × 1800 utterances × 5 codes        | 900 / 900       | 0            |
| SVD coordinates    | same                                        | 180 / 180       | 0            |
| Means rotation     | same                                        | 180 / 180       | < 4e-16      |

See `tests/test_adjacency.py`, `tests/test_projection.py` for the regression
tests, and `notebooks/02_rena_comparison.ipynb` for the original validation
workflow against rENA via rpy2.

## Testing

```bash
pytest
```

101 tests cover reference regression, mathematical properties of each
algorithm, and external cross-checks against scipy (Welch, Mann-Whitney) and
the rENA reference outputs.

## Citation

If you use pyena in your research, please cite:

```bibtex
@software{song_pyena_2026,
  author    = {Song, JongHwi},
  title     = {{pyena: Python implementation of rENA-exact Epistemic Network Analysis}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.2},
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
