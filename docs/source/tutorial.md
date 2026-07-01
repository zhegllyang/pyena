# Tutorial

This tutorial walks through a complete ENA analysis on **RS.data**, a
real-world engineering-design discourse dataset bundled with rENA. It contains
3,824 utterances from 48 units across two conditions (`FirstGame`,
`SecondGame`), coded against six categories.

## Loading and preparing data

pyena expects a long-format table: one row per utterance, with columns
identifying the unit, the conversation, and one binary column per code.

```python
import pandas as pd
from pyena import ENA

df = pd.read_csv("RS_data.csv")

# Units and conversations are defined by combining condition with speaker/group
df["unit_id"] = df["Condition"] + "::" + df["UserName"]
df["conv_id"] = df["Condition"] + "::" + df["GroupName"]
```

## Fitting the model

Construct an `ENA` estimator with the code list, the unit and conversation
columns, the stanza window size, and — for means rotation — the two groups to
contrast. Calling `fit` runs the whole pipeline.

```python
ena = ENA(
    codes=[
        "Data", "Technical.Constraints", "Performance.Parameters",
        "Client.and.Consultant.Requests", "Design.Reasoning", "Collaboration",
    ],
    unit_col="unit_id",
    conversation_col="conv_id",
    window_size=4,
    rotation="means",
    mr_groups=("FirstGame", "SecondGame"),
)
ena.fit(df)
```

After fitting, every intermediate result is available as an attribute with a
trailing underscore: `ena.av_` (adjacency vectors), `ena.coords_` (unit
coordinates), `ena.nodes_` (network node positions), and `ena.rotation_matrix_`.

## Comparing groups

`compare` tests the two conditions along a chosen axis, returning Welch's t,
Mann-Whitney U, and a permutation test in one call.

```python
result = ena.compare(axis="x")
print(result["cohens_d"], result["welch_p"])
```

On RS.data the two conditions separate strongly along the rotated first axis
(Cohen's *d* ≈ 1.63).

## Plotting the network

`plot` renders the standard ENA network: code nodes at their least-squares
positions, edges weighted by the group-mean co-occurrence difference, and unit
points with centroids and 95% confidence ellipses.

```python
ena.plot()
```

## Measuring spread across repeated runs

The `reproducibility` method summarizes how tightly a set of repeated analyses
— run under nominally identical conditions — cluster in ENA space, using four
geometric indicators (centroid dispersion, mean pairwise distance, 95%
confidence-ellipse area, and convex-hull area).

```python
metrics = ena.reproducibility()
```
