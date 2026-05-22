"""Sphere normalization of adjacency vectors.

Each adjacency vector is divided by its L2 norm, projecting it onto the unit
hypersphere. Zero-norm vectors are returned unchanged, following rENA's
convention.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sphere_normalize"]


def sphere_normalize(av):
    """L2-normalize each vector to unit length.

    Parameters
    ----------
    av : np.ndarray
        Either a single adjacency vector of shape ``(n_pairs,)`` or a
        stacked matrix of shape ``(n_units, n_pairs)``.

    Returns
    -------
    np.ndarray
        Same shape as the input. Each row (or the single vector) is rescaled
        to unit L2 norm. Rows whose L2 norm is exactly zero are returned
        unchanged, matching rENA's behaviour.
    """
    av = np.asarray(av, dtype=float)
    if av.ndim == 1:
        norm = np.linalg.norm(av)
        return av if norm == 0 else av / norm
    norms = np.linalg.norm(av, axis=1, keepdims=True)
    norms_safe = np.where(norms == 0, 1.0, norms)
    return av / norms_safe
