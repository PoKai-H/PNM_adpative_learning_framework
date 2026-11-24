"""Surrogate interface.

Every surrogate learns x -> regime from per-point label counts and exposes *posterior samples
of the class probabilities*. That is the single currency the acquisition functions consume,
so GP, TabICL or any other model are interchangeable.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Surrogate(Protocol):
    n_classes: int

    def fit(self, X: np.ndarray, counts: np.ndarray) -> None:
        """X: (n_points, n_features) in [0, 1]; counts: (n_points, n_classes) label counts over seeds."""

    def sample_proba(self, X: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Posterior samples of p(class | x), shape (n_samples, n_points, n_classes).

        Ensembles may return their own number of members instead of ``n_samples``.
        """


def softmax(f: np.ndarray, axis: int = -1) -> np.ndarray:
    z = f - f.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def expand_counts(X: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-point counts into one row per (point, label) for hard-label classifiers.

    Counts are rounded; a point whose counts all round to zero keeps one row for its argmax.
    """
    rows, labels = [], []
    for x, c in zip(X, counts):
        reps = np.rint(c).astype(int)
        if reps.sum() == 0:
            reps[int(np.argmax(c))] = 1
        for k, r in enumerate(reps):
            rows.extend([x] * int(r))
            labels.extend([k] * int(r))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)
