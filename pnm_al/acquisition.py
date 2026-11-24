"""Acquisition functions for finding regime boundaries, and batch selection.

All scores take posterior samples of class probabilities ``P`` with shape
(n_samples, n_points, n_classes) - the common output of every surrogate.

* ``bald``    – mutual information between the label and the model (epistemic only):
                H[E_s p_s] - E_s H[p_s]. Ignores places where seeds genuinely disagree,
                so the budget is not wasted re-sampling irreducible noise. Default.
* ``entropy`` – predictive entropy H[E_s p_s] (epistemic + aleatoric).
* ``margin``  – 1 - (p_top1 - p_top2) of the mean prediction; large right on a boundary.
* ``random``  – uniform baseline (for comparisons).

Batch strategies (``[acquisition].batch_strategy``):

* ``diverse`` – greedy picks with a distance penalty so a batch spreads along the boundary (default).
* ``top_k``   – pure-uncertainty baseline: the q highest scores, no diversity.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_EPS = 1e-12


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def entropy(P: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    return _entropy(P.mean(axis=0))


def bald(P: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    return np.maximum(_entropy(P.mean(axis=0)) - _entropy(P).mean(axis=0), 0.0)


def margin(P: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    m = np.sort(P.mean(axis=0), axis=-1)
    return 1.0 - (m[:, -1] - m[:, -2]) if m.shape[-1] > 1 else np.zeros(m.shape[0])


def random_scores(P: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    return rng.random(P.shape[1])


ACQUISITIONS: dict[str, Callable[..., np.ndarray]] = {
    "bald": bald,
    "entropy": entropy,
    "margin": margin,
    "random": random_scores,
}


def get_acquisition(name: str) -> Callable[..., np.ndarray]:
    if name not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {name!r}; choose one of {sorted(ACQUISITIONS)}")
    return ACQUISITIONS[name]


BATCH_STRATEGIES = ("diverse", "top_k")


def select_top_k(scores: np.ndarray, q: int, available: np.ndarray | None = None) -> list[int]:
    """Pure-uncertainty baseline: the q highest scores, no diversity at all."""
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores) if available is None else (np.asarray(available, bool) & np.isfinite(scores))
    if not valid.any() or q <= 0:
        return []
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(-scores[idx], kind="stable")]
    return [int(i) for i in order[:q]]


def choose_batch(strategy: str, scores: np.ndarray, X_unit: np.ndarray, q: int, radius: float,
                 available: np.ndarray | None = None) -> list[int]:
    """``"diverse"``: greedy with distance penalty (default). ``"top_k"``: plain top-q baseline."""
    if strategy == "top_k":
        return select_top_k(scores, q, available)
    if strategy == "diverse":
        return select_batch(scores, X_unit, q, radius, available)
    raise ValueError(f"unknown batch_strategy {strategy!r}; choose one of {BATCH_STRATEGIES}")


def select_batch(scores: np.ndarray, X_unit: np.ndarray, q: int, radius: float,
                 available: np.ndarray | None = None) -> list[int]:
    """Greedy batch selection with a soft distance penalty.

    After picking the best candidate, every score is multiplied by
    ``1 - exp(-(d / radius)^2)`` (d = distance in unit space), so the next picks spread out
    along the boundary instead of clustering at its single most uncertain spot.
    ``radius = 0`` gives plain top-q.
    """
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores) if available is None else (np.asarray(available, bool) & np.isfinite(scores))
    if not valid.any() or q <= 0:
        return []
    # multiplicative penalties need positive scores; only shift when some are negative, so the
    # lowest-scoring candidate keeps a real (non-zero) score
    s = scores - min(float(scores[valid].min()), 0.0) + _EPS
    chosen: list[int] = []
    for _ in range(min(q, int(valid.sum()))):
        i = int(np.argmax(np.where(valid, s, -np.inf)))
        chosen.append(i)
        valid[i] = False
        if radius > 0:
            d2 = ((X_unit - X_unit[i]) ** 2).sum(axis=1)
            s = s * (1.0 - np.exp(-d2 / radius**2))
    return chosen
