"""Bootstrap ensembles of hard-label classifiers (TabICL, random forest, any sklearn model).

A single classifier only gives p(class | x) - no epistemic uncertainty, so BALD would be
meaningless. Bootstrapping over *points* (all seeds of a point stay together) and refitting
gives ``n_members`` plausible models; their predictions act as posterior samples.
For TabICL "fitting" just stores the context, so each member is cheap to build; prediction
cost scales with members x candidates, so keep ``n_members`` and the candidate pool modest.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

import numpy as np

from .base import expand_counts

ClassifierFactory = Callable[[int], Any]   # seed -> unfitted classifier with fit / predict_proba / classes_


class BootstrapEnsemble:
    def __init__(self, n_classes: int, factory: ClassifierFactory, n_members: int = 8,
                 bootstrap: bool = True, predict_chunk: int = 4096, random_state: int = 0) -> None:
        if n_members < 1:
            raise ValueError("n_members must be >= 1")
        self.n_classes = n_classes
        self.factory = factory
        self.n_members = n_members
        self.bootstrap = bootstrap and n_members > 1
        self.predict_chunk = predict_chunk
        self.random_state = random_state
        self.members: list[tuple[str, Any]] = []

    def fit(self, X: np.ndarray, counts: np.ndarray) -> None:
        X, counts = np.asarray(X, float), np.asarray(counts, float)
        rng = np.random.default_rng(self.random_state)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(X), len(X)) if self.bootstrap else np.arange(len(X))
            Xr, yr = expand_counts(X[idx], counts[idx])
            present = np.unique(yr)
            if len(present) == 1:                      # degenerate resample: constant prediction
                self.members.append(("constant", int(present[0])))
                continue
            model = self.factory(self.random_state + m)
            model.fit(Xr, yr)
            self.members.append(("model", model))

    def sample_proba(self, X: np.ndarray, n_samples: int | None = None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        if not self.members:
            raise RuntimeError("fit the surrogate first")
        X = np.asarray(X, float)
        out = np.zeros((len(self.members), len(X), self.n_classes))
        for b, (kind, member) in enumerate(self.members):
            if kind == "constant":
                out[b, :, member] = 1.0
                continue
            classes = np.asarray(member.classes_, dtype=int)
            for start in range(0, len(X), self.predict_chunk):
                chunk = slice(start, start + self.predict_chunk)
                out[b, chunk][:, classes] = member.predict_proba(X[chunk])
        return out


def _accepts(cls: type, name: str) -> bool:
    try:
        return name in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False


def tabicl_factory(params: Mapping[str, Any]) -> ClassifierFactory:
    try:
        from tabicl import TabICLClassifier
    except ImportError as err:
        raise ImportError("surrogate 'tabicl' needs the tabicl package: pip install tabicl") from err
    params = dict(params)
    seeded = _accepts(TabICLClassifier, "random_state") and "random_state" not in params

    def make(seed: int) -> Any:
        return TabICLClassifier(**params, **({"random_state": seed} if seeded else {}))

    return make


def random_forest_factory(params: Mapping[str, Any]) -> ClassifierFactory:
    from sklearn.ensemble import RandomForestClassifier

    params = {"n_estimators": 200, "min_samples_leaf": 1, **params}

    def make(seed: int) -> Any:
        return RandomForestClassifier(random_state=seed, **params)

    return make
