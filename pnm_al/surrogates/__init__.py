"""Surrogate registry: ``build_surrogate("dirichlet_gp" | "tabicl" | "random_forest", ...)``."""
from __future__ import annotations

from typing import Any, Mapping

from .base import Surrogate, expand_counts, softmax
from .ensemble import BootstrapEnsemble, random_forest_factory, tabicl_factory
from .gp import DirichletGPClassifier

SURROGATES = ("dirichlet_gp", "tabicl", "random_forest")
_ENSEMBLE_KEYS = ("n_members", "bootstrap", "predict_chunk")


def build_surrogate(kind: str, n_classes: int, params: Mapping[str, Any] | None = None,
                    random_state: int = 0) -> Surrogate:
    params = dict(params or {})
    if kind == "dirichlet_gp":
        return DirichletGPClassifier(n_classes, random_state=random_state, **params)
    if kind in ("tabicl", "random_forest"):
        ensemble = {k: params.pop(k) for k in _ENSEMBLE_KEYS if k in params}
        model_params = params.pop("model", {}) | params
        factory = tabicl_factory(model_params) if kind == "tabicl" else random_forest_factory(model_params)
        return BootstrapEnsemble(n_classes, factory, random_state=random_state, **ensemble)
    raise ValueError(f"unknown surrogate {kind!r}; choose one of {SURROGATES}")


__all__ = ["Surrogate", "BootstrapEnsemble", "DirichletGPClassifier", "build_surrogate",
           "expand_counts", "softmax", "SURROGATES"]
