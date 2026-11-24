"""Labelers: finished simulation -> regime probabilities (option A: label, then learn x -> label).

Interface (the future image/field classifier implements exactly this):

    labeler.name                         # recorded as label_source for every label
    labeler.label(run_dir, run) -> {class: probability} | None     (None = cannot label)

* ``CSVLabeler``         – looks the label up in the task's CSV by (point, seed). Oracle for
                           the replay sanity check (L1).
* ``PlaceholderLabeler`` – UNCALIBRATED rules on karst outputs, until the classifier exists.
* ``CallableLabeler``    – ``"path/to/file.py:function"`` or ``"module:function"``.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from .karst_io import read_pattern_analysis, read_run_config, read_time_evolution


class Labeler(Protocol):
    name: str

    def label(self, run_dir: Path, run: Mapping[str, Any]) -> dict[str, float] | None: ...


def normalize_label(value: Any, classes: list[str]) -> dict[str, float]:
    """Accept a class name or a {class: weight} mapping; return normalised probabilities."""
    if isinstance(value, str):
        value = value.strip()
        if value not in classes:
            raise ValueError(f"label {value!r} is not one of {classes}")
        return {value: 1.0}
    if isinstance(value, Mapping):
        unknown = set(value) - set(classes)
        if unknown:
            raise ValueError(f"labels {sorted(unknown)} are not among {classes}")
        total = float(sum(value.values()))
        if total <= 0 or any(v < 0 for v in value.values()):
            raise ValueError(f"label probabilities must be non-negative and sum > 0: {value}")
        return {k: float(v) / total for k, v in value.items() if v > 0}
    raise TypeError(f"cannot interpret label {value!r}")


class CSVLabeler:
    name = "csv"

    def __init__(self, table: pd.DataFrame, point_ids: list[str], seed_column: str, label_column: str,
                 classes: list[str]) -> None:
        self.classes = classes
        self.lookup: dict[tuple[str, int], str] = {}
        for pid, seed, label in zip(point_ids, table[seed_column], table[label_column]):
            self.lookup[(pid, int(seed))] = str(label)

    def label(self, run_dir: Path, run: Mapping[str, Any]) -> dict[str, float] | None:
        value = self.lookup.get((run["point_id"], int(run["seed"])))
        return None if value is None else normalize_label(value, self.classes)


class PlaceholderLabeler:
    """Stand-in until the classifier exists. Thresholds are NOT calibrated: use the resulting
    maps to test the pipeline only, and calibrate on re-simulated labeled points (check L2).

    Features from karst outputs:
      pattern_frac = final dissolution-pattern length / N_y      (pattern_analysis.out)
      dissolved    = 1 - VA_tot(end) / VA_tot(0)                 (time_evolution.out)
    Rules: short pattern -> compact; large dissolved fraction -> uniform; else wormholing.
    """

    name = "placeholder_rule_v0"

    def __init__(self, classes: list[str], compact_max_pattern: float = 0.25,
                 uniform_min_dissolved: float = 0.5) -> None:
        missing = {"uniform", "wormholing", "compact"} - set(classes)
        if missing:
            raise ValueError(f"the placeholder rules need the classes {sorted(missing)}")
        self.classes = classes
        self.compact_max_pattern = compact_max_pattern
        self.uniform_min_dissolved = uniform_min_dissolved

    @staticmethod
    def features(run_dir: Path) -> dict[str, float] | None:
        pattern, evolution = read_pattern_analysis(run_dir), read_time_evolution(run_dir)
        if pattern is None or evolution is None:
            return None
        n_y = float(read_run_config(run_dir).get("N_y", "nan").rstrip(";"))
        va0 = float(evolution["VA_tot"].iloc[0])
        return {"pattern_frac": float(pattern["diss_pattern_l"].iloc[-1]) / n_y,
                "dissolved": 1.0 - float(evolution["VA_tot"].iloc[-1]) / va0 if va0 > 0 else float("nan")}

    def label(self, run_dir: Path, run: Mapping[str, Any]) -> dict[str, float] | None:
        f = self.features(run_dir)
        if f is None or not all(np.isfinite(list(f.values()))):
            return None
        if f["pattern_frac"] < self.compact_max_pattern:
            return {"compact": 1.0}
        if f["dissolved"] > self.uniform_min_dissolved:
            return {"uniform": 1.0}
        return {"wormholing": 1.0}


def _load_callable(target: str) -> Callable[..., Any]:
    module_name, _, func_name = target.rpartition(":")
    if not module_name or not func_name:
        raise ValueError(f"a callable labeler looks like 'file.py:function' or 'module:function', got {target!r}")
    if module_name.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(module_name).stem, Path(module_name).resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, func_name)


class CallableLabeler:
    """``fn(run_dir: Path, params: dict) -> class name | {class: prob} | None``."""

    def __init__(self, target: str, classes: list[str], param_names: list[str]) -> None:
        self.fn = _load_callable(target)
        self.classes = classes
        self.param_names = param_names
        self.name = f"callable:{target}"

    def label(self, run_dir: Path, run: Mapping[str, Any]) -> dict[str, float] | None:
        params = {p: float(run[p]) for p in self.param_names} | {"seed": int(run["seed"])}
        value = self.fn(Path(run_dir), params)
        return None if value is None else normalize_label(value, self.classes)
