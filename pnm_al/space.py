"""Parameter space: dimensions, scaling, candidate grid and surrogate features."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

DerivedFeature = Callable[[pd.DataFrame], np.ndarray]   # raw parameters -> one feature column
Box = dict[str, list[float]]                              # {parameter: [low, high]}, inclusive


def parse_box(text: str) -> Box:
    """``"Pe=100:500,DaII=0.01:10"`` -> {"Pe": [100.0, 500.0], "DaII": [0.01, 10.0]} (inclusive ranges)."""
    box: Box = {}
    for part in text.split(","):
        name, _, rng = part.strip().partition("=")
        low, sep, high = rng.partition(":")
        if not name or not sep:
            raise ValueError(f"cannot parse exclude box {text!r}; expected e.g. 'Pe=100:500,DaII=0.01:10'")
        lo, hi = float(low), float(high)
        if lo > hi:
            raise ValueError(f"exclude box {text!r}: {name} low {lo} > high {hi}")
        box[name.strip()] = [lo, hi]
    return box


def in_boxes(params: pd.DataFrame, boxes: list[Box], rtol: float = 1e-9) -> np.ndarray:
    """True for rows inside any box (a box constrains only the parameters it names)."""
    mask = np.zeros(len(params), dtype=bool)
    for box in boxes:
        inside = np.ones(len(params), dtype=bool)
        for name, (lo, hi) in box.items():
            v = params[name].to_numpy(float)
            inside &= (v >= lo - rtol * abs(lo)) & (v <= hi + rtol * abs(hi))
        mask |= inside
    return mask


@dataclass(frozen=True)
class Dimension:
    """One physical input. ``log=True`` means it is modelled and gridded in log10 space."""

    name: str
    low: float
    high: float
    log: bool = True
    n_grid: int = 100

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError(f"dimension {self.name!r}: high must be > low")
        if self.log and self.low <= 0:
            raise ValueError(f"dimension {self.name!r}: a log-scaled dimension needs low > 0")
        if self.n_grid < 2:
            raise ValueError(f"dimension {self.name!r}: n_grid must be >= 2")

    def to_unit(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if self.log:
            lo, hi = math.log10(self.low), math.log10(self.high)
            return (np.log10(v) - lo) / (hi - lo)
        return (v - self.low) / (self.high - self.low)

    def grid(self) -> np.ndarray:
        if self.log:
            return np.logspace(math.log10(self.low), math.log10(self.high), self.n_grid)
        return np.linspace(self.low, self.high, self.n_grid)


class ParameterSpace:
    """The search space. ``derived`` adds surrogate features computed from raw parameters,
    e.g. ``{"log_DaI": lambda p: np.log10(p.DaII / p.Pe)}``; they are min-max scaled over the grid."""

    def __init__(self, dims: Iterable[Dimension], derived: Mapping[str, DerivedFeature] | None = None) -> None:
        self.dims = list(dims)
        if not self.dims:
            raise ValueError("the parameter space needs at least one dimension")
        names = [d.name for d in self.dims]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension names: {names}")
        self.derived = dict(derived or {})
        self._derived_range: dict[str, tuple[float, float]] = {}
        if self.derived:
            grid = self.grid()
            for name, fn in self.derived.items():
                values = np.asarray(fn(grid), dtype=float)
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
                self._derived_range[name] = (lo, hi if hi > lo else lo + 1.0)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    def with_grid(self, n_grid: int) -> "ParameterSpace":
        dims = [Dimension(d.name, d.low, d.high, d.log, n_grid) for d in self.dims]
        return ParameterSpace(dims, self.derived)

    def grid_size(self) -> int:
        return math.prod(d.n_grid for d in self.dims)

    def grid(self) -> pd.DataFrame:
        axes = [d.grid() for d in self.dims]
        return pd.DataFrame(list(itertools.product(*axes)), columns=self.names)

    def contains(self, params: pd.DataFrame, tol: float = 1e-9) -> np.ndarray:
        u = self.unit(params)
        return np.all((u >= -tol) & (u <= 1 + tol), axis=1)

    def unit(self, params: pd.DataFrame) -> np.ndarray:
        """Raw dimensions mapped to [0, 1] (log-scaled where requested); used for distances."""
        return np.column_stack([d.to_unit(params[d.name].to_numpy(float)) for d in self.dims])

    def features(self, params: pd.DataFrame) -> np.ndarray:
        columns = [self.unit(params)]
        for name, fn in self.derived.items():
            lo, hi = self._derived_range[name]
            columns.append(((np.asarray(fn(params), dtype=float) - lo) / (hi - lo))[:, None])
        return np.hstack(columns)

    def point_id(self, row: Mapping[str, float]) -> str:
        """Canonical id of a point, stable against float noise (6 significant digits)."""
        return "|".join(f"{d.name}={float(row[d.name]):.6g}" for d in self.dims)

    def point_ids(self, params: pd.DataFrame) -> list[str]:
        return [self.point_id(row) for row in params[self.names].to_dict("records")]

    @staticmethod
    def safe_name(point_id: str) -> str:
        return point_id.replace("|", "_").replace("=", "-")
