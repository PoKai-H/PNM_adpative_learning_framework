"""A task = one simulation problem: its parameter space, data and karst setup.

Tasks live in ``tasks/<name>.py`` and define ``TASK = Task(...)``. Everything about the AL
algorithm (surrogate, acquisition, batch size, ...) is chosen on the command line instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .simulator import Overrides
from .space import ParameterSpace


@dataclass
class Task:
    name: str
    classes: list[str]
    space: ParameterSpace
    data_csv: Path                          # already-labeled runs (initial data, replay pool, oracle)
    param_columns: dict[str, str]           # space dimension -> CSV column
    seed_column: str
    label_column: str
    karst_dir: Path                         # karst source tree (binary, setup.cc for the key check)
    base_config: Path
    overrides: Overrides                    # (params, seed) -> karst config values
    description: str = ""
    labeler_options: dict = field(default_factory=dict)   # e.g. placeholder thresholds

    @property
    def karst_binary(self) -> Path:
        return self.karst_dir / "bin" / "karst"

    @property
    def setup_cc(self) -> Path:
        return self.karst_dir / "src" / "setup.cc"

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Returns (raw table, parameters in space names, point ids) of ``data_csv``."""
        table = pd.read_csv(self.data_csv)
        missing = [c for c in [*self.param_columns.values(), self.seed_column, self.label_column]
                   if c not in table.columns]
        if missing:
            raise ValueError(f"{self.data_csv} has no columns {missing}")
        params = pd.DataFrame({p: table[c].astype(float) for p, c in self.param_columns.items()})
        return table, params, self.space.point_ids(params)
