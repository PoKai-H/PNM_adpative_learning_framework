"""Everything that knows karst's file formats: config keys, config parsing, output tables.

The formats are taken from karst's source (``karst/src/setup.cc``, ``analysis.cc``); the
fake karst in ``tools/karst_stub.py`` writes the same formats, so both share this module.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# time_evolution.out columns (Network::write_time_step_properties, analysis.cc)
TIME_EVOLUTION_COLUMNS = ["step", "time", "time_dt_unit", "Q", "P", "VA_tot", "VE_tot", "VX_tot", "plots",
                          "percolation", "sim_state", "porosity", "d_in_max", "d_out_max", "Va1_tot"]
# pattern_analysis.out columns (Network::write_pattern_properties, analysis.cc)
PATTERN_COLUMNS = ["step", "time", "time_dt_unit", "diss_pattern_l", "preci_pattern_l",
                   "horton_diss", "horton_preci"]


@lru_cache(maxsize=None)
def known_config_keys(setup_cc: str | Path) -> frozenset[str]:
    """All config keys karst's parser recognises (``name == "..."`` in read_setup_file)."""
    text = Path(setup_cc).read_text(errors="replace")
    keys = frozenset(re.findall(r'name\s*==\s*"([^"]+)"', text))
    if not keys:
        raise ValueError(f"found no config keys in {setup_cc}; is it karst's setup.cc?")
    return keys


def format_karst_value(key: str, value: Any) -> str:
    """Format a value the way karst's parser expects (it reads a single token after '=')."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"config value {key!r} = {value}")
        return f"{float(value):.10g}"
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"config value {key!r} = {text!r}: karst reads one token, no spaces")
    return text


def parse_karst_config(text: str, known: frozenset[str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Parse a config the way karst does: top-down, later lines win, unknown keys reported.

    Mirrors Network::read_setup_file: lines shorter than 3 chars or starting with '#' are
    skipped; a line is ``name = value`` (first three whitespace tokens).
    Returns (values, problems) with problems formatted like karst's stderr messages.
    """
    values: dict[str, str] = {}
    problems: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) < 3 or line.startswith("#"):
            continue
        tokens = line.split()
        name, eq, value = (tokens + ["", "", ""])[:3]
        if eq != "=" or not value:
            problems.append(f"Problem with parsing line nr {i}")
            continue
        if known is not None and name not in known:
            problems.append(f'Problem with parsing name "{name}" in line {i}')
            continue
        values[name] = value
    return values, problems


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= len(columns):
            try:
                rows.append([float(x) for x in parts[:len(columns)]])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=columns) if rows else None


def read_time_evolution(run_dir: Path) -> pd.DataFrame | None:
    return _read_table(Path(run_dir) / "time_evolution.out", TIME_EVOLUTION_COLUMNS)


def read_pattern_analysis(run_dir: Path) -> pd.DataFrame | None:
    return _read_table(Path(run_dir) / "pattern_analysis.out", PATTERN_COLUMNS)


def read_run_config(run_dir: Path) -> dict[str, str]:
    path = Path(run_dir) / "config.txt"
    return parse_karst_config(path.read_text())[0] if path.exists() else {}
