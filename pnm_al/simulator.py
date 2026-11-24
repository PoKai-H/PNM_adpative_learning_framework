"""Simulator adapters: turn one run (point + seed) into input files and a shell command.

* ``prepare(run, run_dir)`` – write the inputs into ``run_dir``,
* ``command()``             – shell command executed *inside* ``run_dir`` (None: nothing to run),
* ``check(run_dir)``        – problems found after the command exited 0 (empty list = ok).
"""
from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .karst_io import format_karst_value, parse_karst_config

log = logging.getLogger(__name__)
Overrides = Callable[[dict[str, float], int], Mapping[str, Any]]   # (params, seed) -> karst config values


class Simulator(Protocol):
    name: str

    def prepare(self, run: Mapping[str, Any], run_dir: Path) -> None: ...

    def command(self) -> str | None: ...

    def check(self, run_dir: Path) -> list[str]: ...


def _absolute_command(command: str) -> str:
    """Make relative paths in the command absolute (runs execute inside their run dir)."""
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("empty karst command")
    fixed = [str(Path(t).resolve()) if not Path(t).is_absolute() and Path(t).exists() else t for t in tokens]
    return shlex.join(fixed)


class KarstSimulator:
    """Run karst with ``base_config`` plus an override block appended at the end.

    karst parses top-down (later lines win) and silently ignores unknown keys, so both the
    base config and every override are checked against ``known_keys`` *before* anything runs,
    and karst's own "Problem with parsing" messages fail the run afterwards.
    """

    name = "karst"

    def __init__(self, karst_cmd: str, base_config: Path, overrides: Overrides, param_names: list[str],
                 known_keys: frozenset[str] | None = None, mode: str = "single_run") -> None:
        self.karst_cmd = _absolute_command(karst_cmd)
        self.base_config = Path(base_config)
        if not self.base_config.is_file():
            raise FileNotFoundError(f"karst base config not found: {self.base_config}")
        self.overrides = overrides
        self.param_names = list(param_names)
        self.known_keys = known_keys
        self.mode = mode
        base_text = self.base_config.read_text()
        self.base_lines = len(base_text.rstrip("\n").splitlines())   # prepare() appends after these
        self.base_unknown: set[str] = set()
        if known_keys is not None:
            _, problems = parse_karst_config(base_text, known_keys)
            self.base_unknown = {m.group(1) for p in problems if (m := re.search(r'name "([^"]+)"', p))}
            if problems:
                # the original runs used the same base config, so karst ignored these lines there too
                log.warning("base config %s has lines karst ignores (left as they are): %s",
                            self.base_config.name, problems)

    def config_lines(self, run: Mapping[str, Any]) -> list[str]:
        params = {p: float(run[p]) for p in self.param_names}
        values = dict(self.overrides(params, int(run["seed"])))
        if self.known_keys is not None:
            unknown = sorted(set(values) - self.known_keys)
            if unknown:
                raise ValueError(f"karst would silently ignore the override keys {unknown}")
        return [f"{key} = {format_karst_value(key, value)}" for key, value in values.items()]

    def prepare(self, run: Mapping[str, Any], run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        header = f"# ---- pnm_al overrides: run {run['run_id']} ({run['point_id']}, seed {run['seed']}) ----"
        text = self.base_config.read_text().rstrip("\n") + "\n\n" + header + "\n"
        (run_dir / "config.txt").write_text(text + "\n".join(self.config_lines(run)) + "\n")

    def command(self) -> str:
        # -P 0: no ps2pdf / gzip at the end (ghostscript is usually missing on compute nodes)
        return f"{self.karst_cmd} -C config.txt -V {shlex.quote(self.mode)} -P 0 > run_output.txt 2> run_errors.txt"

    def check(self, run_dir: Path) -> list[str]:
        """karst's parse complaints about *our* lines (base-config lines were accepted up front)."""
        errors = run_dir / "run_errors.txt"
        if not errors.exists():
            return ["run_errors.txt is missing"]
        problems = []
        for line in errors.read_text(errors="replace").splitlines():
            if not line.startswith("Problem with parsing"):
                continue
            name = re.search(r'name "([^"]+)"', line)
            number = re.search(r"line(?: nr)? (\d+)", line)
            from_base = (name and name.group(1) in self.base_unknown) or \
                        (name is None and number and int(number.group(1)) <= self.base_lines)
            if not from_base:
                problems.append(line.strip())
        return problems


class LookupSimulator:
    """Runs nothing. Used for the 'CSV only' side of the replay sanity check."""

    name = "lookup"

    def prepare(self, run: Mapping[str, Any], run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def command(self) -> None:
        return None

    def check(self, run_dir: Path) -> list[str]:
        return []
