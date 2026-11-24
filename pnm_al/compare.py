"""Compare two experiment workdirs round by round (replay sanity check).

Two runs agree when, in every round, they simulated the same (point, seed) pairs, got the
same labels, and produced the same regime map. The first round where they differ is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Comparison:
    identical: bool
    rounds: int
    first_divergence: int | None
    details: list[str] = field(default_factory=list)

    def report(self) -> str:
        head = ("IDENTICAL" if self.identical else
                f"DIFFERENT (first divergence in round {self.first_divergence})")
        return "\n".join([f"{head}; rounds compared: {self.rounds}", *self.details])


def _rounds(workdir: Path) -> dict[int, pd.DataFrame]:
    runs = pd.read_csv(Path(workdir) / "runs.csv", dtype=str, keep_default_na=False)
    runs["round"] = runs["round"].astype(int)
    return {r: g.sort_values(["point_id", "seed"])[["point_id", "seed", "status", "label"]].reset_index(drop=True)
            for r, g in runs.groupby("round")}


def compare_workdirs(a: Path, b: Path) -> Comparison:
    ra, rb = _rounds(a), _rounds(b)
    details: list[str] = []
    first: int | None = None
    all_rounds = sorted(set(ra) | set(rb))
    for r in all_rounds:
        problems = []
        if r not in ra or r not in rb:
            problems.append(f"only in {'A' if r in ra else 'B'}")
        else:
            ka = set(map(tuple, ra[r][["point_id", "seed"]].values))
            kb = set(map(tuple, rb[r][["point_id", "seed"]].values))
            if ka != kb:
                problems.append(f"selected runs differ: {len(ka - kb)} only in A, {len(kb - ka)} only in B")
            else:
                diff = (ra[r][["status", "label"]] != rb[r][["status", "label"]]).any(axis=1)
                if diff.any():
                    problems.append(f"{int(diff.sum())} runs have a different status/label, e.g. "
                                    f"{ra[r].loc[diff].iloc[0].to_dict()} vs {rb[r].loc[diff].iloc[0].to_dict()}")
        map_a, map_b = Path(a) / "rounds" / f"round_{r:03d}" / "map.npy", Path(b) / "rounds" / f"round_{r:03d}" / "map.npy"
        if map_a.exists() and map_b.exists():
            ma, mb = np.load(map_a), np.load(map_b)
            if ma.shape != mb.shape or np.any(ma != mb):
                frac = 1.0 if ma.shape != mb.shape else float(np.mean(ma != mb))
                problems.append(f"regime map differs on {frac:.2%} of the pool")
        elif map_a.exists() != map_b.exists():
            problems.append("regime map exists in only one workdir")
        if problems:
            first = r if first is None else first
            details.append(f"round {r}: " + "; ".join(problems))
    return Comparison(first is None, len(all_rounds), first, details)
