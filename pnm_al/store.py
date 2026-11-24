"""Persistent record of every simulation run, plus the loop state (all in the workdir).

``runs.csv`` holds one row per (point, seed) run and is the single source of truth, so a job
killed at any moment can be resumed by the next job.

Run status:  pending -> done -> labeled
                    \\-> failed          (non-zero exit, bad config, timeout, no label)
A ``pending`` run without a DONE marker is simply executed again on resume.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

STATUSES = ("pending", "done", "failed", "labeled")
_TEXT = ["run_id", "point_id", "status", "run_dir", "label", "label_probs", "label_source", "warnings",
         "created", "updated"]
_NUMERIC = ["round", "seed"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class ResultStore:
    def __init__(self, workdir: Path, param_names: list[str], classes: list[str]) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.param_names = list(param_names)
        self.classes = list(classes)
        self.path = self.workdir / "runs.csv"
        self.columns = ["run_id", "point_id", *_NUMERIC, *self.param_names, *_TEXT[2:], "runtime_s"]
        if self.path.exists():
            df = pd.read_csv(self.path, dtype={c: str for c in _TEXT}, keep_default_na=False)
            missing = set(self.columns) - set(df.columns)
            if missing:
                raise ValueError(f"{self.path} lacks columns {sorted(missing)} (different parameter space?)")
            self._df = df[self.columns]
        else:
            self._df = pd.DataFrame(columns=self.columns)

    @property
    def runs(self) -> pd.DataFrame:
        return self._df.copy()

    def by_status(self, *statuses: str) -> pd.DataFrame:
        return self._df[self._df["status"].isin(statuses)].copy()

    def counts_by_status(self) -> dict[str, int]:
        return {s: int((self._df["status"] == s).sum()) for s in STATUSES}

    def save(self) -> None:
        atomic_write_text(self.path, self._df.to_csv(index=False))

    def add_runs(self, records: Iterable[Mapping[str, Any]]) -> None:
        rows = []
        for rec in records:
            for c in ["run_id", "point_id", "round", "seed", "status", *self.param_names]:
                if c not in rec:
                    raise ValueError(f"run record is missing {c!r}: {rec}")
            if rec["status"] not in STATUSES:
                raise ValueError(f"unknown status {rec['status']!r}")
            row = {c: "" for c in _TEXT} | {"runtime_s": ""}
            row.update({c: rec[c] for c in self.columns if c in rec})
            row["created"] = row["updated"] = _now()
            rows.append(row)
        if not rows:
            return
        new = pd.DataFrame(rows, columns=self.columns)
        clash = set(new["run_id"]) & set(self._df["run_id"])
        if clash or new["run_id"].duplicated().any():
            raise ValueError(f"duplicate run ids: {sorted(clash)[:5]}")
        self._df = new if self._df.empty else pd.concat([self._df, new], ignore_index=True)
        self.save()

    def update_many(self, updates: Mapping[str, Mapping[str, Any]]) -> None:
        if not updates:
            return
        self._df = self._df.astype({"runtime_s": object})
        index = {rid: i for i, rid in zip(self._df.index, self._df["run_id"])}
        for run_id, fields in updates.items():
            for key, value in fields.items():
                if key not in self.columns:
                    raise KeyError(f"unknown column {key!r}")
                if key == "status" and value not in STATUSES:
                    raise ValueError(f"unknown status {value!r}")
                self._df.at[index[run_id], key] = value
            self._df.at[index[run_id], "updated"] = _now()
        self.save()

    def label_vector(self, label_probs: str) -> np.ndarray:
        probs = json.loads(label_probs)
        unknown = set(probs) - set(self.classes)
        if unknown:
            raise ValueError(f"labels {sorted(unknown)} are not among the classes {self.classes}")
        return np.array([float(probs.get(c, 0.0)) for c in self.classes])

    def labeled_points(self, max_round: int | None = None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Per point: parameters, (n_points, n_classes) label counts summed over seeds, point ids.

        ``max_round`` keeps only runs from rounds <= max_round (the data available back then).
        Rows are ordered by run_id, so the result does not depend on the order runs finished in.
        """
        df = self.by_status("labeled")
        if max_round is not None:
            df = df[df["round"].astype(int) <= max_round]
        df = df.sort_values("run_id", kind="stable")
        if df.empty:
            return pd.DataFrame(columns=self.param_names), np.zeros((0, len(self.classes))), []
        vectors = np.vstack([self.label_vector(s) for s in df["label_probs"]])
        point_ids = list(dict.fromkeys(df["point_id"]))
        position = {pid: i for i, pid in enumerate(point_ids)}
        counts = np.zeros((len(point_ids), len(self.classes)))
        np.add.at(counts, [position[p] for p in df["point_id"]], vectors)
        params = (df.drop_duplicates("point_id").set_index("point_id")
                  .loc[point_ids, self.param_names].astype(float).reset_index(drop=True))
        return params, counts, point_ids

    def active_point_ids(self) -> set[str]:
        """Points with a run that is not failed (not proposed again)."""
        return set(self._df.loc[self._df["status"] != "failed", "point_id"])


class JsonFile:
    """A small JSON document in the workdir (loop state, experiment settings)."""

    def __init__(self, path: Path, defaults: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = dict(defaults or {})
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text()))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.data, indent=2))
