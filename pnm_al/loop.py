"""The active-learning loop. Runs whole rounds inside one Slurm job and checkpoints to disk.

    run():  initialise (import initial data) -> repeat:
                finish pending runs (resume) -> label
                enough walltime for another round?  no -> return "checkpoint"
                fit surrogate -> score pool -> stop? -> select batch -> prepare runs (pending)
                execute in parallel -> label

Every state change is written to the workdir before the next action, so a job killed at
any point resumes in the next job exactly where it stopped (unfinished runs execute again).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import runner
from .acquisition import BATCH_STRATEGIES, choose_batch, get_acquisition
from .evaluation import Truth, classification_metrics, load_truth
from .karst_io import known_config_keys
from .labelers import CallableLabeler, CSVLabeler, Labeler, PlaceholderLabeler, normalize_label
from .simulator import KarstSimulator, LookupSimulator, Simulator
from .space import in_boxes
from .store import JsonFile, ResultStore
from .surrogates import SURROGATES, Surrogate, build_surrogate
from .task import Task

log = logging.getLogger(__name__)
MAX_GRID = 2_000_000


@dataclass
class Settings:
    """Everything that defines an experiment. Saved to settings.json and must not change on resume."""

    task: str
    surrogate: str = "dirichlet_gp"
    surrogate_params: dict = field(default_factory=dict)
    acquisition: str = "bald"
    batch_strategy: str = "diverse"
    batch_size: int = 20
    seeds: list = field(default_factory=lambda: [41, 42, 43])
    penalty_radius: float = 0.05
    n_samples: int = 64
    max_rounds: int = 15
    max_points: int = 10**9
    map_change_tol: float = 0.005
    patience: int = 2
    pool: str = "grid"                  # "grid": the task's grid; "csv": the points of the task CSV
    n_grid: int | None = None           # grid points per dimension (None: the task's default)
    max_candidates: int = 20000
    init_points: int | None = None      # None: import every CSV point; N: a random split of N points
    init_seeds: list | None = None      # seeds imported for the initial points (None: all)
    split_seed: int = 0
    random_state: int = 0
    labeler: str = "placeholder"        # "placeholder" | "csv" | "file.py:function" | "module:function"
    simulator: str = "karst"            # "karst" | "lookup" (no simulation, replay check only)
    chunk_size: int = 4096
    evaluate: bool = True               # write curve.csv: metrics vs the CSV ground truth after every round
    exclude: list = field(default_factory=list)   # boxes {param: [low, high]} never sampled (extrapolation test)

    def __post_init__(self) -> None:
        self.seeds = [int(s) for s in self.seeds]
        self.init_seeds = None if self.init_seeds is None else [int(s) for s in self.init_seeds]
        checks = {"surrogate": (self.surrogate, SURROGATES), "batch_strategy": (self.batch_strategy, BATCH_STRATEGIES),
                  "pool": (self.pool, ("grid", "csv")), "simulator": (self.simulator, ("karst", "lookup"))}
        for name, (value, allowed) in checks.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        get_acquisition(self.acquisition)
        if not self.seeds:
            raise ValueError("seeds must list at least one seed")
        if self.simulator == "lookup" and self.labeler != "csv":
            raise ValueError("simulator 'lookup' produces no outputs, so it needs labeler 'csv'")


@dataclass
class StepResult:
    status: str        # "finished" | "checkpoint" | "started"
    message: str


class ActiveLearner:
    def __init__(self, task: Task, settings: Settings, workdir: Path, karst_cmd: str | None = None,
                 workers: int | None = None, run_timeout: float | None = None) -> None:
        self.task, self.s = task, settings
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._check_settings()
        self.space = task.space if settings.n_grid is None else task.space.with_grid(settings.n_grid)
        unknown = {name for box in settings.exclude for name in box} - set(self.space.names)
        if unknown:
            raise ValueError(f"exclude boxes use unknown parameters {sorted(unknown)}; known: {self.space.names}")
        self.store = ResultStore(self.workdir, self.space.names, task.classes)
        self.state = JsonFile(self.workdir / "state.json", {"initialized": False, "round": 0, "map_changes": [],
                                                           "round_seconds": [], "finished": False, "finish_reason": ""})
        self.workers = workers or runner.default_workers()
        self.run_timeout = run_timeout
        self.simulator = self._build_simulator(karst_cmd)
        self.labeler = self.build_labeler(settings.labeler)
        self.acquisition = get_acquisition(settings.acquisition)
        self._candidates: pd.DataFrame | None = None

    # ------------------------------------------------------------- building
    def _check_settings(self) -> None:
        path = self.workdir / "settings.json"
        current = asdict(self.s)
        if path.exists():
            saved = json.loads(path.read_text())
            # keys added in newer versions are missing from old settings.json: treat as their default
            defaults = {f.name: (f.default_factory() if f.default_factory is not MISSING else f.default)
                        for f in fields(Settings)}
            diff = {k: (saved.get(k, defaults[k]), v) for k, v in current.items() if saved.get(k, defaults[k]) != v}
            if diff:
                raise ValueError(f"{self.workdir} was started with other settings {diff} (saved, now); "
                                 "use a new --workdir, or `relabel` to change the labeler")
        else:
            path.write_text(json.dumps(current, indent=2))

    def _build_simulator(self, karst_cmd: str | None) -> Simulator:
        if self.s.simulator == "lookup":
            return LookupSimulator()
        cmd = karst_cmd or str(self.task.karst_binary)
        keys = known_config_keys(self.task.setup_cc) if self.task.setup_cc.exists() else None
        return KarstSimulator(cmd, self.task.base_config, self.task.overrides, self.space.names, keys)

    def build_labeler(self, spec: str) -> Labeler:
        if spec == "csv":
            table, _, pids = self.task.load_data()
            return CSVLabeler(table, pids, self.task.seed_column, self.task.label_column, self.task.classes)
        if spec == "placeholder":
            return PlaceholderLabeler(self.task.classes, **self.task.labeler_options)
        return CallableLabeler(spec, self.task.classes, self.space.names)

    @property
    def candidates(self) -> pd.DataFrame:
        """Fixed candidate pool for the whole experiment (also the map used for stopping)."""
        if self._candidates is None:
            if self.s.pool == "csv":
                _, params, pids = self.task.load_data()
                pool = params.assign(_pid=pids).drop_duplicates("_pid").drop(columns="_pid")
            else:
                if self.space.grid_size() > MAX_GRID:
                    raise ValueError(f"grid has {self.space.grid_size():,} points; lower n_grid")
                pool = self.space.grid()
                if len(pool) > self.s.max_candidates:
                    keep = np.sort(np.random.default_rng(self.s.random_state)
                                   .choice(len(pool), self.s.max_candidates, replace=False))
                    pool = pool.iloc[keep]
            pool = pool[~in_boxes(pool, self.s.exclude)]          # excluded region is never sampled
            self._candidates = pool.reset_index(drop=True)
        return self._candidates

    # ------------------------------------------------------- initialisation
    def initialize(self) -> int:
        """Import the initial labeled data (all CSV points, or a split of ``init_points``)."""
        if self.state["initialized"]:
            return 0
        table, params, pids = self.task.load_data()
        excluded = set(np.asarray(pids)[in_boxes(params, self.s.exclude)])
        unique = [pid for pid in dict.fromkeys(pids) if pid not in excluded]   # never import excluded points
        if self.s.init_points is not None:
            if not 0 < self.s.init_points <= len(unique):
                raise ValueError(f"init_points must be in 1..{len(unique)}")
            chosen = set(np.random.default_rng(self.s.split_seed).choice(unique, self.s.init_points, replace=False))
        else:
            chosen = set(unique)
        records = []
        for i, (prow, pid, seed, label) in enumerate(zip(params.to_dict("records"), pids, table[self.task.seed_column],
                                                         table[self.task.label_column])):
            if pid not in chosen or (self.s.init_seeds is not None and int(seed) not in self.s.init_seeds):
                continue
            probs = normalize_label(str(label), self.task.classes)
            records.append({"run_id": f"import_{i:05d}", "point_id": pid, "round": 0, "seed": int(seed),
                            "status": "labeled", "label": max(probs, key=probs.get),
                            "label_probs": json.dumps(probs), "label_source": "import", **prow})
        self.store.add_runs(records)
        self.state["initialized"] = True
        self.state.save()
        log.info("imported %d labeled runs at %d points from %s", len(records), len(chosen), self.task.data_csv)
        return len(records)

    # ------------------------------------------------------------------ run
    def run(self, time_left: Callable[[], float | None] | None = None,
            round_time_guess: float = 3600.0) -> StepResult:
        """Do rounds until finished, or until the remaining walltime is too short for one more."""
        self.initialize()
        while True:
            if self.state["finished"]:
                self.update_curve()
                return StepResult("finished", self.state["finish_reason"])
            self.execute_pending()
            self.label_done()
            self.update_curve()
            left = time_left() if time_left is not None else None
            if left is not None:
                need = 1.25 * max(self.state["round_seconds"] or [round_time_guess]) + 60
                if left < need:
                    return StepResult("checkpoint", f"{left:.0f} s left, a round needs ~{need:.0f} s")
            start = time.monotonic()
            result = self.start_round()
            log.info("[%s] %s", result.status, result.message)
            if result.status == "finished":
                self.update_curve()
                return result
            self.execute_pending()
            self.label_done()
            self.state["round_seconds"] = self.state["round_seconds"] + [time.monotonic() - start]
            self.state.save()

    def execute_pending(self) -> None:
        pending = self.store.by_status("pending")
        if pending.empty:
            return
        dirs = [Path(d) for d in pending["run_dir"]]
        results = runner.execute(dirs, self.simulator.command(), self.workers, self.run_timeout)
        updates = {}
        for run_id, run_dir in zip(pending["run_id"], dirs):
            res = results[run_dir]
            fields = {"runtime_s": "" if res.runtime_s is None else round(res.runtime_s, 2)}
            problems = self.simulator.check(run_dir) if res.status == "done" else []
            if res.status == "failed" or problems:
                fields |= {"status": "failed", "warnings": res.message or "; ".join(problems)}
            else:
                fields["status"] = "done"
            updates[run_id] = fields
        self.store.update_many(updates)

    def label_done(self) -> None:
        done = self.store.by_status("done")
        updates = {}
        for run in done.to_dict("records"):
            try:
                probs = self.labeler.label(Path(run["run_dir"]), run)
            except Exception as err:                       # a broken output must not stop the loop
                probs, error = None, f"labeler error: {err}"
            else:
                error = "labeler returned no label"
            updates[run["run_id"]] = ({"status": "failed", "warnings": error} if probs is None else
                                      {"status": "labeled", "label": max(probs, key=probs.get),
                                       "label_probs": json.dumps(probs), "label_source": self.labeler.name})
        self.store.update_many(updates)

    # ------------------------------------------------------------- surrogate
    def fit_surrogate(self, max_round: int | None = None) -> Surrogate:
        """Fit on all labels (default) or on the labels of rounds <= max_round.

        The random state depends on the round, so fit_surrogate(k) is exactly the surrogate
        that selected the points of round k + 1.
        """
        params, counts, _ = self.store.labeled_points(max_round)
        if len(params) == 0:
            raise RuntimeError("no labeled data to fit the surrogate on")
        k = self.state["round"] if max_round is None else max_round
        surrogate = build_surrogate(self.s.surrogate, len(self.task.classes), self.s.surrogate_params,
                                    random_state=self.s.random_state + k)
        surrogate.fit(self.space.features(params), counts)
        return surrogate

    # ------------------------------------------------------------ evaluation
    @property
    def truth(self) -> Truth:
        if not hasattr(self, "_truth"):
            self._truth = load_truth(self.task)
        return self._truth

    def observed_labels(self, point_ids: list[str], max_round: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What was actually observed at each point (rounds <= max_round).

        Returns the majority class index (-1 where the point was never labeled), the fraction
        of seeds agreeing with it, and the number of labeled seeds.
        """
        _, counts, pids = self.store.labeled_points(max_round)
        position = {p: i for i, p in enumerate(pids)}
        label = np.full(len(point_ids), -1, dtype=int)
        purity = np.full(len(point_ids), np.nan)
        n_seeds = np.zeros(len(point_ids))
        for j, pid in enumerate(point_ids):
            if pid in position:
                c = counts[position[pid]]
                label[j], purity[j], n_seeds[j] = int(c.argmax()), c.max() / c.sum(), c.sum()
        return label, purity, n_seeds

    @staticmethod
    def final_map(model_pred: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Regime map: the observed label where a point was simulated, the surrogate elsewhere."""
        return np.where(observed >= 0, observed, model_pred)

    def evaluate_round(self, k: int) -> dict:
        """Metrics after round k against the CSV truth.

        accuracy, ...         the regime map (observed labels at simulated points, surrogate elsewhere)
        unlabeled_accuracy, ... only the points not simulated yet (the surrogate's own skill)
        model_accuracy, ...   the pure surrogate prediction at every point (ignores observations)
        excluded_accuracy, ... only the excluded region (if --exclude): extrapolation skill
        """
        _, _, train_pids = self.store.labeled_points(k)
        surrogate = self.fit_surrogate(k)
        rng = np.random.default_rng(self.s.random_state + 104729 * (k + 1))   # separate from selection rng
        X = self.space.features(self.truth.params)
        mean = np.zeros((len(X), len(self.task.classes)))
        for start in range(0, len(X), self.s.chunk_size):
            chunk = slice(start, start + self.s.chunk_size)
            mean[chunk] = surrogate.sample_proba(X[chunk], self.s.n_samples, rng).mean(axis=0)
        model_pred = mean.argmax(axis=1)
        observed, _, _ = self.observed_labels(self.truth.point_ids, k)
        pred = self.final_map(model_pred, observed)
        unlabeled = observed < 0
        runs = self.store.runs
        rounds = runs["round"].astype(int)
        row = {"round": k, "n_points": len(train_pids),
               "n_simulations": int(((rounds >= 1) & (rounds <= k) & (runs["status"] == "labeled")).sum()),
               "n_failed": int(((rounds >= 1) & (rounds <= k) & (runs["status"] == "failed")).sum())}
        row |= classification_metrics(self.truth.labels, pred, self.task.classes)
        row |= classification_metrics(self.truth.labels[unlabeled], model_pred[unlabeled], self.task.classes,
                                      prefix="unlabeled_")
        row |= classification_metrics(self.truth.labels, model_pred, self.task.classes, prefix="model_")
        if self.s.exclude:                     # extrapolation into the region that is never sampled
            excluded = in_boxes(self.truth.params, self.s.exclude)
            row |= classification_metrics(self.truth.labels[excluded], model_pred[excluded], self.task.classes,
                                          prefix="excluded_")
        return row

    def update_curve(self, recompute: bool = False) -> pd.DataFrame | None:
        """Append metrics for every completed round missing from curve.csv (all rounds if recompute)."""
        if not self.s.evaluate:
            return None
        path = self.workdir / "curve.csv"
        curve = pd.read_csv(path) if path.exists() and not recompute else pd.DataFrame()
        if len(curve) and "model_accuracy" not in curve.columns:
            curve = pd.DataFrame()           # written by an older version (no observed override): recompute
        have = set(curve["round"].astype(int)) if len(curve) else set()
        runs = self.store.runs
        open_rounds = runs.loc[runs["status"].isin(["pending", "done"]), "round"].astype(int)
        first_open = int(open_rounds.min()) if len(open_rounds) else None
        rows = []
        for k in range(self.state["round"] + 1):
            if k in have or (first_open is not None and k >= first_open):
                continue
            if len(self.store.labeled_points(k)[2]) == 0:
                continue
            rows.append(self.evaluate_round(k))
        if rows:
            curve = pd.concat([curve, pd.DataFrame(rows)], ignore_index=True) if len(curve) else pd.DataFrame(rows)
            curve = curve.sort_values("round").reset_index(drop=True)
            tmp = path.with_name(path.name + ".tmp")
            curve.to_csv(tmp, index=False)
            tmp.replace(path)
        return curve

    def score(self, surrogate: Surrogate, params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.s.random_state + 7919 * (self.state["round"] + 1))
        X = self.space.features(params)
        mean = np.zeros((len(X), len(self.task.classes)))
        scores = np.zeros(len(X))
        for start in range(0, len(X), self.s.chunk_size):
            chunk = slice(start, start + self.s.chunk_size)
            P = surrogate.sample_proba(X[chunk], self.s.n_samples, rng)
            mean[chunk] = P.mean(axis=0)
            scores[chunk] = self.acquisition(P, rng)
        return mean, scores

    def predict_map(self, pool: pd.DataFrame | None = None) -> pd.DataFrame:
        """Columns: p_<class> (surrogate), regime_model, observed (majority of the simulated seeds,
        empty if never simulated), observed_purity, n_seeds, regime (observed if simulated, else model)."""
        pool = self.candidates if pool is None else pool
        mean, scores = self.score(self.fit_surrogate(), pool)
        observed, purity, n_seeds = self.observed_labels(self.space.point_ids(pool))
        classes = np.asarray(self.task.classes)
        out = pool.copy()
        for k, c in enumerate(self.task.classes):
            out[f"p_{c}"] = mean[:, k]
        out["regime_model"] = classes[mean.argmax(axis=1)]
        out["observed"] = np.where(observed >= 0, classes[np.maximum(observed, 0)], "")
        out["observed_purity"] = purity
        out["n_seeds"] = n_seeds.astype(int)
        out["regime"] = classes[self.final_map(mean.argmax(axis=1), observed)]
        out["score"] = scores
        return out

    # ----------------------------------------------------------------- round
    def _finish(self, reason: str) -> StepResult:
        self.state["finished"] = True
        self.state["finish_reason"] = reason
        self.state.save()
        return StepResult("finished", reason)

    def start_round(self) -> StepResult:
        """Fit, score, check stopping, select a batch and prepare its runs (status pending)."""
        r = self.state["round"]
        params, counts, point_ids = self.store.labeled_points()
        if r >= self.s.max_rounds:
            return self._finish(f"reached max_rounds = {self.s.max_rounds}")
        if len(point_ids) >= self.s.max_points:
            return self._finish(f"reached max_points = {self.s.max_points}")

        pool = self.candidates
        pool_ids = self.space.point_ids(pool)
        available = ~np.isin(pool_ids, list(self.store.active_point_ids()))
        mean, scores = self.score(self.fit_surrogate(), pool)
        observed, _, _ = self.observed_labels(pool_ids)
        regime_map = self.final_map(mean.argmax(axis=1), observed)      # observed labels stay fixed
        round_dir = self.workdir / "rounds" / f"round_{r + 1:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        np.save(round_dir / "map.npy", regime_map)
        previous = self.workdir / "rounds" / f"round_{r:03d}" / "map.npy"
        change = float(np.mean(np.load(previous) != regime_map)) if previous.exists() else None
        if change is not None:
            self.state["map_changes"] = self.state["map_changes"] + [change]

        q = min(self.s.batch_size, self.s.max_points - len(point_ids))
        chosen = choose_batch(self.s.batch_strategy, scores, self.space.unit(pool), q, self.s.penalty_radius, available)
        self._save_diagnostics(round_dir, pool, mean, scores, chosen, len(point_ids), change, observed)

        recent = self.state["map_changes"][-self.s.patience:]
        if len(recent) >= self.s.patience and all(c < self.s.map_change_tol for c in recent):
            return self._finish(f"regime map converged (change {recent} < {self.s.map_change_tol})")
        if not chosen:
            return self._finish("no candidate points left")
        self._prepare_runs(pool.iloc[chosen].reset_index(drop=True), r + 1)
        text = "n/a" if change is None else f"{change:.4f}"
        return StepResult("started", f"round {r + 1}: {len(chosen)} points x {len(self.s.seeds)} seeds "
                                     f"({self.s.acquisition}/{self.s.batch_strategy} on {len(point_ids)} points, "
                                     f"map change {text})")

    def _prepare_runs(self, points: pd.DataFrame, round_no: int) -> None:
        records = []
        for j, row in enumerate(points[self.space.names].to_dict("records")):
            pid = self.space.point_id(row)
            for seed in self.s.seeds:
                run_dir = self.workdir / "sims" / f"round_{round_no:03d}" / f"{self.space.safe_name(pid)}_seed{seed}"
                record = {"run_id": f"r{round_no:03d}_p{j:04d}_s{seed}", "point_id": pid, "round": round_no,
                          "seed": seed, "status": "pending", "run_dir": str(run_dir), **row}
                self.simulator.prepare(record, run_dir)
                records.append(record)
        self.store.add_runs(records)          # recorded before anything runs
        self.state["round"] = round_no
        self.state.save()

    def _save_diagnostics(self, round_dir: Path, pool: pd.DataFrame, mean: np.ndarray, scores: np.ndarray,
                          chosen: list[int], n_points: int, change: float | None, observed: np.ndarray) -> None:
        classes = np.asarray(self.task.classes)
        regime_map = self.final_map(mean.argmax(axis=1), observed)
        table = pool.copy()
        for k, c in enumerate(self.task.classes):
            table[f"p_{c}"] = mean[:, k]
        table["regime_model"] = classes[mean.argmax(axis=1)]
        table["observed"] = np.where(observed >= 0, classes[np.maximum(observed, 0)], "")
        table["regime"] = classes[regime_map]
        table["score"] = scores
        table["selected"] = False
        table.loc[chosen, "selected"] = True
        table.to_csv(round_dir / "candidates.csv", index=False)
        summary = {"round": int(round_dir.name.split("_")[-1]), "n_labeled_points": n_points,
                   "surrogate": self.s.surrogate, "acquisition": self.s.acquisition,
                   "batch_strategy": self.s.batch_strategy, "map_change": change, "n_selected": len(chosen),
                   "regime_fractions": {c: float(np.mean(regime_map == k)) for k, c in enumerate(self.task.classes)}}
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --------------------------------------------------------------- relabel
    def relabel(self, spec: str) -> dict[str, int]:
        """Label every simulated run again with another labeler (e.g. the classifier, later).

        Imported runs keep their labels. Runs whose simulation failed stay failed.
        """
        labeler = self.build_labeler(spec)
        runs = self.store.runs
        eligible = runs[(runs["run_dir"] != "") & (runs["status"].isin(["labeled", "failed", "done"]))]
        updates, counts = {}, {"labeled": 0, "no_label": 0, "skipped": 0}
        for run in eligible.to_dict("records"):
            run_dir = Path(run["run_dir"])
            if not (run_dir / "DONE").exists() or self.simulator.check(run_dir):
                counts["skipped"] += 1
                continue
            probs = labeler.label(run_dir, run)
            if probs is None:
                updates[run["run_id"]] = {"status": "failed", "warnings": f"{labeler.name}: no label"}
                counts["no_label"] += 1
            else:
                updates[run["run_id"]] = {"status": "labeled", "label": max(probs, key=probs.get),
                                          "label_probs": json.dumps(probs), "label_source": labeler.name,
                                          "warnings": ""}
                counts["labeled"] += 1
        self.store.update_many(updates)
        settings_path = self.workdir / "settings.json"
        saved = json.loads(settings_path.read_text())
        saved["labeler"] = spec
        settings_path.write_text(json.dumps(saved, indent=2))
        return counts

    # ---------------------------------------------------------------- status
    def summary(self) -> str:
        counts = self.store.counts_by_status()
        _, label_counts, pids = self.store.labeled_points()
        lines = [f"task: {self.task.name}   workdir: {self.workdir}",
                 f"round: {self.state['round']}   finished: {self.state['finished']} {self.state['finish_reason']}",
                 "runs: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
                 f"labeled points: {len(pids)}"]
        if len(pids):
            top = label_counts.argmax(axis=1)
            lines.append("points by majority regime: " + ", ".join(
                f"{c}={int((top == k).sum())}" for k, c in enumerate(self.task.classes)))
        if self.state["map_changes"]:
            lines.append("map change per round: " + ", ".join(f"{c:.4f}" for c in self.state["map_changes"]))
        return "\n".join(lines)


def load_settings(workdir: Path) -> Settings:
    path = Path(workdir) / "settings.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; start the experiment with `run_al.py run` first")
    return Settings(**json.loads(path.read_text()))
