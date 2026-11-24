#!/usr/bin/env python3
"""Benchmark AL strategies on the task CSV only (no simulation) and draw learning curves.

Every (surrogate, acquisition, split seed) is an independent pool-based AL run: the pool is the
CSV's points, labels are looked up in the CSV, and after every round the surrogate is scored
against the CSV ground truth (curve.csv). Runs are aggregated over split seeds (mean +- std).

  python benchmark.py --out runs/bench --surrogates dirichlet_gp random_forest \\
                      --acquisitions bald random --split-seeds 0 1 2 3 4 5 6 7 8 9 --jobs 4

  python benchmark.py --out runs/bench --plot-only      # re-aggregate / re-plot existing runs

The 'random' acquisition always uses batch strategy top_k (uniform random picks); the others
use --batch-strategy (default top_k = pure uncertainty, no diversity). Finished runs are
skipped, so an interrupted benchmark can simply be started again with the same arguments.

Output in --out:
  <surrogate>__<acquisition>__<batch>/split_XXX/   one AL workdir per run (curve.csv inside)
  bench_curves.csv    every curve row of every run, with its configuration
  bench_summary.csv   mean / std over split seeds per configuration and round
  bench_curves.png    learning curves (one panel per metric)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pnm_al.loop import ActiveLearner, Settings  # noqa: E402
from pnm_al.space import parse_box  # noqa: E402
from tasks import TASKS, get_task  # noqa: E402

# metrics missing from a benchmark (e.g. excluded_* without --exclude) are skipped when plotting.
# Panels are laid out row by row: balanced accuracy, accuracy, macro F1; columns = all / unlabeled / excluded.
DEFAULT_METRICS = ["balanced_accuracy", "unlabeled_balanced_accuracy", "excluded_balanced_accuracy",
                   "accuracy", "unlabeled_accuracy", "excluded_accuracy", "macro_f1"]


def run_dir(out: Path, surrogate: str, acquisition: str, batch: str, split_seed: int) -> Path:
    return out / f"{surrogate}__{acquisition}__{batch}" / f"split_{split_seed:03d}"


def run_one(job: dict) -> str:
    """One AL run (executed in a worker process)."""
    logging.basicConfig(level=logging.WARNING)
    settings = Settings(**job["settings"])
    learner = ActiveLearner(get_task(settings.task), settings, Path(job["workdir"]), workers=1)
    result = learner.run()
    return f"{job['workdir']}: {result.status} ({result.message})"


def make_jobs(a: argparse.Namespace) -> list[dict]:
    task = get_task(a.task)
    seeds = a.seeds
    if seeds is None:                                   # default: the first 3 seeds present in the CSV
        table, _, _ = task.load_data()
        seeds = sorted({int(s) for s in table[task.seed_column]})[:3]
    params = json.loads(a.surrogate_params) if a.surrogate_params else {}
    jobs = []
    for surrogate in a.surrogates:
        for acquisition in a.acquisitions:
            batch = "top_k" if acquisition == "random" else a.batch_strategy
            for split_seed in a.split_seeds:
                settings = dict(task=a.task, surrogate=surrogate, surrogate_params=params.get(surrogate, {}),
                                acquisition=acquisition, batch_strategy=batch, batch_size=a.batch_size,
                                seeds=seeds, init_seeds=a.init_seeds if a.init_seeds is not None else seeds,
                                init_points=a.init_points, split_seed=split_seed, max_rounds=a.max_rounds,
                                max_points=a.max_points, map_change_tol=0.0, n_samples=a.n_samples,
                                random_state=a.random_state, pool="csv", simulator="lookup", labeler="csv",
                                evaluate=True, exclude=[parse_box(t) for t in a.exclude])
                jobs.append({"settings": settings,
                             "workdir": str(run_dir(a.out, surrogate, acquisition, batch, split_seed))})
    return jobs


def aggregate(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for curve_path in sorted(out.glob("*__*__*/split_*/curve.csv")):
        settings = json.loads((curve_path.parent / "settings.json").read_text())
        curve = pd.read_csv(curve_path)
        curve.insert(0, "split_seed", settings["split_seed"])
        curve.insert(0, "batch_strategy", settings["batch_strategy"])
        curve.insert(0, "acquisition", settings["acquisition"])
        curve.insert(0, "surrogate", settings["surrogate"])
        frames.append(curve)
    if not frames:
        raise FileNotFoundError(f"no curve.csv found under {out}")
    curves = pd.concat(frames, ignore_index=True)
    curves["config"] = curves["surrogate"] + " + " + curves["acquisition"] + " (" + curves["batch_strategy"] + ")"
    metrics = [c for c in curves.columns if c not in ("surrogate", "acquisition", "batch_strategy", "split_seed",
                                                      "config", "round")]
    grouped = curves.groupby(["config", "round"])[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    summary["n_runs"] = curves.groupby(["config", "round"]).size()
    return curves, summary.reset_index()


def plot(summary: pd.DataFrame, metrics: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if f"{m}_mean" in summary.columns]
    # one column per scope (all / unlabeled / excluded): 3 with --exclude, else 2
    n_cols = min(len(metrics), 3 if any(m.startswith("excluded_") for m in metrics) else 2)
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(metrics):]:
        ax.set_visible(False)
    for ax, metric in zip(flat, metrics):
        for config, g in summary.groupby("config"):
            g = g.sort_values("round")
            x, y, s = g["n_points_mean"], g[f"{metric}_mean"], g[f"{metric}_std"].fillna(0)
            ax.plot(x, y, marker="o", markersize=3, label=config)
            ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.15)
        ax.set_xlabel("labeled points")
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    n_runs = int(summary["n_runs"].max())
    fig.suptitle(f"mean ± std over {n_runs} split seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="pd", choices=sorted(TASKS))
    p.add_argument("--out", type=Path, default=Path("runs/bench"))
    p.add_argument("--surrogates", nargs="+", default=["dirichlet_gp", "random_forest"])
    p.add_argument("--surrogate-params", default="", help='JSON per surrogate, e.g. \'{"random_forest": {"n_members": 8}}\'')
    p.add_argument("--acquisitions", nargs="+", default=["bald", "random"])
    p.add_argument("--batch-strategy", default="top_k", help="for non-random acquisitions: top_k | diverse")
    p.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--init-points", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="CSV seeds used per point (default: first 3)")
    p.add_argument("--init-seeds", type=int, nargs="+", default=None, help="seeds imported for initial points")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-points", type=int, default=10**9, help="stop once this many points are labeled")
    p.add_argument("--exclude", action="append", default=[], metavar="BOX",
                   help='region never sampled but evaluated (excluded_* metrics), e.g. "Pe=100:500,DaII=0.01:10"')
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="AL runs in parallel")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="metrics to plot")
    p.add_argument("--plot-only", action="store_true", help="only aggregate and plot existing runs")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if not a.plot_only:
        jobs = make_jobs(a)
        print(f"{len(jobs)} AL runs -> {a.out}")
        if a.jobs > 1:
            with ProcessPoolExecutor(max_workers=a.jobs) as pool:
                for line in pool.map(run_one, jobs):
                    print(line)
        else:
            for job in jobs:
                print(run_one(job))

    curves, summary = aggregate(a.out)
    curves.to_csv(a.out / "bench_curves.csv", index=False)
    summary.to_csv(a.out / "bench_summary.csv", index=False)
    try:
        plot(summary, a.metrics, a.out / "bench_curves.png")
        print(f"wrote {a.out / 'bench_curves.png'}")
    except ImportError:
        print("matplotlib is not installed: skipped the plot")
    last = summary.sort_values("round").groupby("config").tail(1)
    cols = ["config", "round", "n_points_mean"] + [f"{m}_mean" for m in a.metrics if f"{m}_mean" in summary]
    print(last[cols].to_string(index=False))
    print(f"wrote {a.out / 'bench_curves.csv'} and {a.out / 'bench_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
