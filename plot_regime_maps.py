#!/usr/bin/env python3
"""Regime maps of a benchmark, round by round: predicted class on the CSV grid vs the actual regime.

Figure layout:  one row per round  x  (one column per AL configuration + one 'actual' column).
Every panel is the task's CSV grid itself: rows = DaII values, columns = Pe values (change with
--row-param / --col-param), and each cell is coloured by its class (3 colours for uniform /
wormholing / compact).

"round k" = the surrogate fitted on the labels available after k AL rounds (initial points +
k batches) - the same surrogate whose metrics are in curve.csv. Simulated points show their
observed label (majority of their seeds); --model-only draws the pure surrogate prediction.
Markers (optional):
  .  point labeled by round k (in the training set)       x  point newly selected in round k

  python plot_regime_maps.py --bench runs/bench --split-seed 0
  python plot_regime_maps.py --bench runs/bench --split-seed 3 --rounds 1 2 3 4 -o maps.png

Run benchmark.py first; the configurations are the <surrogate>__<acquisition>__<batch> folders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pnm_al.evaluation import Truth, load_truth  # noqa: E402
from pnm_al.loop import ActiveLearner, load_settings  # noqa: E402
from pnm_al.space import in_boxes  # noqa: E402
from tasks import get_task  # noqa: E402

CLASS_COLORS = {"uniform": "#85B7EB", "wormholing": "#F0997B", "compact": "#7F77DD"}


def grid_index(truth: Truth, row_param: str, col_param: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted row/column values of the CSV grid and each truth point's (row, column) cell."""
    rows = np.unique(truth.params[row_param].to_numpy(float))
    cols = np.unique(truth.params[col_param].to_numpy(float))
    r_idx = np.searchsorted(rows, truth.params[row_param].to_numpy(float))
    c_idx = np.searchsorted(cols, truth.params[col_param].to_numpy(float))
    return rows, cols, r_idx, c_idx


def to_matrix(values: np.ndarray, r_idx: np.ndarray, c_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.full(shape, np.nan)
    m[r_idx, c_idx] = values
    return m


def predictions(workdir: Path, rounds: list[int], truth: Truth, model_only: bool = False) -> tuple[dict, dict, str]:
    """{round: (class per truth point, training point ids, new point ids)}, {round: (bal. acc, acc)}, label.

    By default simulated points keep their observed label (the regime map); with model_only the
    pure surrogate prediction is drawn everywhere.
    """
    settings = load_settings(workdir)
    learner = ActiveLearner(get_task(settings.task), settings, workdir, workers=1)
    runs = learner.store.runs
    run_rounds = runs["round"].astype(int)
    X = learner.space.features(truth.params)
    out = {}
    for k in rounds:
        if k > learner.state["round"] or len(learner.store.labeled_points(k)[2]) == 0:
            continue
        surrogate = learner.fit_surrogate(k)
        rng = np.random.default_rng(settings.random_state + 104729 * (k + 1))    # same draw as curve.csv
        mean = surrogate.sample_proba(X, settings.n_samples, rng).mean(axis=0)
        train = set(learner.store.labeled_points(k)[2])
        new = set(runs.loc[(run_rounds == k) & (runs["status"] == "labeled"), "point_id"])
        pred = mean.argmax(axis=1)
        if not model_only:
            observed, _, _ = learner.observed_labels(truth.point_ids, k)
            pred = learner.final_map(pred, observed)
        out[k] = (pred, train, new)
    learner.update_curve()                               # make sure curve.csv has the current columns
    curve_path = workdir / "curve.csv"
    scores = {}
    prefix = "model_" if model_only else ""
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        nan = pd.Series(np.nan, index=curve.index)
        bal = curve.get(f"{prefix}balanced_accuracy", nan)
        acc = curve.get(f"{prefix}accuracy", nan)
        scores = {int(k): (b, c) for k, b, c in zip(curve["round"], bal, acc)}
    label = f"{settings.surrogate} + {settings.acquisition}\n({settings.batch_strategy})"
    return out, scores, label


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", type=Path, default=Path("runs/bench"), help="benchmark.py --out folder")
    p.add_argument("--split-seed", type=int, default=0, help="which split to draw (maps differ per split)")
    p.add_argument("--configs", nargs="+", default=None,
                   help="configuration folders to draw (default: all <surrogate>__<acq>__<batch> in --bench)")
    p.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--row-param", default="DaII", help="parameter on the rows of every panel")
    p.add_argument("--col-param", default="Pe", help="parameter on the columns of every panel")
    p.add_argument("--no-markers", action="store_true", help="do not mark labeled / newly selected points")
    p.add_argument("--model-only", action="store_true",
                   help="draw the pure surrogate prediction (default: simulated points keep their observed label)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="PNG path (default: <bench>/regime_maps_split<seed>.png)")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    configs = a.configs or sorted(d.name for d in a.bench.glob("*__*__*") if d.is_dir())
    workdirs = [a.bench / c / f"split_{a.split_seed:03d}" for c in configs]
    missing = [str(w) for w in workdirs if not (w / "settings.json").exists()]
    if missing:
        raise SystemExit(f"no finished run in {missing}; run benchmark.py with this split seed first")

    first_settings = load_settings(workdirs[0])
    task = get_task(first_settings.task)
    truth = load_truth(task)
    excluded = in_boxes(truth.params, first_settings.exclude)          # never-sampled region (hatched)
    classes = task.classes
    rows, cols, r_idx, c_idx = grid_index(truth, a.row_param, a.col_param)
    shape = (len(rows), len(cols))
    pid_pos = {pid: i for i, pid in enumerate(truth.point_ids)}

    fallback = plt.get_cmap("tab10").colors
    colors = [CLASS_COLORS.get(c, fallback[i % 10]) for i, c in enumerate(classes)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))
    results = [predictions(w, a.rounds, truth, a.model_only) for w in workdirs]
    actual = to_matrix(truth.labels.astype(float), r_idx, c_idx, shape)

    n_rows, n_cols = len(a.rounds), len(workdirs) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.1 * n_rows + 0.8), squeeze=False)
    for i, k in enumerate(a.rounds):
        for j in range(n_cols):
            ax = axes[i][j]
            if j < len(workdirs):
                preds, scores, label = results[j]
                if k not in preds:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]), ax.set_yticks([])
                    continue
                pred, train, new = preds[k]
                ax.imshow(to_matrix(pred.astype(float), r_idx, c_idx, shape), origin="lower", cmap=cmap,
                          norm=norm, aspect="auto", interpolation="nearest")
                if not a.no_markers:
                    old = [pid_pos[p] for p in train - new if p in pid_pos]
                    fresh = [pid_pos[p] for p in new if p in pid_pos]
                    ax.scatter(c_idx[old], r_idx[old], s=5, c="black", linewidths=0)
                    ax.scatter(c_idx[fresh], r_idx[fresh], s=22, c="black", marker="x", linewidths=0.9)
                bal, acc = scores.get(k, (np.nan, np.nan))
                lines = [f"bal. acc {bal:.2f}" if pd.notna(bal) else "", f"acc {acc:.2f}" if pd.notna(acc) else ""]
                ax.set_title((label + "\n" if i == 0 else "") + "\n".join(lines), fontsize=7)
            else:
                ax.imshow(actual, origin="lower", cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                ax.set_title("actual\n\n\n" if i == 0 else "actual\n", fontsize=7)
            for r, c in zip(r_idx[excluded], c_idx[excluded]):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////",
                                       edgecolor="black", linewidth=0.3, alpha=0.6))
            ax.set_xticks(np.arange(len(cols)), minor=False)
            ax.set_yticks(np.arange(len(rows)), minor=False)
            ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.6)
            ax.tick_params(which="minor", length=0)
            ax.tick_params(which="major", length=2, labelsize=6)
            ax.set_xticklabels([f"{v:g}" for v in cols] if i == n_rows - 1 else [], rotation=90)
            ax.set_yticklabels([f"{v:g}" for v in rows] if j == 0 else [])
            if i == n_rows - 1:
                ax.set_xlabel(a.col_param, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"round {k}\n{a.row_param}", fontsize=7)

    handles = [Patch(color=col, label=cls) for col, cls in zip(colors, classes)]
    if excluded.any():
        handles.append(Patch(facecolor="none", edgecolor="black", hatch="////", label="excluded (never sampled)"))
    if not a.no_markers:
        handles += [Line2D([], [], marker="o", color="black", linestyle="", markersize=3, label="labeled"),
                    Line2D([], [], marker="x", color="black", linestyle="", markersize=5, label="new this round")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    what = "surrogate prediction only" if a.model_only else "observed at simulated points, surrogate elsewhere"
    fig.suptitle(f"split seed {a.split_seed}: class per ({a.row_param}, {a.col_param}) cell ({what}), "
                 "rows = rounds", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    output = a.output or a.bench / f"regime_maps_split{a.split_seed:03d}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
