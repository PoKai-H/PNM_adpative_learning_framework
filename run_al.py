#!/usr/bin/env python3
"""Single entry point. Every experiment setting is a command-line option.

  run       start or resume an experiment (exit code 0 = finished, 3 = checkpoint: resubmit)
  status    progress of a workdir
  predict   regime map from all current labels -> CSV
  relabel   label all simulated runs again with another labeler (e.g. the classifier later)
  evaluate  recompute the learning curve (curve.csv) of a workdir against the CSV ground truth
  compare   compare two workdirs round by round (replay sanity check)

Examples:
  python run_al.py run --task pd --workdir runs/pd_gp --surrogate dirichlet_gp --batch-size 20
  python run_al.py run --task pd --workdir runs/replay_B --pool csv --init-points 8 --seeds 21 22 \\
         --labeler csv --karst-cmd "python tools/karst_stub.py"
  python run_al.py compare runs/replay_A runs/replay_B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnm_al.compare import compare_workdirs  # noqa: E402
from pnm_al.loop import ActiveLearner, Settings, load_settings  # noqa: E402
from pnm_al.slurm import TimeLeft  # noqa: E402
from pnm_al.space import parse_box  # noqa: E402
from tasks import TASKS, get_task  # noqa: E402

EXIT_FINISHED, EXIT_CHECKPOINT = 0, 3


def _settings_from_args(a: argparse.Namespace) -> Settings:
    params = json.loads(a.surrogate_params) if a.surrogate_params else {}
    return Settings(task=a.task, surrogate=a.surrogate, surrogate_params=params, acquisition=a.acquisition,
                    batch_strategy=a.batch_strategy, batch_size=a.batch_size, seeds=a.seeds,
                    penalty_radius=a.penalty_radius, n_samples=a.n_samples, max_rounds=a.max_rounds,
                    max_points=a.max_points, map_change_tol=a.map_change_tol, patience=a.patience,
                    pool=a.pool, n_grid=a.n_grid, max_candidates=a.max_candidates, init_points=a.init_points,
                    init_seeds=a.init_seeds, split_seed=a.split_seed, random_state=a.random_state,
                    labeler=a.labeler, simulator=a.simulator, exclude=[parse_box(t) for t in a.exclude])


def _learner_for(workdir: Path, **runtime) -> ActiveLearner:
    settings = load_settings(workdir)
    return ActiveLearner(get_task(settings.task), settings, workdir, **runtime)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="start or resume an experiment")
    r.add_argument("--task", required=True, choices=sorted(TASKS))
    r.add_argument("--workdir", required=True, type=Path)
    g = r.add_argument_group("active learning")
    g.add_argument("--surrogate", default="dirichlet_gp", help="dirichlet_gp | tabicl | random_forest")
    g.add_argument("--surrogate-params", default="", help='JSON, e.g. \'{"n_members": 8, "model": {"n_estimators": 8}}\'')
    g.add_argument("--acquisition", default="bald", help="bald | entropy | margin | random")
    g.add_argument("--batch-strategy", default="diverse", help="diverse | top_k (pure-uncertainty baseline)")
    g.add_argument("--batch-size", type=int, default=20)
    g.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43], help="seeds simulated at every new point")
    g.add_argument("--penalty-radius", type=float, default=0.05)
    g.add_argument("--n-samples", type=int, default=64)
    g.add_argument("--random-state", type=int, default=0)
    g = r.add_argument_group("stopping")
    g.add_argument("--max-rounds", type=int, default=15)
    g.add_argument("--max-points", type=int, default=10**9)
    g.add_argument("--map-change-tol", type=float, default=0.005)
    g.add_argument("--patience", type=int, default=2)
    g = r.add_argument_group("data and pool")
    g.add_argument("--pool", default="grid", help="grid | csv (the task CSV's points: replay check)")
    g.add_argument("--n-grid", type=int, default=None, help="grid points per dimension (default: task)")
    g.add_argument("--max-candidates", type=int, default=20000)
    g.add_argument("--init-points", type=int, default=None, help="initial split of N CSV points (default: all)")
    g.add_argument("--init-seeds", type=int, nargs="+", default=None, help="seeds imported for initial points")
    g.add_argument("--split-seed", type=int, default=0)
    g.add_argument("--exclude", action="append", default=[], metavar="BOX",
                   help='region never sampled, e.g. "Pe=100:500,DaII=0.01:10" (inclusive; repeat for more boxes)')
    g = r.add_argument_group("simulation and labels")
    g.add_argument("--labeler", default="placeholder", help="placeholder | csv | file.py:function | module:function")
    g.add_argument("--simulator", default="karst", help="karst | lookup (no simulation, replay check side A)")
    g.add_argument("--karst-cmd", default=None, help="karst command (default: the task's karst/bin/karst)")
    g.add_argument("--workers", type=int, default=None, help="parallel simulations (default: SLURM_CPUS_PER_TASK)")
    g.add_argument("--run-timeout", type=float, default=None, help="seconds before one simulation is killed")
    g = r.add_argument_group("walltime")
    g.add_argument("--time-left", type=float, default=None, help="seconds left in the job (default: ask squeue)")
    g.add_argument("--round-time-guess", type=float, default=3600.0, help="seconds per round before one was measured")

    s = sub.add_parser("status")
    s.add_argument("workdir", type=Path)
    pr = sub.add_parser("predict")
    pr.add_argument("workdir", type=Path)
    pr.add_argument("-o", "--output", default="regime_map.csv")
    rl = sub.add_parser("relabel")
    rl.add_argument("workdir", type=Path)
    rl.add_argument("--labeler", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("workdir", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("workdir_a", type=Path)
    c.add_argument("workdir_b", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.command == "run":
        learner = ActiveLearner(get_task(a.task), _settings_from_args(a), a.workdir, karst_cmd=a.karst_cmd,
                                workers=a.workers, run_timeout=a.run_timeout)
        result = learner.run(time_left=TimeLeft(a.time_left), round_time_guess=a.round_time_guess)
        print(f"[{result.status}] {result.message}\n{learner.summary()}")
        return EXIT_CHECKPOINT if result.status == "checkpoint" else EXIT_FINISHED
    if a.command == "status":
        print(_learner_for(a.workdir).summary())
    elif a.command == "predict":
        _learner_for(a.workdir).predict_map().to_csv(a.output, index=False)
        print(f"wrote {a.output}")
    elif a.command == "relabel":
        print(_learner_for(a.workdir).relabel(a.labeler))
        print("labels changed: run `run_al.py evaluate` to recompute curve.csv")
    elif a.command == "evaluate":
        learner = _learner_for(a.workdir)
        if not learner.s.evaluate:
            print("this experiment was run with evaluate = False")
            return 1
        curve = learner.update_curve(recompute=True)
        cols = ["round", "n_points", "n_simulations", "accuracy", "balanced_accuracy", "macro_f1",
                "unlabeled_balanced_accuracy", "unlabeled_n_eval"]
        print(curve[[c for c in cols if c in curve.columns]].to_string(index=False))
        print(f"wrote {a.workdir / 'curve.csv'}")
    elif a.command == "compare":
        result = compare_workdirs(a.workdir_a, a.workdir_b)
        print(result.report())
        return 0 if result.identical else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
