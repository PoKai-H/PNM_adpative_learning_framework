# PNM_al_framework

Active learning to find the **regime boundaries** (uniform / wormholing / compact) of pore-network (karst)
dissolution simulations. The karst source is part of the repo; one sbatch job runs the whole AL loop on MSI,
and every setting is a command-line option.

```
x ──► karst (parallel inside the allocation) ──► labeler ──► (x, label of every seed)
▲                                                                  │
└── acquisition (BALD, ...) ◄── posterior samples of p(regime | x) ◄── surrogate
```

## Layout

| Path | Contents |
|---|---|
| `pnm_al/loop.py` | The AL loop: initialize → every round fit → select → simulate → label; interruptible and resumable |
| `pnm_al/surrogates/` | `dirichlet_gp`, `tabicl`, `random_forest` (bootstrap ensemble) |
| `pnm_al/acquisition.py` | `bald` / `entropy` / `margin` / `random`; batch strategies `diverse` / `top_k` |
| `pnm_al/simulator.py`, `runner.py`, `karst_io.py` | Simulator caller: write configs, check config keys, run in parallel, DONE/FAILED markers, read karst output |
| `pnm_al/labelers.py` | `placeholder` (uncalibrated rule), `csv` (lookup), `file.py:function` (the future classifier) |
| `tasks/` | One file per simulation problem: parameter space, (parameters → karst config) mapping, base config, data |
| `run_al.py` | Single entry point: `run` / `status` / `predict` / `relabel` / `evaluate` / `compare` |
| `pnm_al/evaluation.py`, `benchmark.py`, `plot_regime_maps.py` | Evaluation against the CSV ground truth (`curve.csv`), strategy comparison, learning curves and regime maps |
| `slurm/run_al.sbatch` | One job runs the whole loop and resubmits itself when walltime runs out; `slurm/compare.sbatch` compares strategies |
| `karst/` | karst source (built with the collaborators' procedure; binary at `karst/bin/karst` or passed with `--karst-cmd`) |
| `setup_msi.sh`, `requirements.txt` | One-time setup of the Python environment (pinned versions) |
| `tools/karst_stub.py` | A fake karst (same command line and file formats) for testing where karst is not available |

## Local setup (laptop / workstation)

Everything except real simulations runs locally: tests, the replay check with the fake karst, `benchmark.py`
and the plots. Python **3.11 or 3.12** is recommended (pinned wheels such as torch may not exist yet for newer
versions).

**Option A: the setup script** (same as on MSI; works on macOS and Linux). It creates `./env`, installs
`requirements.txt`, downloads the TabICL checkpoint and runs the tests:

```bash
PYTHON=python3.12 bash setup_msi.sh
```

`PYTHON` is the interpreter used to create the env; any Python 3.11/3.12 works, e.g. one from a conda env
(`PYTHON=/path/to/conda/envs/<name>/bin/python`). Add `SKIP_TABICL_PREFETCH=1` to skip the TabICL download,
`SKIP_TESTS=1` to skip the tests.

**Option B: by hand.**

```bash
python3.12 -m venv env
env/bin/pip install -r requirements.txt
env/bin/pip install -e . --no-deps
env/bin/python -m unittest discover -s tests
```

Then run everything with `env/bin/python` (or `source env/bin/activate` first), e.g.
`env/bin/python benchmark.py ...`.

**Using an existing conda env.** It works as long as it has numpy, pandas, scipy, scikit-learn and matplotlib
(enough for `dirichlet_gp`, `random_forest`, the benchmark and the plots). `--surrogate tabicl` additionally needs
torch and tabicl. Versions that differ from `requirements.txt` may give slightly different results than on MSI
(most likely in the GP hyperparameter optimization).

## On MSI

**One-time setup** (login node):

```bash
PYTHON_MODULE=<a python 3.11+ module> bash setup_msi.sh   # create ./env, install requirements.txt, download TabICL weights, run tests
mkdir -p slurm_logs
```

- `setup_msi.sh` only sets up Python; it **does not build karst**. Build karst with the collaborators' procedure
  and put the binary at `karst/bin/karst` (or link it there), or pass `--karst-cmd /path/to/karst` to `run_al.py`.
- CPU-only torch is enough: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup_msi.sh`.
- Jobs install nothing: `slurm/run_al.sbatch` only activates `./env` (or `PNM_AL_ENV`) and fails immediately if it
  is missing. It sets `HF_HUB_OFFLINE=1`, so TabICL reads the weights downloaded during setup (compute nodes are
  usually offline).
- Libraries karst needs at run time (MUMPS/BLAS) go on the `module load` line of `run_al.sbatch`.

**Submit an experiment**:

```bash
sbatch slurm/run_al.sbatch --task pd --workdir runs/pd_gp --surrogate dirichlet_gp --batch-size 20 --seeds 41 42 43
```

Edit partition, account, cores and time in the header of `slurm/run_al.sbatch`. The number of parallel simulations
defaults to `--cpus-per-task`. When the remaining walltime is too short for another round (or Slurm sends USR1
10 minutes before the limit), the job stops and resubmits itself with the same arguments; the next job continues
from the workdir, and unfinished simulations are rerun.

**Other commands**:

```bash
python run_al.py status runs/pd_gp
python run_al.py predict runs/pd_gp -o map.csv
python run_al.py relabel runs/pd_gp --labeler path/to/classifier.py:predict   # once the classifier exists
sbatch --array=0-3 slurm/compare.sbatch --task pd                             # gp/tabicl x diverse/top_k
```

`python run_al.py run --help` lists all options. Settings are stored in `<workdir>/settings.json`; a resumed run
must use the same settings (otherwise it stops with an error, so data from different settings never mix). Only the
labeler can be changed, through `relabel`.

## Labeler

Interface: `label(run_dir, run) -> {regime: probability} | None`; the future classifier implements the same
interface.

- `placeholder` (default): computes `pattern_frac` (length of the dissolution pattern / N_y) and `dissolved`
  (dissolved fraction of mineral A) from the karst output and applies thresholds. **The thresholds are not
  calibrated**; the resulting regime maps only test the workflow.
- `csv`: looks the label up in the task CSV by (point, seed); used for the replay sanity check.
- `file.py:function`: `function(run_dir, params)` returns a class name or `{class: probability}`.

Every label records its `label_source`. `relabel` labels all simulated runs again with a new labeler (imported data
stays unchanged), so do not delete `<workdir>/sims/`. The task enables `if_save_ps` to keep the images for the
future classifier.

## Replay sanity check

Given the same initial points, the same sample space (the points in the task CSV) and the same settings, the loop
*with the simulator* must give the same result as the loop that *only reads the CSV*:

```bash
COMMON="--task pd --pool csv --init-points 8 --init-seeds 21 22 --seeds 21 22 --batch-size 4 --max-rounds 3 --map-change-tol 0 --labeler csv"
python run_al.py run --workdir runs/replay_A $COMMON --simulator lookup     # A: CSV only
python run_al.py run --workdir runs/replay_B $COMMON                        # B: really runs karst
python run_al.py compare runs/replay_A runs/replay_B                        # round by round; reports the first divergence
```

| Level | How | What it verifies | Where |
|---|---|---|---|
| L1 | B also uses the `csv` labeler | Loop mechanics: completion order, failures, resume, determinism | Locally with `--karst-cmd "python tools/karst_stub.py"` (in the tests); on MSI with the real karst |
| L2 | Run karst on labeled points and compare labeler output with the CSV | (Pe, DaII) → (Da, G1) mapping, base config, labeler | MSI; compare per-point majorities (`rand()` builds different networks on different platforms) |
| L3 | B uses the real labeler, compared with A | End to end | Once the classifier exists |

## Evaluation and benchmark (CSV only, no simulation)

After every round each experiment predicts all points of the task CSV with the current surrogate and compares them
with the ground truth (the majority over all seeds of each point), written to `<workdir>/curve.csv`:

| Columns | Meaning |
|---|---|
| `round`, `n_points`, `n_simulations` | Round, labeled points, simulations so far |
| `accuracy`, `balanced_accuracy`, `macro_f1`, `f1_<class>` | The final regime map: simulated points use their observed label (majority of their seeds), the rest the surrogate prediction |
| `unlabeled_*` | Only points not simulated yet: the surrogate alone. **Use these to compare methods** |
| `model_*` | The pure surrogate prediction at all points (ignoring observations); compared with the above it shows whether the surrogate "forgets" observed points |
| `excluded_*` | Only the `--exclude` region (never sampled): extrapolation |

Simulated points always keep their observed label in the regime map, in `predict` output and in the stopping rule
(change of the regime map per round). `predict` additionally writes `regime_model` (pure surrogate), `observed`
and `observed_purity` (agreement among the seeds). A `curve.csv` from an older version without `model_*` columns is
recomputed automatically, so rerunning `benchmark.py` with the same arguments updates it.

`python run_al.py evaluate <workdir>` recomputes the whole curve (e.g. after `relabel`).

**Compare uncertainty vs random and GP vs RF** (2 x 2, 10 split seeds each):

```bash
python benchmark.py --out runs/bench --surrogates dirichlet_gp random_forest \
                    --acquisitions bald random --split-seeds 0 1 2 3 4 5 6 7 8 9 --jobs 4
```

- Every (surrogate, acquisition, split seed) is an independent pool-based AL run: the pool is the CSV's 48 points,
  labels are looked up in the CSV (`--simulator lookup --labeler csv`), 8 initial points, 4 per round, 8 rounds
  (all adjustable). The split seed decides the initial points; the 10 runs are averaged.
- Each new point "simulates" the first 3 CSV seeds (`--seeds`, default 21 22 23), as a real run would simulate 3
  seeds per point; the ground truth uses all 20 seeds.
- `random` always uses `top_k` (uniform random picks); the other acquisitions default to `top_k` as well (pure
  uncertainty, no diversity). Add `--batch-strategy diverse` to compare diversity.
- Output: `bench_curves.csv` (all curves), `bench_summary.csv` (mean and std per configuration and round),
  `bench_curves.png` (learning curves vs labeled points; first row balanced accuracy, second row accuracy, last
  macro F1; columns are all points / `unlabeled_` / `excluded_`, only the first two without `--exclude`).
- After an interruption run the same command again; finished runs are skipped. `--plot-only` only re-aggregates
  and re-plots.
- Regime maps per round: `python plot_regime_maps.py --bench runs/bench --split-seed 0`.
  Each row is a round, each column a configuration plus the actual regime; every panel is the CSV grid
  (rows = DaII, columns = Pe) colored by class (simulated points use their observed label, the rest the prediction;
  `--model-only` draws the pure surrogate). `.` marks labeled points, `x` points new in that round
  (`--no-markers` turns them off). Panel titles show that round's bal. acc and acc (`model_*` with `--model-only`).

**Extrapolation experiment: leave a region out** (`--exclude`, repeat for several boxes):

```bash
python benchmark.py --out runs/bench_exclude_tr --surrogates dirichlet_gp random_forest --acquisitions bald random \
       --split-seeds 0 1 2 3 4 5 6 7 8 9 --init-points 4 --batch-size 2 --max-rounds 17 --max-points 38 \
       --exclude "Pe=100:500,DaII=0.01:10" --jobs 4
```

A box is `param=low:high` (inclusive); several parameters are joined by commas (intersection). Excluded points never
enter the pool and are never initial points, but are evaluated every round: `excluded_*` covers only this region
(extrapolation of the surrogate); `unlabeled_*` includes it too. `plot_regime_maps.py` hatches the excluded cells.

Note: PD has only 48 points and all 20 seeds agree at every point, so every method gets close to perfect quickly;
the differences show in the first rounds and only the average over several split seeds is meaningful.

## Adding a task

Create `tasks/<name>.py` defining `TASK = Task(...)` (parameter space, classes, data CSV, base config,
`overrides(params, seed) -> dict`) and register it in `tasks/__init__.py`. Override keys are checked against the key
list of `karst/src/setup.cc` before running; a key karst does not know is an error.

## Tests

```bash
python -m unittest discover -s tests
```

Covers the L1 replay (fake karst vs CSV only, including shuffled completion order, interruption, checkpoint and
resume, fault injection), key checks, timeouts, sbatch self-resubmission, evaluation, `--exclude` and the benchmark.

## Not verified yet / to confirm

- `setup_msi.sh` has not been run on MSI (Python module name, TabICL weight download).
- Building karst and the modules it needs at run time follow the collaborators' procedure and are not part of
  `run_al.sbatch` yet.
- The real karst output format is inferred from the source only; after the first run on MSI keep that run folder as
  a test fixture.
- The PD mapping (Pe, DaII) → (Da, G1) is an assumption (`tasks/pd_withmerge.py`), to be confirmed by L2.
- On which platform and with which karst version the original data were produced; this decides whether L2 can
  compare per seed.
