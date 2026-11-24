from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pnm_al import ActiveLearner, Settings  # noqa: E402
from tasks import get_task  # noqa: E402

STUB = f"{sys.executable} {REPO / 'tools' / 'karst_stub.py'}"
CLASSES = ["uniform", "wormholing", "compact"]

# replay setting: pool = the 48 CSV points, 8 initial points, CSV seeds so every run has a label
REPLAY = dict(task="pd", pool="csv", init_points=8, init_seeds=[21, 22], seeds=[21, 22], split_seed=3,
              batch_size=4, max_rounds=3, map_change_tol=0.0, n_samples=32, labeler="csv")


def replay_learner(workdir: Path, simulator: str = "karst", workers: int = 4, run_timeout: float | None = None,
                   **overrides) -> ActiveLearner:
    settings = Settings(**(REPLAY | {"simulator": simulator} | overrides))
    return ActiveLearner(get_task("pd"), settings, workdir, karst_cmd=STUB if simulator == "karst" else None,
                         workers=workers, run_timeout=run_timeout)
