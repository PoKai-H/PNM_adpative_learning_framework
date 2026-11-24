"""Dissolution regime map (uniform / wormholing / compact) in (Pe, DaII), PD_withMerge data.

Data: 960 hand-labeled runs = 6 Pe x 8 DaII x 20 seeds (21-40), d0 = 0.1, with merging.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pnm_al.space import Dimension, ParameterSpace
from pnm_al.task import Task

REPO = Path(__file__).resolve().parents[1]
KARST = REPO / "karst"


def overrides(p: dict[str, float], seed: int) -> dict:
    """(Pe, DaII) + seed -> karst config values.

    ASSUMED mapping Da = DaII / Pe (DaI), G1 = DaII - confirm it matches the original runs
    (replay check L2). Outputs needed by a later classifier are kept (if_save_ps).
    """
    return {
        "Da": p["DaII"] / p["Pe"],
        "G1": p["DaII"],
        "d0": 0.1,
        "random_seed": seed,
        "if_precipitation": False,
        "if_redissolution": False,
        "if_track_grains": True,
        "type_of_merging": "merge_empty_grains",
        "if_full_dissolution": True,
        "if_save_ps": True,          # pictures for labeling / the future classifier
        "if_save_vtk": False,
    }


TASK = Task(
    name="pd",
    description=__doc__,
    classes=["uniform", "wormholing", "compact"],
    space=ParameterSpace(
        [Dimension("Pe", 0.01, 500.0, log=True, n_grid=100),
         Dimension("DaII", 1e-5, 10.0, log=True, n_grid=100)],
        # the uniform/wormholing boundary sits at DaII/Pe ~ 0.01: give the surrogate that axis directly
        derived={"log_DaI": lambda p: np.log10(p["DaII"] / p["Pe"])},
    ),
    data_csv=REPO / "data" / "PD_withMerge_seed_21-40_regimeLabel.csv",
    param_columns={"Pe": "Pe", "DaII": "DaII"},
    seed_column="seed",
    label_column="regime",
    karst_dir=KARST,
    base_config=KARST / "simulation_setups" / "2D" / "config_small_diss.txt",
    overrides=overrides,
)
