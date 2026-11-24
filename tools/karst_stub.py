#!/usr/bin/env python3
"""Fake karst for testing without the real simulator.

Same command line (``-C config.txt -V single_run -P 0``) and the same file contract:
  * parses config.txt with karst's rules and karst's real key list (karst/src/setup.cc),
    reporting unknown keys on stderr as "Problem with parsing name ..." like karst does,
  * writes time_evolution.out and pattern_analysis.out in karst's column layout.
The outputs follow a toy regime rule in (Da, G1) so the placeholder labeler gets
plausible inputs. NOT a physical model.

Fault injection (environment, deterministic per run directory):
  KARST_STUB_MAX_DELAY  random sleep up to this many seconds (scrambles completion order)
  KARST_STUB_FAIL_PROB  probability of exiting with code 1
  KARST_STUB_HANG_PROB  probability of hanging (for --run-timeout tests)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnm_al.karst_io import PATTERN_COLUMNS, TIME_EVOLUTION_COLUMNS, known_config_keys, parse_karst_config  # noqa: E402


def toy_outputs(da: float, g1: float, n_y: float) -> tuple[float, float]:
    """(pattern length / N_y, dissolved fraction) from a toy rule; Pe = G1 / Da here."""
    pe = g1 / da
    if math.log10(da) < -2.0:
        return 1.0, 0.8                         # uniform: everything dissolves
    if math.log10(pe) < -1.5 and math.log10(g1) > -1.5:
        return 0.1, 0.3                         # compact: short pattern at the inlet
    return 1.0, 0.2                             # wormholing: long thin pattern


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-C", dest="config", default="config.txt")
    ap.add_argument("-V", dest="version", default="single_run")
    ap.add_argument("-P", dest="pictures", default="1")
    args, _ = ap.parse_known_args()

    text = Path(args.config).read_text()
    values, problems = parse_karst_config(text, known_config_keys(REPO / "karst" / "src" / "setup.cc"))
    for name, value in values.items():
        print(f"Setting {name} = {value}", file=sys.stderr)
    for problem in problems:
        print(problem, file=sys.stderr)

    rng = random.Random(f"{Path.cwd().name}|{values.get('random_seed', '')}")
    time.sleep(rng.random() * float(os.environ.get("KARST_STUB_MAX_DELAY", "0")))
    if rng.random() < float(os.environ.get("KARST_STUB_HANG_PROB", "0")):
        time.sleep(3600)
    if rng.random() < float(os.environ.get("KARST_STUB_FAIL_PROB", "0")):
        print("stub: injected failure", file=sys.stderr)
        return 1

    n_y = float(values.get("N_y", "10").rstrip(";"))
    pattern_frac, dissolved = toy_outputs(float(values.get("Da", "1")), float(values.get("G1", "1")), n_y)
    va0, steps = 1000.0, 5
    with open("time_evolution.out", "w") as f:
        f.write("#" + "".join(f"{c:>15}" for c in TIME_EVOLUTION_COLUMNS) + "\n#  ---\n")
        for s in range(steps + 1):
            va = va0 * (1 - dissolved * s / steps)
            row = [s, s * 0.1, s * 0.01, 100, 99, va, 0, 0, s, 0, 1, 0.3 + 0.1 * s / steps, 0.1, 0.1, 0]
            f.write("".join(f"{v:>15.6g}" for v in row) + "\n")
    with open("pattern_analysis.out", "w") as f:
        f.write("#" + "".join(f"{c:>15}" for c in PATTERN_COLUMNS) + "\n#  ---\n")
        for s in range(steps + 1):
            row = [s, s * 0.1, s * 0.01, pattern_frac * n_y * s / steps, 0, 1, 0]
            f.write("".join(f"{v:>15.6g}" for v in row) + "\n")
    print("Network is being deleted.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
