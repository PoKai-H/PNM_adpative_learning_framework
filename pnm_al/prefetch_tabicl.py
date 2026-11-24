"""Download the TabICL checkpoint into the local cache (run on a node with internet).

    python -m pnm_al.prefetch_tabicl

Fitting and predicting on a tiny dataset makes TabICL load (and, if needed, download) its
weights, whatever the package's download API looks like. Jobs then run with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("tabicl is not installed (pip install -r requirements.txt)", file=sys.stderr)
        return 1
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    y = (X[:, 0] > 0.5).astype(int)
    clf = TabICLClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    if proba.shape != (30, 2):
        print(f"unexpected predict_proba shape {proba.shape}", file=sys.stderr)
        return 1
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"TabICL checkpoint is available (cache: {cache})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
