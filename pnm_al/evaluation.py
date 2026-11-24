"""Performance against the ground truth in the task CSV.

Truth per point = majority label over all seeds of that point in the CSV (ties -> the class
listed first). After every round the surrogate (fitted on the labels available then)
predicts these points; metrics are computed for

* the regime map (observed label at simulated points, surrogate elsewhere), all truth points
                                           -> accuracy, balanced_accuracy, macro_f1, f1_<class>
* the surrogate on points not simulated yet -> unlabeled_<metric>   (compare methods with these)
* the pure surrogate on all truth points    -> model_<metric>
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .labelers import normalize_label
from .task import Task


@dataclass
class Truth:
    params: pd.DataFrame        # one row per point (space names)
    point_ids: list[str]
    labels: np.ndarray          # (n_points,) class index of the majority label
    purity: np.ndarray          # (n_points,) fraction of seeds agreeing with the majority


def load_truth(task: Task) -> Truth:
    table, params, pids = task.load_data()
    classes = task.classes
    order = list(dict.fromkeys(pids))
    position = {pid: i for i, pid in enumerate(order)}
    counts = np.zeros((len(order), len(classes)))
    for pid, label in zip(pids, table[task.label_column]):
        probs = normalize_label(str(label), classes)
        for name, p in probs.items():
            counts[position[pid], classes.index(name)] += p
    first = params.assign(_pid=pids).drop_duplicates("_pid").set_index("_pid").loc[order].reset_index(drop=True)
    return Truth(first, order, counts.argmax(axis=1), counts.max(axis=1) / counts.sum(axis=1))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], prefix: str = "") -> dict:
    labels = list(range(len(classes)))
    if len(y_true) == 0:
        out = {f"{prefix}{m}": np.nan for m in ("accuracy", "balanced_accuracy", "macro_f1")}
        out |= {f"{prefix}f1_{c}": np.nan for c in classes}
        return out | {f"{prefix}n_eval": 0}
    present = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    with warnings.catch_warnings():
        # expected late in AL: the remaining unlabeled points may lack a class the model still predicts
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced = (float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1
                    else float(np.mean(y_pred == y_true)))
    out = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced accuracy = mean recall over the classes that occur in y_true
        f"{prefix}balanced_accuracy": balanced,
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        f"{prefix}n_eval": int(len(y_true)),
    }
    out |= {f"{prefix}f1_{c}": float(v) for c, v in zip(classes, per_class)}
    return out
