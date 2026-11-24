"""Task registry. Add a task: create tasks/<name>.py defining TASK, then register it here."""
from __future__ import annotations

import importlib

from pnm_al.task import Task

TASKS = {"pd": "tasks.pd_withmerge"}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return importlib.import_module(TASKS[name]).TASK
