"""Remaining walltime of the current Slurm job (for checkpointing before the limit)."""
from __future__ import annotations

import os
import subprocess
import time


def parse_slurm_time(text: str) -> float | None:
    """Parse squeue's %L / sbatch --time formats: [D-]HH:MM:SS, MM:SS, M, UNLIMITED -> seconds."""
    text = text.strip()
    if not text or text.upper() in ("UNLIMITED", "INVALID", "NOT_SET"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 1:
        h, m, s = 0, parts[0], 0          # plain minutes
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = parts[-3:]
    return float(days * 86400 + h * 3600 + m * 60 + s)


class TimeLeft:
    """Callable returning the seconds left: from --time-left, else from squeue, else None."""

    def __init__(self, seconds: float | None = None, job_id: str | None = None) -> None:
        self.start = time.monotonic()
        self.initial = seconds
        self.job_id = job_id if job_id is not None else os.environ.get("SLURM_JOB_ID")
        if self.initial is None and self.job_id:
            try:
                out = subprocess.run(["squeue", "-h", "-j", self.job_id, "-o", "%L"],
                                     capture_output=True, text=True, timeout=30)
                self.initial = parse_slurm_time(out.stdout) if out.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                self.initial = None

    def __call__(self) -> float | None:
        if self.initial is None:
            return None
        return self.initial - (time.monotonic() - self.start)
