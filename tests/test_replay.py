"""Replay sanity check L1 (no real karst needed).

Same initial split, same pool (the CSV points), same settings, labels from the CSV:
  A = no simulation (lookup)      B = fake karst (tools/karst_stub.py) through the real code path
must give identical rounds, labels and regime maps - also with a scrambled completion order,
after an interruption, and after a walltime checkpoint.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from helpers import replay_learner

import run_al
from pnm_al.compare import compare_workdirs


class StubEnv:
    """Temporarily set fault-injection variables for the fake karst."""

    def __init__(self, **values: str) -> None:
        self.values = {f"KARST_STUB_{k.upper()}": v for k, v in values.items()}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class TestReplay(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.a = self.tmp / "A"
        result = replay_learner(self.a, simulator="lookup").run()
        self.assertEqual(result.status, "finished", result.message)

    def tearDown(self):
        self._tmp.cleanup()

    def assertSame(self, b: Path):
        comparison = compare_workdirs(self.a, b)
        self.assertTrue(comparison.identical, comparison.report())
        self.assertEqual(comparison.rounds, 4)          # round 0 (import) + 3 AL rounds

    def test_stub_karst_matches_csv_only(self):
        with StubEnv(max_delay="0.3"):                  # completion order differs from submission order
            learner = replay_learner(self.tmp / "B", workers=4)
            self.assertEqual(learner.run().status, "finished")
        runs = learner.store.runs
        sims = runs[runs["round"].astype(int) > 0]
        self.assertEqual(len(sims), 3 * 4 * 2)
        self.assertTrue((sims["status"] == "labeled").all())
        self.assertTrue(all((Path(d) / "time_evolution.out").exists() for d in sims["run_dir"]))
        self.assertSame(self.tmp / "B")

    def test_resume_after_interruption_mid_round(self):
        b = self.tmp / "B"
        first = replay_learner(b)
        first.initialize()
        first.start_round()                             # runs prepared, then the job "dies"
        self.assertEqual(len(first.store.by_status("pending")), 8)
        self.assertEqual(replay_learner(b).run().status, "finished")
        self.assertSame(b)

    def test_checkpoint_then_resume(self):
        b = self.tmp / "B"
        result = replay_learner(b).run(time_left=lambda: 10.0, round_time_guess=3600)
        self.assertEqual(result.status, "checkpoint")
        self.assertEqual(replay_learner(b).run().status, "finished")
        self.assertSame(b)

    def test_failures_are_recorded_not_dropped(self):
        b = self.tmp / "B"
        with StubEnv(fail_prob="0.3"):
            learner = replay_learner(b)
            self.assertEqual(learner.run().status, "finished")
        runs = learner.store.runs
        failed = runs[runs["status"] == "failed"]
        self.assertGreater(len(failed), 0)
        self.assertTrue(failed["warnings"].str.contains("exit code 1").all())
        self.assertFalse(compare_workdirs(self.a, b).identical)    # labels are missing, so B must differ

    def test_settings_cannot_change_on_resume(self):
        b = self.tmp / "B"
        replay_learner(b).initialize()
        with self.assertRaisesRegex(ValueError, "other settings"):
            replay_learner(b, batch_size=5)

    def test_relabel_with_placeholder(self):
        b = self.tmp / "B"
        learner = replay_learner(b)
        learner.run()
        counts = learner.relabel("placeholder")
        self.assertEqual(counts["labeled"], 24)
        runs = learner.store.runs
        sims = runs[runs["round"].astype(int) > 0]
        self.assertTrue((sims["label_source"] == "placeholder_rule_v0").all())
        self.assertTrue((runs[runs["round"].astype(int) == 0]["label_source"] == "import").all())

    def test_cli(self):
        common = ["--task", "pd", "--pool", "csv", "--init-points", "8", "--init-seeds", "21", "22",
                  "--seeds", "21", "22", "--split-seed", "3", "--batch-size", "4", "--max-rounds", "3",
                  "--map-change-tol", "0", "--n-samples", "32", "--labeler", "csv"]
        b = self.tmp / "B_cli"
        rc = run_al.main(["run", "--workdir", str(b), *common, "--karst-cmd", replay_learner.__globals__["STUB"],
                          "--workers", "4"])
        self.assertEqual(rc, 0)
        self.assertEqual(run_al.main(["compare", str(self.a), str(b)]), 0)
        self.assertEqual(run_al.main(["status", str(b)]), 0)
        self.assertEqual(run_al.main(["predict", str(b), "-o", str(self.tmp / "map.csv")]), 0)
        self.assertTrue((self.tmp / "map.csv").exists())
        c = self.tmp / "C_cli"                          # checkpoint exit code for the sbatch wrapper
        self.assertEqual(run_al.main(["run", "--workdir", str(c), *common, "--simulator", "lookup",
                                      "--time-left", "5"]), 3)


if __name__ == "__main__":
    unittest.main()
