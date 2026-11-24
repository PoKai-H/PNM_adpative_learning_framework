"""Unit tests: python -m unittest discover -s tests"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from helpers import CLASSES, REPO, STUB

from pnm_al.acquisition import bald, choose_batch, entropy, margin, select_batch
from pnm_al.karst_io import format_karst_value, known_config_keys, parse_karst_config
from pnm_al.labelers import PlaceholderLabeler
from pnm_al.runner import execute
from pnm_al.simulator import KarstSimulator
from pnm_al.slurm import parse_slurm_time
from pnm_al.space import Dimension, ParameterSpace
from pnm_al.store import ResultStore
from pnm_al.surrogates import build_surrogate

SETUP_CC = REPO / "karst" / "src" / "setup.cc"
BASE_CONFIG = REPO / "karst" / "simulation_setups" / "2D" / "config_small_diss.txt"


class TestSpace(unittest.TestCase):
    def test_grid_features_ids(self):
        space = ParameterSpace([Dimension("Pe", 0.01, 100, n_grid=5), Dimension("x", 0, 1, log=False, n_grid=3)],
                               {"r": lambda p: np.log10(p["Pe"]) * p["x"]})
        grid = space.grid()
        self.assertEqual(len(grid), 15)
        X = space.features(grid)
        self.assertEqual(X.shape, (15, 3))
        self.assertTrue(np.all((X > -1e-12) & (X < 1 + 1e-12)))
        self.assertEqual(space.point_id({"Pe": 0.009999999999, "x": 0.5}), "Pe=0.01|x=0.5")
        self.assertEqual(space.with_grid(4).grid_size(), 16)


class TestAcquisition(unittest.TestCase):
    def test_scores(self):
        sure = np.tile([[[1.0, 0.0, 0.0]]], (4, 1, 1))
        disagree = np.array([[[1, 0, 0]], [[0, 1, 0]], [[1, 0, 0]], [[0, 1, 0]]], dtype=float)
        noisy = np.tile([[[0.5, 0.5, 0.0]]], (4, 1, 1))
        P = np.concatenate([sure, disagree, noisy], axis=1)
        b = bald(P)
        self.assertLess(b[0], 1e-6)
        self.assertGreater(b[1], 0.5)            # model disagreement -> high BALD
        self.assertLess(b[2], 1e-6)              # pure aleatoric noise -> BALD ~ 0
        self.assertGreater(entropy(P)[2], 0.5)
        self.assertAlmostEqual(margin(P)[2], 1.0)

    def test_batch_strategies(self):
        X, s = np.array([[0.0], [0.01], [1.0]]), np.array([1.0, 0.99, 0.5])
        self.assertEqual(select_batch(s, X, 2, radius=0.1), [0, 2])
        self.assertEqual(choose_batch("diverse", s, X, 2, 0.1), [0, 2])
        self.assertEqual(choose_batch("top_k", s, X, 2, 0.1), [0, 1])
        self.assertEqual(choose_batch("top_k", s, X, 2, 0.1, available=np.array([False, True, True])), [1, 2])
        with self.assertRaises(ValueError):
            choose_batch("nope", s, X, 1, 0.1)


class TestStore(unittest.TestCase):
    def test_aggregation_is_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), ["Pe"], CLASSES)
            labels = ["uniform", "uniform", "compact"]
            store.add_runs([{"run_id": f"a{i}", "point_id": "Pe=1", "round": 0, "seed": i, "status": "labeled",
                             "Pe": 1.0, "label_probs": json.dumps({lab: 1.0})} for i, lab in enumerate(labels)][::-1])
            _, counts, pids = ResultStore(Path(tmp), ["Pe"], CLASSES).labeled_points()
            self.assertEqual(pids, ["Pe=1"])
            np.testing.assert_allclose(counts, [[2, 0, 1]])


class TestSurrogates(unittest.TestCase):
    def test_fit_and_sample(self):
        rng = np.random.default_rng(0)
        X = rng.random((30, 2))
        counts = np.zeros((30, 3))
        counts[np.arange(30), (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.8)] = 3
        for kind, params in [("dirichlet_gp", {}), ("random_forest", {"n_members": 4})]:
            model = build_surrogate(kind, 3, params)
            model.fit(X, counts)
            P = model.sample_proba(X, 16, rng)
            np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-9)
            self.assertGreater(np.mean(P.mean(0).argmax(1) == counts.argmax(1)), 0.9, kind)


class TestKarstContract(unittest.TestCase):
    def test_known_keys_come_from_setup_cc(self):
        keys = known_config_keys(SETUP_CC)
        self.assertGreater(len(keys), 100)
        self.assertTrue({"Da", "G1", "random_seed", "type_of_merging", "if_save_ps", "d_min_factor"} <= keys)
        self.assertNotIn("d_min", keys)          # karst silently ignores d_min = ...

    def test_parser_follows_karst_rules(self):
        values, problems = parse_karst_config("# c\nDa = 1\n\nd_min = 0.1\nDa = 2 # later wins\nbroken line\n",
                                              known_config_keys(SETUP_CC))
        self.assertEqual(values, {"Da": "2"})
        self.assertEqual(problems, ['Problem with parsing name "d_min" in line 4', "Problem with parsing line nr 6"])
        self.assertEqual(format_karst_value("x", True), "true")
        self.assertEqual(format_karst_value("x", 41), "41")
        self.assertEqual(format_karst_value("x", 0.1 / 3), "0.03333333333")
        with self.assertRaises(ValueError):
            format_karst_value("x", "two words")

    def test_karst_simulator_checks_keys(self):
        keys = known_config_keys(SETUP_CC)
        with self.assertLogs("pnm_al.simulator", "WARNING"):          # base config contains d_min
            ok = KarstSimulator(STUB, BASE_CONFIG, lambda p, s: {"Da": p["Pe"], "random_seed": s}, ["Pe"], keys)
        self.assertIn("d_min", ok.base_unknown)
        self.assertEqual(ok.config_lines({"Pe": 0.5, "seed": 7}), ["Da = 0.5", "random_seed = 7"])
        bad = KarstSimulator(STUB, BASE_CONFIG, lambda p, s: {"d_min": 0.01}, ["Pe"], keys)
        with self.assertRaisesRegex(ValueError, "silently ignore"):
            bad.config_lines({"Pe": 1.0, "seed": 1})
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_errors.txt").write_text(
                'Problem with parsing name "d_min" in line 12\nProblem with parsing name "Daa" in line 140\n')
            self.assertEqual(ok.check(run_dir), ['Problem with parsing name "Daa" in line 140'])

    def test_stub_reports_unknown_override_and_run_fails_check(self):
        keys = known_config_keys(SETUP_CC)
        sim = KarstSimulator(STUB, BASE_CONFIG, lambda p, s: {"Da": 0.001, "G1": 1.0, "random_seed": s}, ["Pe"], keys)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            sim.prepare({"run_id": "x", "point_id": "p", "seed": 3, "Pe": 1.0}, run_dir)
            with open(run_dir / "config.txt", "a") as f:
                f.write("Daa = 3\n")                                   # a typo karst would ignore
            res = execute([run_dir], sim.command(), workers=1)
            self.assertEqual(res[run_dir].status, "done")
            self.assertEqual(len(sim.check(run_dir)), 1)
            self.assertIn("Daa", sim.check(run_dir)[0])
            self.assertEqual(PlaceholderLabeler(CLASSES).label(run_dir, {}), {"uniform": 1.0})


class TestRunner(unittest.TestCase):
    def test_timeout_kills_and_done_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            slow, fast = Path(tmp) / "slow", Path(tmp) / "fast"
            slow.mkdir(), fast.mkdir()
            (fast / "DONE").touch()
            start = time.monotonic()
            res = execute([slow, fast], "sleep 30", workers=2, timeout=1.0)
            self.assertLess(time.monotonic() - start, 10)
            self.assertEqual(res[slow].status, "failed")
            self.assertIn("timeout", (slow / "FAILED").read_text())
            self.assertEqual(res[fast].message, "already finished")

    def test_slurm_time_parsing(self):
        self.assertEqual(parse_slurm_time("1-02:03:04"), 93784)
        self.assertEqual(parse_slurm_time("10:00"), 600)
        self.assertEqual(parse_slurm_time("45"), 2700)
        self.assertIsNone(parse_slurm_time("UNLIMITED"))


class TestSbatchScript(unittest.TestCase):
    def test_resubmits_on_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calls = tmp / "sbatch_calls.txt"
            fake_sbatch = tmp / "sbatch"
            fake_sbatch.write_text(f'#!/bin/bash\necho "$@" >> {calls}\n')
            fake_sbatch.chmod(0o755)
            for code, expect in [(3, True), (0, False)]:
                fake_python = tmp / f"python{code}"
                fake_python.write_text(f"#!/bin/bash\nexit {code}\n")
                fake_python.chmod(0o755)
                env = {"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "SBATCH": str(fake_sbatch),
                       "SLURM_SUBMIT_DIR": str(tmp), "PNM_AL_SBATCH_SCRIPT": "slurm/run_al.sbatch"}
                calls.unlink(missing_ok=True)
                rc = subprocess.run(["bash", str(REPO / "slurm" / "run_al.sbatch"), "--task", "pd", "--workdir", "w"],
                                    env=env, capture_output=True, text=True).returncode
                self.assertEqual(rc, 0)
                self.assertEqual(calls.exists(), expect)
                if expect:
                    self.assertIn("--task pd --workdir w", calls.read_text())

    def test_missing_env_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PATH": "/usr/bin:/bin", "SLURM_SUBMIT_DIR": tmp, "PNM_AL_ENV": str(Path(tmp) / "no_env")}
            out = subprocess.run(["bash", str(REPO / "slurm" / "run_al.sbatch"), "--task", "pd"],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)
            self.assertIn("setup_msi.sh", out.stdout)

    def test_setup_script_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(REPO / "setup_msi.sh")]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
