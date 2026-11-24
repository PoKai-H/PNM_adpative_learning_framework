"""Learning-curve evaluation and the benchmark script (CSV only, no simulation)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import replay_learner

import benchmark
import run_al
from pnm_al.evaluation import classification_metrics, load_truth
from pnm_al.space import in_boxes, parse_box
from tasks import get_task


class TestTruthAndMetrics(unittest.TestCase):
    def test_pd_truth_is_one_label_per_point(self):
        truth = load_truth(get_task("pd"))
        self.assertEqual(len(truth.point_ids), 48)
        self.assertEqual(len(set(truth.point_ids)), 48)
        np.testing.assert_allclose(truth.purity, 1.0)          # all 20 seeds agree at every PD point

    def test_metrics(self):
        classes = ["a", "b", "c"]
        m = classification_metrics(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), classes)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["balanced_accuracy"], (0.5 + 1 + 1) / 3)
        self.assertEqual(m["n_eval"], 4)
        empty = classification_metrics(np.array([], dtype=int), np.array([], dtype=int), classes, prefix="u_")
        self.assertEqual(empty["u_n_eval"], 0)
        self.assertTrue(np.isnan(empty["u_accuracy"]))


class TestCurve(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.workdir = self.tmp / "A"
        self.learner = replay_learner(self.workdir, simulator="lookup")
        self.assertEqual(self.learner.run().status, "finished")

    def tearDown(self):
        self._tmp.cleanup()

    def test_curve_has_a_row_per_round(self):
        curve = pd.read_csv(self.workdir / "curve.csv")
        self.assertEqual(list(curve["round"]), [0, 1, 2, 3])
        self.assertEqual(list(curve["n_points"]), [8, 12, 16, 20])          # 8 initial + 4 per round
        self.assertEqual(list(curve["n_simulations"]), [0, 8, 16, 24])      # 4 points x 2 seeds per round
        self.assertTrue((curve["n_eval"] == 48).all())
        self.assertEqual(list(curve["unlabeled_n_eval"]), [40, 36, 32, 28])
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "unlabeled_accuracy"]:
            self.assertTrue(curve[col].between(0, 1).all(), col)

    def test_simulated_points_keep_their_observed_label(self):
        out = self.learner.predict_map(self.learner.truth.params)
        simulated = (out["observed"] != "").to_numpy()
        self.assertEqual(int(simulated.sum()), 20)                      # 8 initial + 3 rounds x 4
        self.assertTrue((out["regime"][simulated] == out["observed"][simulated]).all())
        truth = np.asarray(self.learner.task.classes)[self.learner.truth.labels]
        self.assertTrue((out["regime"].to_numpy()[simulated] == truth[simulated]).all())
        curve = pd.read_csv(self.workdir / "curve.csv")
        self.assertTrue({"model_accuracy", "model_balanced_accuracy"} <= set(curve.columns))
        # map metrics can only be better than or equal to the pure model on the same points
        self.assertTrue((curve["accuracy"] >= curve["model_accuracy"] - 1e-12).all())

    def test_old_curve_without_model_columns_is_recomputed(self):
        path = self.workdir / "curve.csv"
        old = pd.read_csv(path)
        old.drop(columns=[c for c in old.columns if c.startswith("model_")]).to_csv(path, index=False)
        curve = self.learner.update_curve()
        self.assertIn("model_accuracy", curve.columns)
        self.assertEqual(len(curve), len(old))

    def test_recompute_reproduces_the_live_curve(self):
        live = pd.read_csv(self.workdir / "curve.csv")
        recomputed = self.learner.update_curve(recompute=True)
        pd.testing.assert_frame_equal(live, recomputed, check_dtype=False, rtol=1e-12)
        self.assertEqual(run_al.main(["evaluate", str(self.workdir)]), 0)

    def test_old_settings_without_evaluate_key_still_resume(self):
        path = self.workdir / "settings.json"
        saved = json.loads(path.read_text())
        saved.pop("evaluate")
        path.write_text(json.dumps(saved))
        self.assertEqual(replay_learner(self.workdir, simulator="lookup").run().status, "finished")


class TestExclude(unittest.TestCase):
    BOX = "Pe=100:500,DaII=0.01:10"                 # top-right corner: 5 DaII x 2 Pe = 10 points

    def test_box_selects_the_top_right_corner(self):
        truth = load_truth(get_task("pd"))
        mask = in_boxes(truth.params, [parse_box(self.BOX)])
        self.assertEqual(int(mask.sum()), 10)
        self.assertEqual(sorted(set(truth.params["Pe"][mask])), [100.0, 500.0])
        with self.assertRaises(ValueError):
            parse_box("Pe=500:100")

    def test_excluded_region_is_never_sampled_but_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            learner = replay_learner(Path(tmp) / "X", simulator="lookup", exclude=[parse_box(self.BOX)],
                                     init_points=4, batch_size=2, max_rounds=17, max_points=38)
            self.assertEqual(len(learner.candidates), 38)
            self.assertEqual(learner.run().status, "finished")
            _, _, pids = learner.store.labeled_points()
            self.assertEqual(len(pids), 38)                              # every non-excluded point
            truth = learner.truth
            excluded_ids = set(np.asarray(truth.point_ids)[in_boxes(truth.params, [parse_box(self.BOX)])])
            self.assertFalse(excluded_ids & set(learner.store.runs["point_id"]))
            curve = pd.read_csv(Path(tmp) / "X" / "curve.csv")
            self.assertTrue((curve["excluded_n_eval"] == 10).all())
            self.assertEqual(int(curve["n_points"].iloc[-1]), 38)


class TestBenchmark(unittest.TestCase):
    def test_small_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench"
            args = ["--out", str(out), "--surrogates", "dirichlet_gp", "random_forest",
                    "--surrogate-params", '{"random_forest": {"n_members": 3}}',
                    "--acquisitions", "bald", "random", "--split-seeds", "0", "1",
                    "--max-rounds", "2", "--n-samples", "16"]
            self.assertEqual(benchmark.main(args), 0)
            curves = pd.read_csv(out / "bench_curves.csv")
            self.assertEqual(len(curves), 2 * 2 * 2 * 3)                  # surrogates x acq x splits x rounds 0..2
            self.assertEqual(set(curves.loc[curves["acquisition"] == "random", "batch_strategy"]), {"top_k"})
            summary = pd.read_csv(out / "bench_summary.csv")
            self.assertEqual(len(summary), 4 * 3)
            self.assertTrue((summary["n_runs"] == 2).all())
            self.assertTrue((out / "bench_curves.png").exists())
            self.assertEqual(benchmark.main(args), 0)                     # rerun: finished runs are skipped
            self.assertEqual(len(pd.read_csv(out / "bench_curves.csv")), 24)


if __name__ == "__main__":
    unittest.main()
