import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kinetics as k  # noqa: E402


class TestRateLaws(unittest.TestCase):
    def test_half_vmax_at_km(self):
        self.assertAlmostEqual(k.michaelis_menten(25.0, 100.0, 25.0), 50.0)

    def test_hill_reduces_to_mm_when_n_is_1(self):
        S = np.array([1.0, 5.0, 50.0])
        np.testing.assert_allclose(k.hill(S, 80, 10, 1), k.michaelis_menten(S, 80, 10))

    def test_inhibitors_reduce_to_mm_without_inhibitor(self):
        S = np.array([3.0, 30.0, 300.0])
        base = k.michaelis_menten(S, 90, 20)
        for f in (k.competitive, k.uncompetitive, k.noncompetitive):
            np.testing.assert_allclose(f(S, 0.0, 90, 20, 5), base)

    def test_competitive_inhibition_leaves_vmax_unchanged(self):
        self.assertAlmostEqual(k.competitive(1e9, 50.0, 100, 10, 5) / 100, 1.0, places=5)


class TestFitting(unittest.TestCase):
    def test_noise_free_recovery(self):
        S, v = k.synthetic_mm(120.0, 40.0, noise=0.0, n=8)
        fit = k.fit_michaelis_menten(S, v)
        self.assertAlmostEqual(fit["Vmax"], 120.0, places=4)
        self.assertAlmostEqual(fit["Km"], 40.0, places=4)

    def test_noisy_recovery_within_tolerance(self):
        S, v = k.synthetic_mm(100.0, 25.0, noise=0.03, seed=2, n=15)
        fit = k.fit_michaelis_menten(S, v)
        self.assertLess(abs(fit["Vmax"] - 100) / 100, 0.08)
        self.assertLess(abs(fit["Km"] - 25) / 25, 0.20)

    def test_lineweaver_burk_exact_on_clean_data(self):
        S, v = k.synthetic_mm(70.0, 12.0, noise=0.0, n=6)
        lb = k.lineweaver_burk_fit(S, v)
        self.assertAlmostEqual(lb["Vmax"], 70.0, places=6)
        self.assertAlmostEqual(lb["Km"], 12.0, places=6)

    def test_too_few_points_rejected(self):
        with self.assertRaises(ValueError):
            k.fit_michaelis_menten([1, 2], [1, 2])

    def test_bootstrap_ci_brackets_truth(self):
        S, v = k.synthetic_mm(100.0, 25.0, noise=0.04, seed=4, n=14)
        ci = k.bootstrap_ci(S, v, n_boot=400, seed=1)
        self.assertLess(ci["Vmax"][0], 100.0 * 1.02)
        self.assertGreater(ci["Vmax"][1], 100.0 * 0.98)
        self.assertLess(ci["Km"][0], ci["Km"][1])


class TestModelSelection(unittest.TestCase):
    def test_competitive_data_selects_competitive(self):
        rng = np.random.default_rng(5)
        Sg, Ig = np.meshgrid(np.geomspace(5, 300, 8), [0, 10, 30])
        S, I = Sg.ravel(), Ig.ravel()
        v = k.competitive(S, I, 100, 25, 12) * (1 + 0.02 * rng.standard_normal(len(S)))
        res = k.fit_inhibition(S, I, v)
        self.assertEqual(res["best"], "competitive")
        self.assertAlmostEqual(res["competitive"]["params"]["Ki"], 12, delta=3)


if __name__ == "__main__":
    unittest.main()
