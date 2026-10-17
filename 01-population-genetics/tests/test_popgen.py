import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import popgen  # noqa: E402


class TestHardyWeinberg(unittest.TestCase):
    def test_perfect_equilibrium(self):
        res = popgen.hardy_weinberg_test(360, 480, 160)  # p = 0.6
        self.assertAlmostEqual(res["p"], 0.6)
        self.assertAlmostEqual(res["chi2"], 0.0, places=10)
        self.assertAlmostEqual(res["p_value"], 1.0, places=6)

    def test_strong_deviation_rejected(self):
        res = popgen.hardy_weinberg_test(500, 0, 500)  # no heterozygotes
        self.assertLess(res["p_value"], 1e-10)

    def test_monomorphic_raises(self):
        with self.assertRaises(ValueError):
            popgen.hardy_weinberg_test(100, 0, 0)


class TestWrightFisher(unittest.TestCase):
    def test_trajectory_bounds_and_length(self):
        t = popgen.wright_fisher(50, 0.3, 100, rng=np.random.default_rng(0))
        self.assertEqual(len(t), 101)
        self.assertTrue(np.all((t >= 0) & (t <= 1)))

    def test_absorbing_states(self):
        t = popgen.wright_fisher(20, 0.5, 5000, rng=np.random.default_rng(3))
        self.assertIn(t[-1], (0.0, 1.0))

    def test_neutral_fixation_equals_initial_frequency(self):
        est, se = popgen.simulate_fixation_probability(N=25, s=0.0, replicates=3000, seed=7)
        self.assertLess(abs(est - 1 / 50), 4 * se + 0.005)

    def test_selection_matches_kimura(self):
        N, s = 50, 0.02
        est, se = popgen.simulate_fixation_probability(N, s, replicates=4000, seed=11)
        theory = popgen.kimura_fixation_probability(N, s)
        # diffusion is an approximation: allow 4 SE plus 10% relative slack
        self.assertLess(abs(est - theory), 4 * se + 0.1 * theory)

    def test_kimura_neutral_limit(self):
        self.assertAlmostEqual(popgen.kimura_fixation_probability(100, 0.0), 1 / 200)

    def test_heterozygosity_decays_at_theoretical_rate(self):
        N, T = 100, 50
        H = popgen.heterozygosity_decay(N, T, loci=4000, seed=5)
        expected = H[0] * (1 - 1 / (2 * N)) ** T
        self.assertLess(abs(H[-1] - expected), 0.02)


if __name__ == "__main__":
    unittest.main()
