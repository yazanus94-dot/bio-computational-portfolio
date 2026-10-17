import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import epidemic as ep  # noqa: E402


class TestAnalytic(unittest.TestCase):
    def test_r0(self):
        self.assertAlmostEqual(ep.r0(0.6, 0.2), 3.0)

    def test_herd_immunity_threshold(self):
        self.assertAlmostEqual(ep.herd_immunity_threshold(4.0), 0.75)
        self.assertEqual(ep.herd_immunity_threshold(0.8), 0.0)

    def test_final_size_known_value(self):
        # R0 = 2 -> z ~ 0.7968 (classic result)
        self.assertAlmostEqual(ep.final_size(2.0), 0.79681, places=4)

    def test_final_size_subcritical_is_zero(self):
        self.assertEqual(ep.final_size(0.9), 0.0)

    def test_final_size_satisfies_equation(self):
        for R0 in (1.1, 1.5, 3.0, 6.0):
            z = ep.final_size(R0)
            self.assertAlmostEqual(z, 1 - np.exp(-R0 * z), places=8)


class TestSimulation(unittest.TestCase):
    def test_population_conserved(self):
        _, y = ep.simulate_seir(0.6, 1 / 3, 0.2, days=200)
        np.testing.assert_allclose(y.sum(axis=0), 1.0, atol=1e-7)

    def test_compartments_non_negative(self):
        _, y = ep.simulate_seir(0.6, 1 / 3, 0.2, days=200)
        self.assertGreaterEqual(y.min(), -1e-9)

    def test_ode_attack_rate_matches_final_size(self):
        beta, gamma = 0.6, 0.2
        _, y = ep.simulate_seir(beta, 1 / 3, gamma, i0=1e-7, days=1500, n_points=3000)
        self.assertAlmostEqual(y[3][-1], ep.final_size(beta / gamma), places=2)

    def test_vaccination_above_threshold_prevents_outbreak(self):
        beta, gamma = 0.6, 0.2
        hit = ep.herd_immunity_threshold(3.0)
        _, y = ep.simulate_seir(beta, 1 / 3, gamma, i0=1e-5, vaccinated=hit + 0.05, days=400)
        self.assertLess(y[3][-1] - (hit + 0.05), 0.01)  # almost no additional infections

    def test_invalid_vaccination_rejected(self):
        with self.assertRaises(ValueError):
            ep.simulate_seir(0.6, 1 / 3, 0.2, vaccinated=1.0)


class TestInferenceAndStochastic(unittest.TestCase):
    def test_parameter_recovery_noise_free(self):
        t, y = ep.simulate_seir(0.55, 1 / 3, 0.2, i0=1e-3, days=120, n_points=121)
        beta, gamma = ep.fit_beta_gamma(t, y[2], 1 / 3, 1e-3, 120)
        self.assertAlmostEqual(beta, 0.55, places=2)
        self.assertAlmostEqual(gamma, 0.2, places=2)

    def test_stochastic_major_outbreak_probability(self):
        # Branching-process theory: P(major) = 1 - 1/R0 = 0.5 for R0 = 2
        p = ep.major_outbreak_probability(2.0, N=1000, replicates=300, seed=3)
        self.assertAlmostEqual(p, 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
