import unittest

from exp.analysitic import plot_backward_svd_parameter_grid as grid


class BackwardSvdParameterGridTests(unittest.TestCase):
    def test_two_experiment_groups_separate_dynamics_and_noise_scans(self):
        dynamics_group, noise_group = grid.make_experiment_groups()

        self.assertEqual("deterministic", dynamics_group.key)
        self.assertEqual("noise", noise_group.key)
        self.assertGreaterEqual(len(dynamics_group.combos), 4)
        self.assertGreaterEqual(len(noise_group.combos), 4)

        first_noise = dynamics_group.combos[0]
        for combo in dynamics_group.combos:
            self.assertEqual(first_noise.a, combo.a)
            self.assertEqual(first_noise.b, combo.b)
            self.assertEqual(first_noise.process_noise, combo.process_noise)
        self.assertGreater(len({(combo.lam, combo.mu) for combo in dynamics_group.combos}), 1)

        first_dynamics = noise_group.combos[0]
        for combo in noise_group.combos:
            self.assertEqual(first_dynamics.lam, combo.lam)
            self.assertEqual(first_dynamics.mu, combo.mu)
        self.assertGreater(len({(combo.a, combo.b, combo.process_noise) for combo in noise_group.combos}), 1)

    def test_noise_scan_accepts_larger_maximum_process_noise(self):
        _, noise_group = grid.make_experiment_groups(noise_process_max=0.05)

        process_noises = [combo.process_noise for combo in noise_group.combos]

        self.assertAlmostEqual(0.0002, process_noises[0])
        self.assertAlmostEqual(0.05, process_noises[-1])
        self.assertGreater(process_noises[-1] / process_noises[-2], 2.0)


if __name__ == "__main__":
    unittest.main()
