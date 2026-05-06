import unittest

import numpy as np

import tools


class NoisyParabolicMapTests(unittest.TestCase):
    def test_zero_noise_matches_parabolic_map_recursion(self):
        trajectory = tools.simulate_noisy_parabolic_map(
            initial_state=(2.0, 3.0),
            steps=3,
            lam=0.5,
            mu=0.1,
            noise_scale=0.0,
            seed=7,
        )

        expected = np.array(
            [
                [2.0, 3.0],
                [1.0, 0.9],
                [0.5, 0.24],
                [0.25, 0.0615],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(trajectory, expected)

    def test_noise_is_reproducible_for_same_seed(self):
        first = tools.simulate_many_trajectories(
            initial_states=[(-1.0, 0.5), (1.0, -0.5)],
            steps=8,
            lam=0.8,
            mu=0.4,
            noise_scale=1e-3,
            seed=11,
        )
        second = tools.simulate_many_trajectories(
            initial_states=[(-1.0, 0.5), (1.0, -0.5)],
            steps=8,
            lam=0.8,
            mu=0.4,
            noise_scale=1e-3,
            seed=11,
        )

        self.assertEqual((2, 9, 2), first.shape)
        np.testing.assert_allclose(first, second)

    def test_noise_scan_sigma_helper_lives_in_tools(self):
        sigma = tools.make_manual_sigma_matrix(a=0.5, b=0.1)

        expected = np.array(
            [
                [0.5, 0.0, 0.1],
                [0.0, 0.5, 0.0],
                [0.1, 0.0, 0.5],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(sigma, expected)

    def test_two_stage_w_helper_lives_in_tools(self):
        A = tools.make_step_system_matrix(lam=0.1, mu=0.9)
        Sigma = tools.make_manual_sigma_matrix(a=0.5, b=0.01)

        result = tools.build_w_from_two_stage(A, Sigma, metrics_eps=1e-10, manual_r=1)

        self.assertEqual((1, 3), result["W"].shape)
        self.assertGreater(result["sv_info"]["sv_stage2"][0], 0.0)

    def test_observation_matrices_include_analytic_a_and_covariance(self):
        trajectories = tools.simulate_many_trajectories(
            initial_states=[(-1.0, 0.5), (1.0, -0.5)],
            steps=8,
            lam=0.8,
            mu=0.4,
            noise_scale=1e-3,
            seed=11,
        )

        result = tools.compute_noisy_parabolic_observation_matrices(
            trajectories=trajectories,
            lam=0.8,
            mu=0.4,
        )

        expected_a = tools.make_step_system_matrix(lam=0.8, mu=0.4)
        np.testing.assert_allclose(result["A"], expected_a)
        self.assertEqual((3, 3), result["covariance"].shape)
        np.testing.assert_allclose(result["covariance"], result["covariance"].T)
        self.assertEqual(tools.FEATURE_NAMES, result["feature_names"])


if __name__ == "__main__":
    unittest.main()
