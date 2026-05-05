import subprocess
import textwrap
import unittest


class ToolsImportTests(unittest.TestCase):
    def test_noisy_parabolic_helpers_import_in_py311_environment(self):
        code = textwrap.dedent(
            """
            from tools import (
                compute_noisy_parabolic_observation_matrices,
                configure_noisy_parabolic_publication_style,
                make_initial_state_grid,
                plot_noisy_parabolic_observation_matrices,
                plot_noisy_parabolic_publication_figure,
                simulate_many_trajectories,
                summarize_noisy_parabolic_trajectories,
            )
            print("ok")
            """
        )

        result = subprocess.run(
            ["/opt/anaconda3/envs/py311/bin/python", "-c", code],
            cwd=".",
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
