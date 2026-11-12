import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bot.features import D_LEAD_SYNERGY, D_MOVE, D_TP, D_VALUE, PRIOR_MOVE
from bot.policy import Policy, Step, Trajectory, softmax


class PolicyLearningTests(unittest.TestCase):
    def test_legacy_weights_initialize_anchor_and_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "weights.json"
            weights_path.write_text(json.dumps({
                "w_move": [0.0] * D_MOVE,
                "w_tp": [0.0] * D_TP,
                "v": [0.0] * 5,
            }))
            policy = Policy(weights_path, lr=0.1, prior_pull=0.0, batch_size=1)
            phi = np.zeros((2, D_MOVE))
            phi[1, 0] = 1.0
            trajectory = Trajectory(steps=[
                Step(phi, 0, np.zeros(D_VALUE), 0.0, np.array([0.0, 2.0]))
            ])

            policy.learn(trajectory, won=True)

            self.assertTrue(np.array_equal(policy.anchor, PRIOR_MOVE))
            self.assertAlmostEqual(
                policy.w[0], -0.1 * softmax(np.array([0.0, 2.0]))[1]
            )
            saved_weights = json.loads(weights_path.read_text())
            self.assertEqual(len(saved_weights["anchor_move"]), D_MOVE)

    def test_team_preview_learning_uses_pair_synergy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = Policy(
                Path(temp_dir) / "weights.json", lr=0.05, prior_pull=0.0, batch_size=1
            )
            policy.w_tp[:] = 0.0
            rows = np.zeros((3, D_TP))
            rows[1, 0] = 1.0
            synergy = np.zeros((3, 3, D_LEAD_SYNERGY))
            synergy[0, 1, 0] = synergy[1, 0, 0] = 1.0
            trajectory = Trajectory(
                tp_rows=rows,
                tp_picks=[0, 1],
                tp_pair_synergy=synergy,
            )
            expected_gradient = (
                -1 / 3 + 1 - softmax(np.array([0.0, 0.65]))[1]
            ) / 2

            policy.learn(trajectory, won=True)

            self.assertAlmostEqual(
                policy.w_tp[0], 0.5 * 0.05 * expected_gradient
            )


if __name__ == "__main__":
    unittest.main()
