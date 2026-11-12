import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bot.features import D_MOVE, D_TP, D_VALUE, PRIOR_MOVE, SINGLE_FEATURES, PAIR_FEATURES
from bot.policy import Policy, Step, Trajectory


def trajectory(chosen=1, potentials=(0.0,)):
    phi = np.zeros((2, D_MOVE))
    phi[1, 0] = 1
    return Trajectory(steps=[
        Step(phi.copy(), chosen, np.zeros(D_VALUE), pot, np.zeros(2))
        for pot in potentials
    ])


class BatchLearningTests(unittest.TestCase):
    def test_parameters_frozen_until_full_batch_and_combined_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(Path(directory) / 'weights.json', batch_size=2,
                            lr=0.1, prior_pull=0)
            policy.w[:] = 0
            before = (policy.w.copy(), policy.w_tp.copy(), policy.v.copy())
            first = policy.learn(trajectory(chosen=1), True)
            self.assertFalse(first['updated'])
            self.assertEqual(policy.pending_battles, 1)
            for actual, expected in zip((policy.w, policy.w_tp, policy.v), before):
                np.testing.assert_array_equal(actual, expected)
            second = policy.learn(trajectory(chosen=0), False)
            self.assertTrue(second['updated'])
            self.assertAlmostEqual(policy.w[0], 0.05)
            self.assertEqual(policy.pending_battles, 0)
            self.assertEqual((policy.battles_seen, policy.wins, policy.updates), (2, 1, 1))
            self.assertGreaterEqual(second['mean_kl'], 0)
            self.assertAlmostEqual(second['mean_entropy'], np.log(2))

    def test_partial_batch_flush_persists_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.json'
            policy = Policy(path, batch_size=16)
            policy.learn(trajectory(), True)
            stats = policy.flush()
            self.assertEqual(stats['batch_battles'], 1)
            restored = Policy(path)
            np.testing.assert_array_equal(restored.w, policy.w)
            np.testing.assert_array_equal(restored.v, policy.v)
            self.assertEqual(restored.battles_seen, 1)
            self.assertEqual(policy.flush(), {})
            self.assertEqual(policy.updates, 1)

    def test_discounted_shaping_telescopes_and_intermediate_hp_does_not_change_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(Path(directory) / 'weights.json', gamma=0.9, shaping_scale=0.25)
            traj = trajectory(potentials=(0.2, -0.8, 0.7))
            expected = 0.9 ** 2 - 0.25 * 0.2
            self.assertAlmostEqual(policy.returns(traj, True)[0], expected)
            traj.steps[1].pot = -0.1
            traj.steps[2].pot = 0.2
            self.assertAlmostEqual(policy.returns(traj, True)[0], expected)
            policy.shaping_scale = 0
            np.testing.assert_allclose(policy.returns(traj, False), [-0.81, -0.9, -1])

    def test_legacy_checkpoint_migrates_all_weights_and_extends_value_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.json'
            names = SINGLE_FEATURES + PAIR_FEATURES
            path.write_text(json.dumps({
                'w_move': list(range(30)), 'anchor_move': [0.5] * 30,
                'feature_names': names[:30], 'w_tp': [0.2] * D_TP,
                'v': [1, 2, 3, 4, 5], 'battles_seen': 40, 'wins': 10,
            }))
            policy = Policy(path)
            np.testing.assert_array_equal(policy.w[:30], np.arange(30))
            np.testing.assert_array_equal(policy.w[30:], PRIOR_MOVE[30:])
            np.testing.assert_array_equal(policy.anchor[:30], [0.5] * 30)
            np.testing.assert_array_equal(policy.v[:5], [1, 2, 3, 4, 5])
            np.testing.assert_array_equal(policy.v[5:], np.zeros(D_VALUE - 5))
            self.assertEqual(policy.battles_seen, 40)
            # Name-based restoration must also survive different saved ordering.
            blob = json.loads(path.read_text())
            blob['feature_names'].reverse()
            blob['w_move'].reverse()
            path.write_text(json.dumps(blob))
            np.testing.assert_array_equal(Policy(path).w, policy.w)

    def test_preview_batch_uses_fixed_weights_and_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(Path(directory) / 'weights.json', batch_size=2, prior_pull=0)
            policy.w_tp[:] = 0
            rows = np.zeros((2, D_TP))
            rows[1, 0] = 1
            policy.learn(Trajectory(tp_rows=rows, tp_picks=[1]), True)
            self.assertEqual(policy.w_tp[0], 0)
            policy.learn(Trajectory(tp_rows=rows, tp_picks=[0]), False)
            self.assertAlmostEqual(policy.w_tp[0], 0.0125)
            self.assertEqual(policy.tp_baseline, 0)

    def test_empty_trajectory_and_nonfinite_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(Path(directory) / 'weights.json', batch_size=1)
            stats = policy.learn(Trajectory(), None)
            self.assertEqual(stats['batch_steps'], 0)
            original = policy.w.copy()
            bad = trajectory()
            bad.steps[0].phi[0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                policy.learn(bad, True)
            np.testing.assert_array_equal(policy.w, original)

    def test_failed_save_rolls_back_and_partial_batch_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(Path(directory) / 'weights.json', batch_size=16)
            policy.learn(trajectory(), True)
            before = (policy.w.copy(), policy.v.copy(), policy.w_tp.copy())
            with patch.object(policy, 'save', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    policy.flush()
            for actual, expected in zip((policy.w, policy.v, policy.w_tp), before):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(policy.updates, 0)
            self.assertEqual(policy.pending_battles, 1)
            self.assertTrue(policy.flush()['updated'])
            self.assertEqual(policy.updates, 1)


if __name__ == '__main__':
    unittest.main()
