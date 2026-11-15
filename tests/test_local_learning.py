import asyncio
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import evaluate as evaluation
import VGCBot.run_ladder_games as runner
from bot.features import D_MOVE, D_VALUE
from bot.policy import Policy, Step, Trajectory


@asynccontextmanager
async def fake_server(*args):
    yield


class LocalLearningTests(unittest.TestCase):
    def play(self, root, games, fail_after=None, frozen=False, fresh=False):
        pool = root / 'teams'
        pool.mkdir()
        for name in ('fast.txt', 'slow_tr_.txt'):
            (pool / name).write_text('Pikachu\n- Protect\n')
        weights = root / 'weights.json'
        if not fresh:
            Policy(weights).save()
        initial = weights.read_bytes() if weights.exists() else None
        players, updates = [], []

        def create_player(**kwargs):
            obj = SimpleNamespace(
                policy=kwargs['policy'], learning_enabled=kwargs['learn'],
                n_finished_battles=0, n_won_battles=0, battles={},
                ps_client=SimpleNamespace(stop_listening=AsyncMock()), update_team=Mock(),
                team_id=kwargs['team_id'], team_file=kwargs['team_file'],
                prioritize_trick_room=kwargs['prioritize_trick_room'])
            obj.original_weights = obj.policy.w.copy()

            async def battle_against(other, n_battles):
                for _ in range(n_battles):
                    updates.append(players[0].policy.updates)
                    for index, player in enumerate(players):
                        won = index == 0
                        player.n_finished_battles += 1
                        player.n_won_battles += won
                        player.battles[str(player.n_finished_battles)] = SimpleNamespace(won=won)
                        if player.learning_enabled:
                            phi = np.zeros((2, D_MOVE))
                            phi[1, 0] = 1
                            psi = np.zeros(D_VALUE)
                            psi[0] = 1
                            player.policy.learn(Trajectory(steps=[
                                Step(phi, 1, psi, 0.0, np.zeros(2))]), won)
                    np.testing.assert_array_equal(players[1].policy.w, players[1].original_weights)
                    if fail_after and players[0].n_finished_battles == fail_after:
                        raise RuntimeError('connection lost')

            obj.battle_against = battle_against
            players.append(obj)
            return obj

        output = root / 'run'
        args = ['runLadderGames.py', '--local', '--team', str(pool), '--opponent-team', str(pool),
                '--weights', str(weights), '--games', str(games), '--eval-output', str(output),
                '--seed', '1', '--gamma', '0.95', '--shaping-scale', '0.1']
        if frozen:
            args.append('--no-learn')
        with patch('sys.argv', args), patch.object(runner, 'local_showdown', fake_server), \
                patch.object(evaluation, 'LearningVGCPlayer', side_effect=create_player), \
                patch('sys.stderr'):
            if fail_after:
                with self.assertRaises(SystemExit) as error:
                    asyncio.run(runner.main())
                self.assertEqual(error.exception.code, 2)
            else:
                asyncio.run(runner.main())
        for player in players:
            player.ps_client.stop_listening.assert_awaited_once()
        return players, updates, json.loads((output / 'report.json').read_text()), weights, initial

    def test_forty_games_train_in_batches_while_opponent_stays_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            players, updates, report, weights, initial = self.play(Path(temporary), 40)
            self.assertEqual([p.learning_enabled for p in players], [True, False])
            self.assertEqual(updates, [0] * 16 + [1] * 16 + [2] * 8)
            saved = Policy(weights)
            self.assertEqual((saved.battles_seen, saved.updates), (40, 3))
            self.assertFalse(np.array_equal(saved.w, players[0].original_weights))
            self.assertNotEqual(weights.read_bytes(), initial)
            self.assertEqual(players[0].policy.pending_battles, 0)
            self.assertEqual(players[0].policy.gamma, 0.95)
            self.assertEqual(players[0].policy.shaping_scale, 0.1)
            self.assertEqual(report['mode'], 'training')
            self.assertEqual(report['training']['updates'], 3)
            opponent = Path(report['config']['opponents'][0]['weights'])
            self.assertEqual(opponent.read_bytes(), initial)

    def test_failure_flushes_completed_battles_and_marks_report_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            players, _, report, weights, _ = self.play(Path(temporary), 40, fail_after=17)
            saved = Policy(weights)
            self.assertEqual((saved.battles_seen, saved.updates), (17, 2))
            self.assertEqual(players[0].policy.pending_battles, 0)
            self.assertFalse(report['complete'])
            self.assertIn('connection lost', report['error'])

    def test_new_training_destination_is_created_only_by_learning(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, report, weights, initial = self.play(Path(temporary), 1, fresh=True)
            self.assertIsNone(initial)
            self.assertEqual(Policy(weights).battles_seen, 1)
            self.assertEqual(report['training']['weights'], str(weights.resolve()))

    def test_no_learn_preserves_checkpoint_and_freezes_both_players(self):
        with tempfile.TemporaryDirectory() as temporary:
            players, _, report, weights, initial = self.play(Path(temporary), 3, frozen=True)
            self.assertEqual([p.learning_enabled for p in players], [False, False])
            self.assertEqual(weights.read_bytes(), initial)
            self.assertEqual(report['mode'], 'evaluation')
            self.assertNotIn('training', report)


if __name__ == '__main__':
    unittest.main()
