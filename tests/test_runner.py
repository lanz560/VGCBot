import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

import runLadderGames as runner
from bot.features import D_MOVE, D_VALUE
from bot.policy import Policy, Step, Trajectory


@asynccontextmanager
async def fake_server(*args):
    yield


class RunnerTests(unittest.TestCase):
    def setUp(self):
        server = patch.object(runner, 'local_showdown', fake_server)
        server.start()
        self.addCleanup(server.stop)

    def test_final_partial_batch_is_saved_and_connection_closed_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            team = root / 'team.txt'
            team.write_text('Pikachu\n- Thunderbolt\n')
            policy = Policy(root / 'weights.json', batch_size=16)
            phi = np.zeros((2, D_MOVE))
            phi[1, 0] = 1
            player = SimpleNamespace(ps_client=SimpleNamespace(stop_listening=AsyncMock()))

            async def ladder(games):
                policy.learn(Trajectory(steps=[
                    Step(phi, 1, np.zeros(D_VALUE), 0, np.zeros(2))]), True)
                raise RuntimeError('connection dropped after one game')

            player.ladder = ladder
            arguments = ['runLadderGames.py', '--team', str(team)]
            with patch.dict('os.environ', {'PS_USER': 'test', 'PS_PASS': 'test'}), \
                    patch('sys.argv', arguments), patch.object(runner, 'Policy', return_value=policy), \
                    patch.object(runner, 'LearningVGCPlayer', return_value=player):
                with self.assertRaisesRegex(RuntimeError, 'connection dropped'):
                    asyncio.run(runner.main())
            player.ps_client.stop_listening.assert_awaited_once()
            self.assertEqual(policy.pending_battles, 0)
            self.assertEqual(Policy(policy.path).battles_seen, 1)

    def test_frozen_run_does_not_write_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            team = root / 'team.txt'
            team.write_text('Pikachu\n- Thunderbolt\n')
            policy = Policy(root / 'weights.json')
            player = SimpleNamespace(ps_client=SimpleNamespace(stop_listening=AsyncMock()),
                                     ladder=AsyncMock(), n_won_battles=0, n_finished_battles=1)
            arguments = ['runLadderGames.py', '--no-learn', '--team', str(team)]
            with patch.dict('os.environ', {'PS_USER': 'test', 'PS_PASS': 'test'}), \
                    patch('sys.argv', arguments), patch.object(runner, 'Policy', return_value=policy), \
                    patch.object(runner, 'LearningVGCPlayer', return_value=player) as constructor:
                asyncio.run(runner.main())
            self.assertIs(constructor.call_args.kwargs['learn'], False)
            self.assertFalse(policy.path.exists())
            player.ps_client.stop_listening.assert_awaited_once()

    def test_local_runs_evaluator_without_credentials_and_preserves_training_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            team = root / 'team_tr_.txt'
            team.write_text('Pikachu\n- Thunderbolt\n')
            weights = root / 'weights.json'
            Policy(weights).save()
            original = weights.read_bytes()
            output = root / 'evaluation'
            calls = []

            async def block(config, candidate, opponent, seed, path, reverse):
                calls.append(config)
                self.assertEqual(config['games_per_matchup'], 3)
                self.assertEqual(seed, 7)
                self.assertEqual(candidate['sha256'], opponent['sha256'])
                self.assertTrue(candidate['prioritize_trick_room'])
                self.assertFalse(candidate['lead_synergy'])
                self.assertTrue(config['save_replays'])
                return dict(variant='candidate', opponent='opponent', seed=seed,
                            games=3, wins=2, ties=0)

            args = ['runLadderGames.py', '--local', '--team', str(team),
                    '--weights', str(weights), '--games', '3', '--seed', '7',
                    '--eval-output', str(output), '--no-lead-synergy', '--save-replays']
            with patch.dict('os.environ', {}, clear=True), patch('sys.argv', args), \
                    patch.object(runner.evaluation, 'run_block', side_effect=block), \
                    patch.object(runner, 'LearningVGCPlayer') as ladder_player:
                asyncio.run(runner.main())
            ladder_player.assert_not_called()
            self.assertEqual(len(calls), 1)
            self.assertEqual(weights.read_bytes(), original)
            self.assertTrue(json.loads((output / 'report.json').read_text())['complete'])

    def test_first_local_run_uses_priors_and_separate_opponent_team(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            team = root / 'team.txt'
            team.write_text('Pikachu\n- Thunderbolt\n')
            other = root / 'opponent.txt'
            other.write_text('Raichu\n- Thunderbolt\n')

            async def evaluate(config, output):
                candidate, opponent = config['variants'][0], config['opponents'][0]
                self.assertEqual(opponent['team'], str(other.resolve()))
                self.assertNotEqual(candidate['team'], opponent['team'])
                self.assertTrue(Path(candidate['weights']).is_file())
                self.assertEqual(candidate['weights'], opponent['weights'])
                return {'summary': {}}

            args = ['runLadderGames.py', '--local', '--team', str(team),
                    '--opponent-team', str(other), '--data-dir', str(root / 'data')]
            with patch('sys.argv', args), \
                    patch.object(runner.evaluation, 'evaluate', side_effect=evaluate) as evaluator:
                asyncio.run(runner.main())
            evaluator.assert_awaited_once()
            self.assertFalse((root / 'data' / 'weights.json').exists())

    def test_local_rejects_explicit_missing_weights_and_conflicting_challenge(self):
        with tempfile.TemporaryDirectory() as directory:
            team = Path(directory) / 'team.txt'
            team.write_text('Pikachu\n- Thunderbolt\n')
            for extra in (['--weights', str(Path(directory) / 'missing.json')],
                          ['--challenge', 'Someone']):
                args = ['runLadderGames.py', '--local', '--team', str(team), *extra]
                with patch('sys.argv', args), \
                        patch.object(runner.evaluation, 'evaluate') as evaluator, \
                        patch('sys.stderr'):
                    with self.assertRaises(SystemExit) as error:
                        asyncio.run(runner.main())
                self.assertEqual(error.exception.code, 2)
                evaluator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
