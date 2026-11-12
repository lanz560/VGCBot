import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

import evaluate as E
from bot.features import D_MOVE, D_TP


class EvaluationTests(unittest.TestCase):
    def config(self, directory):
        root = Path(directory)
        (root / 'team.txt').write_text('Pikachu\n- Thunderbolt\n')
        (root / 'weights.json').write_text(json.dumps({
            'w_move': [0] * D_MOVE, 'w_tp': [0] * D_TP, 'v': [0] * 5,
        }))
        config = {'format': 'gen9doublescustomgame', 'games_per_matchup': 2,
                  'seeds': [1, 2],
                  'variants': [{'name': name, 'team': 'team.txt', 'weights': 'weights.json'}
                               for name in ['baseline', 'candidate']],
                  'opponents': [{'name': 'opponent', 'team': 'team.txt', 'weights': 'weights.json'}]}
        path = root / 'config.json'
        path.write_text(json.dumps(config))
        return E.load_config(path)

    def test_missing_checkpoints_fail_before_networking(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            Path(config['variants'][0]['weights']).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing weights'):
                E.load_config(Path(directory) / 'config.json')

    def test_schedule_snapshot_and_report_without_server(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            output = Path(directory) / 'evaluation'
            calls = []

            async def block(config, variant, opponent, seed, path, reverse):
                calls.append((variant['name'], seed, reverse))
                # The harness must operate on copies even if training changes originals.
                self.assertNotEqual(variant['weights'], variant['sources']['weights'])
                self.assertTrue(Path(variant['weights']).is_file())
                return dict(variant=variant['name'], opponent=opponent['name'], seed=seed,
                            games=2, wins=1, ties=0)

            with patch.object(E, 'run_block', side_effect=block):
                report = asyncio.run(E.evaluate(config, output))
            self.assertTrue(report['complete'])
            self.assertEqual(calls, [('baseline', 1, False), ('candidate', 1, False),
                                     ('candidate', 2, True), ('baseline', 2, True)])
            self.assertEqual(report['summary']['candidate']['games'], 4)
            self.assertEqual(report['summary']['candidate']['win_rate'], 0.5)
            self.assertTrue(json.loads((output / 'report.json').read_text())['complete'])
            self.assertIn('bot/policy.py', report['source_sha256'])

    def test_run_block_freezes_players_and_cleans_up_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            output = Path(directory) / 'inputs-run'
            output.mkdir()
            frozen = E.snapshot_inputs(config, output)
            players = []

            def player(**kwargs):
                self.assertIs(kwargs['learn'], False)
                before = kwargs['policy'].w.copy()
                obj = SimpleNamespace(
                    n_finished_battles=2, n_won_battles=1,
                    battles={'a': SimpleNamespace(won=True), 'b': SimpleNamespace(won=None)},
                    ps_client=SimpleNamespace(stop_listening=AsyncMock()),
                    battle_against=AsyncMock(), policy=kwargs['policy'], before=before)
                players.append(obj)
                return obj

            with patch.object(E, 'LearningVGCPlayer', side_effect=player):
                result = asyncio.run(E.run_block(
                    frozen, frozen['variants'][0], frozen['opponents'][0], 1,
                    output / 'block', reverse=True))
            self.assertEqual(result['ties'], 1)
            players[1].battle_against.assert_awaited_once_with(players[0], n_battles=2)
            for p in players:
                p.ps_client.stop_listening.assert_awaited_once()
                np.testing.assert_array_equal(p.before, p.policy.w)

    def test_failure_leaves_incomplete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            output = Path(directory) / 'failure'
            with patch.object(E, 'run_block', side_effect=RuntimeError('server unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'server unavailable'):
                    asyncio.run(E.evaluate(config, output))
            report = json.loads((output / 'report.json').read_text())
            self.assertFalse(report['complete'])
            self.assertIn('server unavailable', report['error'])

    def test_win_interval_and_tie_accounting(self):
        stats = E.summarize([dict(games=10, wins=4, ties=2)])
        self.assertEqual(stats['losses'], 4)
        self.assertEqual(stats['win_rate'], 0.4)
        low, high = stats['win_rate_95ci']
        self.assertLess(low, 0.4)
        self.assertGreater(high, 0.4)
        self.assertGreater(E.wilson_interval(0, 10)[1], 0)
        self.assertLess(E.wilson_interval(10, 10)[0], 1)


if __name__ == '__main__':
    unittest.main()
