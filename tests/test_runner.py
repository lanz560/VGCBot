import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

import runLadderGames as runner
from bot.features import D_MOVE, D_VALUE
from bot.policy import Policy, Step, Trajectory


class RunnerTests(unittest.TestCase):
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
            arguments = ['runLadderGames.py', '--local', '--team', str(team)]
            with patch('sys.argv', arguments), patch.object(runner, 'Policy', return_value=policy), \
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
            arguments = ['runLadderGames.py', '--local', '--no-learn', '--team', str(team)]
            with patch('sys.argv', arguments), patch.object(runner, 'Policy', return_value=policy), \
                    patch.object(runner, 'LearningVGCPlayer', return_value=player) as constructor:
                asyncio.run(runner.main())
            self.assertIs(constructor.call_args.kwargs['learn'], False)
            self.assertFalse(policy.path.exists())
            player.ps_client.stop_listening.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
