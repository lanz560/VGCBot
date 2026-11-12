import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bot.teams import TeamPool, apply_team


class TeamPoolTests(unittest.TestCase):
    def test_directory_samples_txt_files_reproducibly_with_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ('a.txt', 'b_tr_.txt', 'c.TXT'):
                (folder / name).write_text(f'{name}\n- Protect\n')
            (folder / 'notes.md').write_text('ignored')
            (folder / 'nested').mkdir()
            (folder / 'nested' / 'ignored.txt').write_text('ignored')
            left, right = TeamPool(folder, seed=4), TeamPool(folder, seed=4)
            picks = [left.choose().path.name for _ in range(30)]
            self.assertEqual(picks, [right.choose().path.name for _ in range(30)])
            self.assertEqual(set(picks), {'a.txt', 'b_tr_.txt', 'c.TXT'})
            self.assertTrue(any(a == b for a, b in zip(picks, picks[1:])))

    def test_file_remains_fixed_and_teams_are_loaded_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'team.txt'
            source.write_text('Pikachu\n- Protect\n')
            pool = TeamPool(source)
            original = pool.choose()
            source.write_text('Changed after loading')
            self.assertEqual(pool.choose(), original)
            self.assertFalse(pool.is_directory)

    def test_team_switch_updates_metadata_and_trick_room_each_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'slow_tr_.txt').write_text('Hatterene\n- Trick Room\n')
            (folder / 'fast.txt').write_text('Pikachu\n- Thunderbolt\n')
            teams = {team.path.name: team for team in TeamPool(folder).teams}
            player = SimpleNamespace(update_team=Mock())
            slow, fast = teams['slow_tr_.txt'], teams['fast.txt']
            apply_team(player, slow)
            self.assertTrue(player.prioritize_trick_room)
            self.assertEqual(player.team_id, slow.team_id)
            self.assertEqual(player.team_file, str(slow.path))
            apply_team(player, fast)
            self.assertFalse(player.prioritize_trick_room)
            player.update_team.assert_called_with(fast.text)
            apply_team(player, slow, prioritize_trick_room=False)
            self.assertFalse(player.prioritize_trick_room)

    def test_empty_or_missing_sources_fail_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'No .txt team files'):
                TeamPool(folder)
            with self.assertRaisesRegex(ValueError, 'not found'):
                TeamPool(folder / 'missing')
            (folder / 'empty.txt').write_text('  \n')
            with self.assertRaisesRegex(ValueError, 'Team file is empty'):
                TeamPool(folder)


if __name__ == '__main__':
    unittest.main()
