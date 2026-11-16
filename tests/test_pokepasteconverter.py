import io
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "convert_paste.py"
URL = "https://pokepast.es/example"
HTML = ('<article><pre><span>Salamence</span> @ Salamencite\n'
        '<span>Ability: </span>Intimidate\n- Protect\n</pre></article>')
TEAM = "Salamence @ Salamencite\nAbility: Intimidate\n- Protect\n"


class ConverterTests(unittest.TestCase):
    def run_converter(self, folder, html=HTML):
        response = SimpleNamespace(text=html, raise_for_status=Mock())
        with patch("sys.argv", [str(SCRIPT), URL, str(folder)]), \
                patch("requests.get", return_value=response) as get, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        get.assert_called_once_with(URL, timeout=10)
        response.raise_for_status.assert_called_once()
        return output.getvalue()

    def test_cli_creates_output_folder_and_saves_team(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "my teams"
            output = self.run_converter(folder)
            self.assertEqual((folder / "team.txt").read_text(), TEAM)
            self.assertIn(str(folder / "team.txt"), output)

    def test_existing_team_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "team.txt").write_text("first team")
            (folder / "team1.txt").write_text("second team")
            self.run_converter(folder)
            self.assertEqual((folder / "team.txt").read_text(), "first team")
            self.assertEqual((folder / "team1.txt").read_text(), "second team")
            self.assertEqual((folder / "team2.txt").read_text(), TEAM)

    def test_page_without_team_does_not_create_empty_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "teams"
            with self.assertRaisesRegex(ValueError, "No team found"):
                self.run_converter(folder, html="<html>Not a paste</html>")
            self.assertFalse(folder.exists())

    def test_missing_arguments_show_usage_without_fetching(self):
        with patch("sys.argv", [str(SCRIPT)]), patch("sys.stderr", new_callable=io.StringIO), \
                patch("requests.get") as get:
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertEqual(error.exception.code, 2)
        get.assert_not_called()
