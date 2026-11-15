import io
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "pokepasteconverter.py"


class ConverterTests(unittest.TestCase):
    def test_cli_passes_url_and_prints_team_text(self):
        url = "https://pokepast.es/example"
        response = SimpleNamespace(
            text='<article><pre><span>Salamence</span> @ Salamencite\n'
                 '<span>Ability: </span>Intimidate\n- Protect\n</pre></article>',
            raise_for_status=Mock(),
        )
        with patch("sys.argv", [str(SCRIPT), url]), \
                patch("requests.get", return_value=response) as get, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        get.assert_called_once_with(url, timeout=10)
        response.raise_for_status.assert_called_once()
        self.assertEqual(output.getvalue(),
                         "Salamence @ Salamencite\nAbility: Intimidate\n- Protect\n")

    def test_missing_url_shows_usage_without_fetching(self):
        with patch("sys.argv", [str(SCRIPT)]), patch("sys.stderr", new_callable=io.StringIO), \
                patch("requests.get") as get:
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertEqual(error.exception.code, 2)
        get.assert_not_called()
