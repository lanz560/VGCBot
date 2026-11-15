"""Save a Poképaste team export to a numbered text file in a folder."""
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def access_paste_contents(link: str) -> str:
    """Extract the team while preserving line breaks."""
    response = requests.get(link, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    team_text = "\n\n".join(
        pre.get_text().strip()
        for pre in soup.select("article pre")
    )

    if not team_text:
        raise ValueError("No team found in the Poképaste page")
    return team_text


def main(link: str, path: str) -> None:
    paste = access_paste_contents(link)
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / "team.txt"
    counter = 1
    while filename.exists():
        filename = directory / f"team{counter}.txt"
        counter += 1
    filename.write_text(paste + "\n", encoding="utf-8")
    print(f"Saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("link", help="Poképaste URL to convert")
    parser.add_argument("path", help="Output folder for team files")
    args = parser.parse_args()
    main(args.link, args.path)
