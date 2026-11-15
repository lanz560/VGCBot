"""Print the team export from a Poképaste URL."""
import argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

def access_paste_contents(link: str) -> str:
    response = requests.get(link, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    team_text = "\n\n".join(
        pre.get_text().strip()
        for pre in soup.select("article pre")
    )

    return team_text


def main(link: str):
    
    paste = access_paste_contents(link)
    base_name = "team"
    ext = ".txt"
    filename = f"{base_name}{ext}"
    counter = 1


    while Path(filename).exists():
        filename = f"{base_name}{counter}{ext}"
        counter += 1

        try:
            print(paste, file=f)
        except: 





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("link", help="Poképaste URL to convert")
    main(parser.parse_args().link)
