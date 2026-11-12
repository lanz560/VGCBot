"""Load team exports and sample directory pools independently of move sampling."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path


def team_files(source: str | Path) -> list[Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"Team file or directory not found: {source}")
    files = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError(f"No .txt team files in directory: {source}")
    return files


@dataclass(frozen=True)
class Team:
    path: Path
    text: str

    @property
    def team_id(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def trick_room(self) -> bool:
        return "_tr_" in self.path.stem.lower()


class TeamPool:
    def __init__(self, source: str | Path, seed: int | None = None):
        source = Path(source).expanduser().resolve()
        self.is_directory = source.is_dir()
        self.teams = [Team(path, path.read_text()) for path in team_files(source)]
        for team in self.teams:
            if not team.text.strip():
                raise ValueError(f"Team file is empty: {team.path}")
        self.rng = random.Random(seed)

    def choose(self) -> Team:
        return self.rng.choice(self.teams) if self.is_directory else self.teams[0]


def apply_team(player, team: Team, prioritize_trick_room: bool | None = None) -> None:
    """Change teams only between completed battles (players run sequentially)."""
    player.update_team(team.text)
    player.team_id = team.team_id
    player.team_file = str(team.path)
    player.prioritize_trick_room = team.trick_room if prioritize_trick_room is None else prioritize_trick_room
