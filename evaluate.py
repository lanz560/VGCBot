"""Evaluate frozen checkpoints against a fixed pool on a local Showdown server.

Usage: python evaluate.py --config evaluation.example.json --output data/eval/run1
Paths inside the JSON are relative to the config file. Server battle randomness
is not seeded by this script; seeds control the bots' action sampling only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from bot import LearningVGCPlayer, Policy
from poke_env import LocalhostServerConfiguration


def wilson_interval(wins: int, games: int) -> list[float]:
    """95% Wilson interval for the win fraction, treating ties as non-wins."""
    if not games:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records: list[dict]) -> dict:
    games = sum(r["games"] for r in records)
    wins = sum(r["wins"] for r in records)
    ties = sum(r["ties"] for r in records)
    return {"games": games, "wins": wins, "losses": games - wins - ties,
            "ties": ties, "win_rate": wins / games if games else 0.0,
            "win_rate_95ci": wilson_interval(wins, games)}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if not isinstance(config.get("format"), str) or not config["format"]:
        raise ValueError("config needs a format")
    count = config.get("games_per_matchup", 20)
    seeds = config.get("seeds", [1, 2, 3])
    if type(count) is not int or count < 1:
        raise ValueError("games_per_matchup must be a positive integer")
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    config["games_per_matchup"], config["seeds"] = count, seeds
    for flag in ("accept_open_team_sheet", "save_replays"):
        if type(config.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    for group in ("variants", "opponents"):
        entries = config.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"config needs nonempty {group}")
        names = set()
        for entry in entries:
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"{group} names must be nonempty and unique")
            names.add(name)
            for key in ("team", "weights"):
                source = (path.parent / entry[key]).resolve()
                if not source.is_file():
                    raise ValueError(f"Missing {key}: {source}")
                entry[key] = str(source)
            for key in ("lead_synergy", "prioritize_trick_room"):
                if key in entry and type(entry[key]) is not bool:
                    raise ValueError(f"{key} must be a boolean")
            # Validate checkpoint schemas before making any server connections.
            Policy(entry["weights"])
    return config


def snapshot_inputs(config: dict, output: Path) -> dict:
    """Freeze exact file contents for every block, including against concurrent training."""
    frozen = json.loads(json.dumps(config))
    inputs = output / "inputs"
    inputs.mkdir()
    for group in ("variants", "opponents"):
        for index, entry in enumerate(frozen[group]):
            entry.setdefault("prioritize_trick_room", "_tr_" in Path(entry["team"]).stem.lower())
            entry.setdefault("lead_synergy", True)
            entry["sources"], entry["sha256"] = {}, {}
            for key in ("team", "weights"):
                original = Path(entry[key])
                data = original.read_bytes()
                target = inputs / f"{group}-{index}-{key}{original.suffix}"
                target.write_bytes(data)
                entry["sources"][key] = str(original)
                entry["sha256"][key] = hashlib.sha256(data).hexdigest()
                entry[key] = str(target.resolve())
    return frozen


async def run_block(config: dict, variant: dict, opponent: dict, seed: int,
                    directory: Path, reverse: bool) -> dict:
    players = []
    try:
        for entry, offset, label in ((variant, 0, "variant"), (opponent, 1, "opponent")):
            policy = Policy(entry["weights"], seed=2 * seed + offset)
            team = Path(entry["team"]).read_text()
            players.append(LearningVGCPlayer(
                log_level=logging.WARNING,
                policy=policy, learn=False, data_dir=directory / label,
                team_id=entry["sha256"]["team"],
                lead_synergy=entry["lead_synergy"],
                prioritize_trick_room=entry["prioritize_trick_room"],
                team=team, battle_format=config["format"],
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                accept_open_team_sheet=config.get("accept_open_team_sheet", False),
                save_replays=str(directory / label / "replays") if config.get("save_replays") else False,
            ))
        candidate, adversary = players
        first, second = (adversary, candidate) if reverse else (candidate, adversary)
        await first.battle_against(second, n_battles=config["games_per_matchup"])
        games = candidate.n_finished_battles
        if games != config["games_per_matchup"]:
            raise RuntimeError(f"Expected {config['games_per_matchup']} battles, finished {games}")
        return {"variant": variant["name"], "opponent": opponent["name"], "seed": seed,
                "games": games, "wins": candidate.n_won_battles,
                "ties": sum(b.won is None for b in candidate.battles.values()),
                "challenger": "opponent" if reverse else "variant"}
    finally:
        await asyncio.gather(*(p.ps_client.stop_listening() for p in players))


async def evaluate(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    frozen = snapshot_inputs(config, output)
    report = {"started_at": datetime.now(UTC).isoformat(),
              "python": platform.python_version(), "poke_env": version("poke-env"),
              "source_sha256": {
                  name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                  for name in ("evaluate.py", "bot/features.py", "bot/policy.py", "bot/agent.py")
              },
              "config": frozen, "blocks": [], "complete": False,
              "seed_scope": "Policy sampling only; Showdown battle RNG is not controlled."}

    def save():
        report["summary"] = {
            entry["name"]: summarize([r for r in report["blocks"] if r["variant"] == entry["name"]])
            for entry in frozen["variants"]
        }
        temp = output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output / "report.json")

    save()
    try:
        for seed_index, seed in enumerate(frozen["seeds"]):
            for opponent_index, opponent in enumerate(frozen["opponents"]):
                # Alternate variant scheduling to reduce time/order confounds.
                variants = frozen["variants"][::(-1 if seed_index % 2 else 1)]
                for variant in variants:
                    block = len(report["blocks"])
                    result = await run_block(
                        frozen, variant, opponent, seed, output / f"block-{block:04d}",
                        reverse=bool((seed_index + opponent_index) % 2),
                    )
                    report["blocks"].append(result)
                    save()
                    print(f"{variant['name']} vs {opponent['name']} seed={seed}: "
                          f"{result['wins']}/{result['games']} wins", flush=True)
        report["complete"] = True
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()


def print_summary(report: dict) -> None:
    for name, stats in report["summary"].items():
        low, high = stats["win_rate_95ci"]
        print(f"{name}: {stats['win_rate']:.1%} wins ({stats['wins']}/{stats['games']}), "
              f"95% Wilson interval {low:.1%}-{high:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for frozen inputs, logs, and report.json")
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    report = asyncio.run(evaluate(config, args.output.resolve()))
    print_summary(report)


if __name__ == "__main__":
    main()
