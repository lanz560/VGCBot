"""Compare win rates from baseline and lead-synergy battle logs."""
import argparse
import json
from pathlib import Path
from typing import Any


def load_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Battle log not found: {path}")
    with path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise SystemExit(f"Battle log is empty: {path}")
    return records


def experiment_signature(records: list[dict[str, Any]]) -> tuple[str, str]:
    if any(not record.get("team_id") for record in records):
        raise SystemExit("Battle logs must include team_id; rerun both variants with this version.")
    signatures = {
        (record.get("format", ""), record["team_id"])
        for record in records
    }
    if len(signatures) != 1:
        raise SystemExit("Each log must contain battles from one format and one exact team file.")
    return next(iter(signatures))


def block_summary(records: list[dict[str, Any]], start: int, end: int) -> tuple[int, int, int]:
    block = records[start:end]
    wins = sum(record.get("won") is True for record in block)
    ties = sum(record.get("won") is None for record in block)
    return wins, len(block), ties


def format_rate(summary: tuple[int, int, int]) -> str:
    wins, games, ties = summary
    rate = 100 * wins / games if games else 0.0
    return f"{rate:5.1f}% ({wins}/{games}, {ties} ties)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("synergy_log", type=Path)
    parser.add_argument("--block-size", type=int, default=50)
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("--block-size must be at least 1")

    baseline = load_log(args.baseline_log)
    synergy = load_log(args.synergy_log)
    if experiment_signature(baseline) != experiment_signature(synergy):
        raise SystemExit("Baseline and synergy logs must use the same format and team.")

    print(f"Comparing {len(baseline)} baseline games and {len(synergy)} synergy games")
    print(f"Blocks contain up to {args.block_size} consecutive games per variant.")
    print("Block | Baseline win rate             | Synergy win rate")
    block_count = max(
        (len(baseline) + args.block_size - 1) // args.block_size,
        (len(synergy) + args.block_size - 1) // args.block_size,
    )
    for block_index in range(block_count):
        start = block_index * args.block_size
        end = start + args.block_size
        base = block_summary(baseline, start, end)
        enhanced = block_summary(synergy, start, end)
        print(f"{block_index + 1:5} | {format_rate(base):29} | {format_rate(enhanced)}")

    print(f"Overall | {format_rate(block_summary(baseline, 0, len(baseline))):29} | "
          f"{format_rate(block_summary(synergy, 0, len(synergy)))}")


if __name__ == "__main__":
    main()
