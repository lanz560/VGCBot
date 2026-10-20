"""Play the Showdown ladder and learn from every battle.

    export PS_USER=MyBotName PS_PASS=secret
    python run_ladder.py --format gen9championsvgc2026regmc --team teams/my_team.txt --games 50

Local test server (no password needed):
    python run_ladder.py --local --format gen9doublescustomgame --team teams/example_team.txt --games 3
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path

from poke_env import (AccountConfiguration, LocalhostServerConfiguration,
                      ShowdownServerConfiguration)

from bot import LearningVGCPlayer, Policy


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="gen9championsvgc2026regmc",
                    help="Showdown format id (lowercase, no spaces or brackets)")
    ap.add_argument("--team", required=True, help="Showdown export-format team file")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-learn", action="store_true", help="freeze weights (evaluation mode)")
    ap.add_argument("--local", action="store_true", help="use a local Showdown server on :8000")
    ap.add_argument("--no-ots", action="store_true",
                    help="don't wait for Open Team Sheets (use if the bot stalls at team preview "
                         "in a format that has no OTS rule)")
    ap.add_argument("--save-replays", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    team = Path(args.team).read_text()
    policy = Policy(Path(args.data_dir) / "weights.json")

    if args.local:
        server, account = LocalhostServerConfiguration, None
    else:
        user, pw = os.environ.get("PS_USER"), os.environ.get("PS_PASS")
        if not user or not pw:
            raise SystemExit("Set PS_USER and PS_PASS (a registered account) or pass --local")
        server, account = ShowdownServerConfiguration, AccountConfiguration(user, pw)

    player = LearningVGCPlayer(
        policy=policy, data_dir=args.data_dir, learn=not args.no_learn,
        prioritize_trick_room="_tr_" in Path(args.team).stem.lower(),
        account_configuration=account, server_configuration=server,
        battle_format=args.format, team=team, max_concurrent_battles=1,
        accept_open_team_sheet=("vgc" in args.format and not args.no_ots), save_replays=str(Path(args.data_dir) / "replays") if args.save_replays else False,
    )
    await player.ladder(args.games)
    print(f"Done. {player.n_won_battles}/{player.n_finished_battles} won this session. "
          f"Weights saved to {policy.path}")


if __name__ == "__main__":
    asyncio.run(main())