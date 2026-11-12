"""Run public battles with learning, or manage local bot evaluations."""
import argparse
import asyncio
import hashlib
import logging
import math
import os
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import evaluate as evaluation
from bot import LearningVGCPlayer, Policy
from local_server import DEFAULT_DIRECTORY, local_showdown
from poke_env import AccountConfiguration, ShowdownServerConfiguration

logger = logging.getLogger(__name__)


async def evaluate_local(args) -> None:
    """Evaluate a frozen policy against a second bot, using existing CLI inputs."""
    team = Path(args.team).resolve()
    opponent_team = Path(args.opponent_team).resolve() if args.opponent_team else team
    weights = (Path(args.weights) if args.weights else Path(args.data_dir) / "weights.json").resolve()
    opponent_weights = Path(args.opponent_weights).resolve() if args.opponent_weights else weights
    for path in (team, opponent_team):
        if not path.is_file():
            raise ValueError(f"Team file not found: {path}")
    # Explicit checkpoint typos must not silently turn into untrained opponents.
    for supplied in (args.weights, args.opponent_weights):
        if supplied and not Path(supplied).is_file():
            raise ValueError(f"Weights file not found: {supplied}")
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    output = (Path(args.eval_output) if args.eval_output else
              Path(args.data_dir) / "eval" / datetime.now(UTC).strftime("local-%Y%m%d-%H%M%S-%f"))
    with tempfile.TemporaryDirectory(prefix="vgc-local-") as temporary:
        if not weights.exists():
            # Let a first local run work without creating/modifying training weights.
            weights = Path(temporary) / "priors.json"
            Policy(weights).save()
            if not args.opponent_weights:
                opponent_weights = weights
        config = {
            "format": args.format, "games_per_matchup": args.games, "seeds": [seed],
            "accept_open_team_sheet": "vgc" in args.format and not args.no_ots,
            "save_replays": args.save_replays,
            "variants": [{"name": "candidate", "team": str(team), "weights": str(weights),
                          "lead_synergy": not args.no_lead_synergy}],
            "opponents": [{"name": "opponent", "team": str(opponent_team),
                           "weights": str(opponent_weights),
                           "lead_synergy": not args.no_lead_synergy}],
        }
        logger.info("Local evaluation: two frozen bots, %d games, seed %d", args.games, seed)
        if output.exists():
            raise ValueError(f"Evaluation output directory already exists: {output}")
        # Check policy schemas before downloading or starting the server.
        Policy(weights)
        Policy(opponent_weights)
        async with local_showdown(args.showdown_dir or DEFAULT_DIRECTORY):
            report = await evaluation.evaluate(config, output.resolve())
    evaluation.print_summary(report)
    print(f"Report: {output.resolve() / 'report.json'}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="gen9championsvgc2026regmc",
                    help="Showdown format id (lowercase, no spaces or brackets)")
    ap.add_argument("--team", required=True, help="Showdown export-format team file")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--challenge", metavar="USERNAME",
                    help="challenge this player directly instead of searching the ladder")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--weights", help="policy weights file (defaults to <data-dir>/weights.json)")
    ap.add_argument("--seed", type=int, help="random seed for reproducible policy sampling")
    ap.add_argument("--no-learn", action="store_true", help="freeze weights (evaluation mode)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="completed battles per update; final partial batch is flushed")
    ap.add_argument("--gamma", type=float, default=0.99, help="return discount in [0, 1]")
    ap.add_argument("--shaping-scale", type=float, default=0.25,
                    help="HP potential shaping strength; 0 trains from outcomes only")
    ap.add_argument("--no-lead-synergy", action="store_true",
                    help="disable pairwise lead synergy preferences for an A/B comparison")
    ap.add_argument("--local", action="store_true",
                    help="set up/start Showdown, then run two frozen bots on localhost:8000")
    ap.add_argument("--showdown-dir", help=f"Showdown checkout/cache (default: {DEFAULT_DIRECTORY})")
    ap.add_argument("--opponent-team", help="local opponent team (defaults to --team)")
    ap.add_argument("--opponent-weights", help="local opponent checkpoint (defaults to our weights)")
    ap.add_argument("--eval-output", help="new local report directory (defaults to data-dir/eval/local-TIMESTAMP)")
    ap.add_argument("--no-ots", action="store_true",
                    help="don't wait for Open Team Sheets (use if the bot stalls at team preview "
                         "in a format that has no OTS rule)")
    ap.add_argument("--save-replays", action="store_true")
    args = ap.parse_args()
    if args.games < 1 or args.batch_size < 1:
        ap.error("--games and --batch-size must be positive")
    if not 0 <= args.gamma <= 1 or not math.isfinite(args.shaping_scale) or args.shaping_scale < 0:
        ap.error("--gamma must be in [0, 1] and --shaping-scale nonnegative")
    if args.seed is not None and args.seed < 0:
        ap.error("--seed must be nonnegative")
    if args.local and args.challenge:
        ap.error("--local arranges two-bot games and cannot be combined with --challenge")
    if not args.local and any((args.opponent_team, args.opponent_weights, args.eval_output, args.showdown_dir)):
        ap.error("--opponent-team, --opponent-weights, --eval-output and --showdown-dir require --local")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.local:
        try:
            await evaluate_local(args)
        except (ValueError, OSError, RuntimeError) as exc:
            ap.error(str(exc))
        return
    team_path = Path(args.team)
    team = team_path.read_text()
    team_id = hashlib.sha256(team.encode()).hexdigest()
    weights_path = Path(args.weights) if args.weights else Path(args.data_dir) / "weights.json"
    policy = Policy(weights_path, seed=args.seed, batch_size=args.batch_size,
                    gamma=args.gamma, shaping_scale=args.shaping_scale)

    user, pw = os.environ.get("PS_USER"), os.environ.get("PS_PASS")
    if not user or not pw:
        raise SystemExit("Set PS_USER and PS_PASS (a registered account) or pass --local")
    server, account = ShowdownServerConfiguration, AccountConfiguration(user, pw)

    player = LearningVGCPlayer(
        log_level=logging.WARNING,
        policy=policy, data_dir=args.data_dir, learn=not args.no_learn,
        prioritize_trick_room="_tr_" in Path(args.team).stem.lower(),
        lead_synergy=not args.no_lead_synergy,
        team_id=team_id,
        account_configuration=account, server_configuration=server,
        battle_format=args.format, team=team, max_concurrent_battles=1,
        accept_open_team_sheet=("vgc" in args.format and not args.no_ots), save_replays=str(Path(args.data_dir) / "replays") if args.save_replays else False,
    )
    try:
        if args.challenge is not None:
            await player.send_challenges(args.challenge, args.games)
        else:
            await player.ladder(args.games)
    finally:
        try:
            if not args.no_learn:
                stats = policy.flush()
                if stats:
                    logger.info("Final batch: %s", stats)
        finally:
            await player.ps_client.stop_listening()
    print(f"Done. {player.n_won_battles}/{player.n_finished_battles} won this session. "
          + (f"Weights saved to {policy.path}" if not args.no_learn else "Weights frozen."))


if __name__ == "__main__":
    asyncio.run(main())
