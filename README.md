# VGC BOT

# Requirements

poke-env>=0.16
numpy

# Running the bot

From the `VGCBot` directory, install the requirements and start a ladder run:

```bash
python -m pip install -r requirements.txt
export PS_USER="YourShowdownUsername"
export PS_PASS="YourShowdownPassword"
python runLadderGames.py --team teams/my_team.txt --games 50
```

The bot uses `gen9championsvgc2026regmc` by default. To choose another format:

```bash
python runLadderGames.py --format gen9doublescustomgame --team teams/my_team.txt --games 3
```

To challenge another player directly, provide their Showdown username with
`--challenge`. The other player must accept the challenge. `--games` controls
how many challenges to send:

```bash
python runLadderGames.py --format gen9doublescustomgame --team teams/my_team.txt --challenge Opponent --games 1
```

## Command-line options

| Flag | Default | Description |
| --- | --- | --- |
| `--team PATH` | Required | Team file in Showdown export format. If the filename (without extension) contains `_tr_`, the bot prioritizes using Trick Room once while Trick Room is inactive. |
| `--format ID` | `gen9championsvgc2026regmc` | Showdown format ID. |
| `--games N` | `20` | Number of ladder games to play. |
| `--challenge USERNAME` | Off | Challenge this player directly instead of searching the ladder. |
| `--data-dir PATH` | `data` | Directory for policy weights, battle logs, and optional replays. |
| `--weights PATH` | `<data-dir>/weights.json` | Policy weights file. Can be shared between evaluation runs. |
| `--seed N` | Random | Random seed for policy sampling. |
| `--no-learn` | Off | Freeze policy weights for evaluation; does not update weights after battles. |
| `--no-lead-synergy` | Off | Disable pairwise lead bonuses for a baseline comparison. |
| `--local` | Off | Connect to a local Showdown server on port 8000; no account credentials are required. |
| `--no-ots` | Off | Do not wait for Open Team Sheets at preview. Use if the bot stalls at preview in a format without OTS. |
| `--save-replays` | Off | Save battle replays under `<data-dir>/replays`. |

For a local run:

```bash
python runLadderGames.py --local --format gen9doublescustomgame --team teams/my_team.txt --games 3
```

## Lead-synergy A/B comparison

The default lead picker adds small, fixed bonuses to the second lead for these
pair features: support plus attacker, a Trick Room setter plus a slow attacker,
and Psychic Surge plus a teammate with Expanding Force. On `team2_tr_`, this
can favor Indeedee-F + Hatterene. These bonuses do not modify the learned
weights.

For a fair comparison, use the same team, format, frozen weights, and game
count for both variants. Keep separate data directories for their logs, and
alternate which variant you run first between blocks:

```bash
python runLadderGames.py --team teams/team2_tr_.txt --games 50 --no-learn --weights data/weights.json --seed 1 --data-dir data/ab/baseline --no-lead-synergy
python runLadderGames.py --team teams/team2_tr_.txt --games 50 --no-learn --weights data/weights.json --seed 1 --data-dir data/ab/synergy
```

Repeat both commands for additional blocks, alternating which one runs first.
Each directory appends to its own `battles.jsonl`. Compare block and overall
win rates with:

```bash
python compare_battles.py data/ab/baseline/battles.jsonl data/ab/synergy/battles.jsonl --block-size 50
```

The comparison checks that both logs use the same format and team. Win rates
from a small number of games are noisy; gather multiple blocks before drawing
conclusions.

# Team roles

At team preview the bot logs an informational role for each Pokémon, inferred
from its currently known moves:

- `offensive`: attacks without a detected setup or substantial support role.
- `support`: at most two attacks and at least two recognized support moves.
- `setup`: has a recognized stat-boosting setup move.
- `mix`: has at least two attacks and at least one recognized support move.

These are heuristic labels, not manually assigned species identities. They
are written to the console and to `my_team_roles` in each `battles.jsonl`
record. Roles do not currently affect team preview or battle decisions.
