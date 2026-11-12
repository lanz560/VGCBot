# VGC BOT

# Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The requirements are `poke-env>=0.16`, `numpy`, and `requests`.

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
| `--team PATH` | Required | Team file in Showdown export format. If the filename (without extension) contains `_tr_`, the bot applies a contextual Trick Room preference while Trick Room is inactive. |
| `--format ID` | `gen9championsvgc2026regmc` | Showdown format ID. |
| `--games N` | `20` | Number of ladder games to play. |
| `--challenge USERNAME` | Off | Challenge this player directly instead of searching the ladder. |
| `--data-dir PATH` | `data` | Directory for policy weights, battle logs, and optional replays. |
| `--weights PATH` | `<data-dir>/weights.json` | Policy weights file. Can be shared between evaluation runs. |
| `--seed N` | Random | Random seed for policy sampling. |
| `--batch-size N` | `16` | Battles collected with fixed weights before each update; remaining battles are flushed on exit. |
| `--gamma X` | `0.99` | Discount applied to future rewards. |
| `--shaping-scale X` | `0.25` | HP potential shaping strength; set to `0` for terminal-outcome-only training. |
| `--no-learn` | Off | Freeze policy weights for evaluation; does not update weights after battles. |
| `--no-lead-synergy` | Off | Disable pairwise lead bonuses for a baseline comparison. |
| `--local` | Off | Connect to a local Showdown server on port 8000; no account credentials are required. |
| `--no-ots` | Off | Do not wait for Open Team Sheets at preview. Use if the bot stalls at preview in a format without OTS. |
| `--save-replays` | Off | Save battle replays under `<data-dir>/replays`. |

For a local run:

```bash
python runLadderGames.py --local --format gen9doublescustomgame --team teams/my_team.txt --games 3
```

## Battle timer

The bot does not automatically enable Showdown's battle timer. To turn it on,
send `/timer` in the battle room chat. The OTS team-preview timeout handling
is separate: it submits the selected team when the preview warning arrives,
but does not start the battle timer.

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

## Learning and checkpoint evaluation

The linear softmax policy collects 16 completed battles under fixed parameters
before each gradient update. A final partial batch is trained when the runner
exits normally or unwinds through its cleanup handler. Pending trajectories are
held in memory: an abrupt process kill loses that unfinished batch. Saved
checkpoints contain the parameters and counters through the last update.

The policy now estimates speed matchups gained/lost from Tailwind and Trick
Room, extra damage/KOs enabled by Helping Hand, known single-target threat
that redirection could absorb, and the safety/usefulness of setup. These are
heuristics, not a battle simulator. The value baseline uses 12 state features.
Trick Room's filename-triggered preference scales with speed matchups improved;
it no longer gives a blanket bonus for setting a harmful Trick Room.

Rewards combine the terminal outcome with discounted potential shaping:
`outcome + scale * (gamma * next_potential - current_potential)`, with the
outcome added only on the last decision and terminal potential fixed to zero.
Compare shaping strengths using the same evaluation pool. For example:

```bash
python runLadderGames.py --team teams/team2_tr_.txt --games 64 --batch-size 16 --gamma 0.99 --shaping-scale 0.25
```

Old 30-feature checkpoints are migrated by feature name, preserving existing
weights and initializing six new features from their priors. Old five-value
baselines are extended with zeros. Migration happens in memory; training saves
the new schema. Frozen evaluation does not rewrite inputs. Save a baseline copy
before training if you want to compare checkpoints.

`evaluate.py` compares frozen policies against an explicit pool on a local
Showdown server on port 8000. The server must support the chosen format and team
species. Create `data/checkpoints/baseline.json`, `candidate.json`, and
`opponent.json` from the checkpoints you want to measure, then edit
[evaluation.example.json](evaluation.example.json) to select teams, opponents,
seeds, and games. Paths in that file are relative to the config's directory.

```bash
python evaluate.py --config evaluation.example.json --output data/eval/run1
```

Use a new output directory for each run. The harness copies the exact input
files, records their hashes and source-code hashes, alternates variant order
and challenger assignment across blocks, and writes progress to `report.json`.
It reports per-seed/per-opponent blocks and aggregate win fractions with 95%
Wilson intervals; ties count as non-wins. These intervals are descriptive and
do not account for correlation between games against the same opponents.
Seeds control bot sampling, not the server's damage rolls or other battle RNG.
Both variants run the current code: this compares checkpoints, not historical
versions of the bot. Inspect per-opponent results as well as the overall rate.

Batch diagnostics in battle summaries include gradient/update norms, mean
policy entropy, mean old-to-new KL divergence, and baseline MSE. The final
partial batch's diagnostics are printed by the runner. Evaluation keeps
sampling stochastically while disabling learning.

Run the test suite with:

```bash
python3 -m unittest discover -s tests -v
```

## Replay-training scaffold

`scrape_logs.py` implements replay downloading and filtering. The replay
parser, imitation learner, replay-training runner, and self-play runner
(`bot/replay_data.py`, `bot/imitation.py`, `train_from_replays.py`, and
`selfplay_local.py`) are empty. Scraped logs are not consumed by the policy.

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
