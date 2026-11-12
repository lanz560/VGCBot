# VGC Bot

A Pokémon Showdown doubles bot with a linear softmax policy and batched online
learning. It scores joint actions for both active Pokémon using estimated
damage, speed, support, and switch matchups.

## Setup

Use Python 3.11 or newer:

```bash
python3 -m pip install -r requirements.txt
```

Local server setup also requires Git, npm, and Node.js 22.18.0 or newer.

## Local games

```bash
python3 runLadderGames.py --local --team teams/team1.txt --games 20
```

This installs Showdown under `data/pokemon-showdown` on the first run, starts
it on port 8000, and runs two bots. Later runs reuse the installation. A server
started by this command stops when the run ends; an already-running Showdown
server is reused and left running. Managed servers bind to `127.0.0.1` and use
`--no-security` for password-free bot connections.

Both bots use the same team and weights by default. To choose another opponent:

```bash
python3 runLadderGames.py --local --team teams/team1.txt --opponent-team teams/team2_tr_.txt --opponent-weights data/checkpoints/opponent.json --games 20 --seed 1
```

Local games are **evaluation only**: neither bot learns. Missing default weights
use handwritten priors; explicit weight paths must exist. Reports, input
snapshots, and battle logs go under `data/eval/local-TIMESTAMP`. Use
`--eval-output PATH` to choose a new report directory, or `--showdown-dir PATH`
for a different server checkout. First-time setup needs internet access.
Setup and server logs are saved beside the checkout.

## Public ladder and challenges

```bash
export PS_USER="YourShowdownUsername"
export PS_PASS="YourShowdownPassword"
python3 runLadderGames.py --team teams/team1.txt --games 50
```

The default format is `gen9championsvgc2026regmc`. Use `--format ID` to change it
or `--challenge USERNAME` to send direct challenges instead of searching the
ladder. The opponent must accept. `--local` and `--challenge` are incompatible.

Public runs learn by default. Weights are saved to `data/weights.json` after
each batch; `--weights PATH` selects another checkpoint. `--data-dir PATH`
changes the default weights/log directory. Keep credentials in environment
variables; the runner does not load `.env` files.

## Learning controls

| Flag | Default | Effect |
| --- | --- | --- |
| `--batch-size N` | `16` | Completed battles per update. The final partial batch is flushed during cleanup. |
| `--gamma X` | `0.99` | Discount on future rewards. |
| `--shaping-scale X` | `0.25` | Discounted HP-potential shaping; zero uses terminal outcomes only. |
| `--no-learn` | Off | Freeze policy parameters. Actions remain stochastic. |
| `--seed N` | Random | Seed policy sampling, including team preview. |
| `--no-lead-synergy` | Off | Disable fixed second-lead synergy bonuses. |
| `--no-ots` | Off | Disable Open Team Sheet waiting. |
| `--save-replays` | Off | Save battle replays. |

Weights stay fixed while a batch is collected. Pending trajectories live in
memory and are lost on an abrupt process kill. Older checkpoints are migrated
by feature name. A team filename containing `_tr_` enables a contextual Trick
Room preference. Logged team roles are informational.

For all options:

```bash
python3 runLadderGames.py --help
```

## Compare checkpoints

Copy the checkpoints you want to compare, then edit
[evaluation.example.json](evaluation.example.json) to select variants, opponents,
teams, seeds, and game counts. Paths resolve relative to that config file.
Direct evaluator use requires a Showdown server already running on port 8000:

```bash
python3 evaluate.py --config evaluation.example.json --output data/eval/run1
```

The evaluator freezes inputs and reports per-opponent results and aggregate win
fractions with 95% Wilson intervals. Ties count as non-wins; the intervals do
not account for dependence between games. Seeds control bot sampling, not
Showdown's battle RNG. Both variants use the current source code.

`compare_battles.py` also compares existing ladder logs from the same team and
format. `scrape_logs.py` downloads replay logs separately; the policy does not
train from them.

## Tests and generated files

```bash
python3 -m unittest discover -s tests -v
```

`data/` and `battle_logs/` contain local outputs and are excluded from Git.
Share selected trained checkpoints separately. The detailed decision notes
remain in the locally ignored `notes.md`.
