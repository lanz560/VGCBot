# VGC Bot

A Pokémon Showdown doubles bot that scores pairs of actions with a linear
softmax policy. Public battles train the policy; local games evaluate frozen
bots against each other.

For the full setup, configuration, troubleshooting, and decision algorithm,
see [notes.md](notes.md), the detailed project guide. That file is currently
excluded from Git.

## Setup

Run commands from the `VGCBot` directory with Python 3.11 or newer:

```bash
python3 -m pip install -r requirements.txt
```

Automatic local server setup also requires Git, npm, and Node.js 22.18.0 or
newer. The first setup needs internet access.

## Play locally

```bash
python3 runLadderGames.py --local --team teams/team1.txt --games 20
```

`--local` downloads and caches Showdown under `data/pokemon-showdown`, starts
it on port 8000, and arranges games between two bots. It stops a server it
started after the run; an existing Showdown server is reused and left running.
**Neither local bot learns.** Results and input snapshots are saved under
`data/eval/local-TIMESTAMP`.

Both bots use the same team source and weights by default. Choose a different
opponent with `--opponent-team` and `--opponent-weights`.

## Use a team folder

Both team options accept a single Showdown export or a folder of `.txt` files:

```bash
python3 runLadderGames.py --local --team teams --opponent-team teams --games 20 --seed 1
```

Each bot independently chooses a random team before every battle. Repeats are
allowed. Only `.txt` files directly inside the folder are included; subfolders
are ignored. A fixed file keeps the same team. Folder selection also works on
the public ladder and in direct challenges.

`--seed` controls policy and team sampling, not Showdown's battle randomness.
The selected filename controls the `_tr_` Trick Room preference. Logs record
the actual team file and its content hash.

## Train on the public ladder

```bash
export PS_USER="YourShowdownUsername"
export PS_PASS="YourShowdownPassword"
python3 runLadderGames.py --team teams --games 50
```

The default format is `gen9championsvgc2026regmc`. Change it with `--format ID`.
Use `--challenge USERNAME` for direct challenges; the opponent must accept.
Credentials come from environment variables, not `.env` files.

Public runs update weights every 16 completed battles by default and flush the
final partial batch during cleanup. Weights default to `data/weights.json`.
Use `--no-learn` to freeze them; move selection remains stochastic.

## Main options

| Option | Purpose |
| --- | --- |
| `--weights PATH` | Select a checkpoint; defaults to `<data-dir>/weights.json`. |
| `--data-dir PATH` | Choose where public logs/weights and default local reports go. |
| `--batch-size N` | Battles per training update; default 16. |
| `--gamma X` | Reward discount; default 0.99. |
| `--shaping-scale X` | HP-potential shaping strength; default 0.25, zero for outcome-only rewards. |
| `--no-lead-synergy` | Disable fixed second-lead pairing bonuses. |
| `--no-ots` | Disable Open Team Sheet waiting. |
| `--save-replays` | Save battle replays. |
| `--showdown-dir PATH` | Use a different cached server checkout in local mode. |
| `--eval-output PATH` | Select a new local report directory. |

For every option and its syntax:

```bash
python3 runLadderGames.py --help
```

## Compare checkpoints

Edit [evaluation.example.json](evaluation.example.json) to select checkpoints,
team files/folders, opponents, seeds, and game counts. Each referenced weights
file must exist. With Showdown already running on port 8000:

```bash
python3 evaluate.py --config evaluation.example.json --output data/eval/run1
```

The evaluator snapshots inputs and reports per-opponent results and win-rate
intervals. It compares checkpoints using the current source code. The detailed
guide explains server startup, checkpoint preparation, and report fields.

## Tests and local data

```bash
python3 -m unittest discover -s tests -v
```

Generated weights, logs, reports, and the downloaded server stay under ignored
`data/` and `battle_logs/` directories. Share selected trained checkpoints
separately. `compare_battles.py` compares existing ladder logs;
`scrape_logs.py` downloads replays for offline analysis, not policy training.
