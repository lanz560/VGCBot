

# Requirements
poke-env>=0.16
numpy
 

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
