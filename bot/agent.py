from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.player import DefaultBattleOrder, DoubleBattleOrder, Player

from . import features as F
from .policy import Policy, Step, Trajectory

log = logging.getLogger("vgc_bot")


class LearningVGCPlayer(Player):
    """Plays VGC doubles with a softmax policy and takes a gradient step after every battle."""

    def __init__(
        self,
        policy: Policy,
        data_dir: str | Path = "data",
        learn: bool = True,
        prioritize_trick_room: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy
        self.learning_enabled = learn
        self.prioritize_trick_room = prioritize_trick_room
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._traj: dict[str, Trajectory] = {}

    # ------------------------------------------------------------ team preview
    def teampreview(self, battle) -> str:
        try:
            rows = F.teampreview_features(battle)
            k = battle.max_team_size or len(battle.team)
            picks = self.policy.pick_team(rows, k)
            mons = list(battle.team.values())
            for i in picks:
                mons[i]._selected_in_teampreview = True
            traj = self._traj.setdefault(battle.battle_tag, Trajectory())
            traj.tp_rows, traj.tp_picks = rows, picks
            return "/team " + "".join(str(i + 1) for i in picks)
        except Exception:
            log.exception("teampreview failed, falling back to random")
            return self.random_teampreview(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        if split_messages and split_messages[0]:
            room_header = split_messages[0][0]
            if room_header.startswith(">battle-"):
                result = next(
                    (
                        message for message in split_messages[1:]
                        if len(message) > 1 and message[1] in ("win", "tie")
                    ),
                    None,
                )
                if result is not None:
                    if result[1] == "tie":
                        outcome = "tied"
                    else:
                        outcome = (
                            "Another victory under my belt." if result[2].casefold() == self.username.casefold()
                            else " My training continues..."
                        )
                    try:
                        await self.ps_client.send_message(
                            f"{outcome}", room_header[1:]
                        )
                    except Exception:
                        log.exception("failed to send battle result to chat")

        await super()._handle_battle_message(split_messages)

    # ------------------------------------------------------------------ moves
    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        try:
            if battle.wait:
                return DoubleBattleOrder(DefaultBattleOrder(), DefaultBattleOrder())
            orders, phi = F.build_candidates(battle)
            if not orders:
                return self.choose_random_move(battle)
            trick_room_counts = [
                sum(getattr(single_order.order, "id", None) == "trickroom"
                    for single_order in (order.first_order, order.second_order))
                for order in orders
            ]
            bias = [0.0] * len(orders)
            if self.prioritize_trick_room and Field.TRICK_ROOM not in battle.fields:
                eligible = [i for i, count in enumerate(trick_room_counts) if count <= 1]
                if eligible:
                    orders = [orders[i] for i in eligible]
                    phi = phi[eligible]
                    trick_room_counts = [trick_room_counts[i] for i in eligible]
                    bias = [0.0] * len(orders)
                for i, count in enumerate(trick_room_counts):
                    if count == 1:
                        bias[i] = 3.0
            bias_array = np.asarray(bias)
            idx = self.policy.act(phi, bias_array)
            summ = F.state_summary(battle)
            self._traj.setdefault(battle.battle_tag, Trajectory()).steps.append(
                Step(phi=phi, chosen=idx, psi=summ["psi"], pot=summ["pot"],
                     bias=bias_array)
            )
            return orders[idx]
        except Exception:
            log.exception("choose_move failed, falling back to random")
            return self.choose_random_move(battle)

    # ---------------------------------------------------------------- learning
    def _battle_finished_callback(self, battle) -> None:
        traj = self._traj.pop(battle.battle_tag, Trajectory())
        try:
            traj.final_pot = F.state_summary(battle)["pot"]
        except Exception:
            traj.final_pot = traj.steps[-1].pot if traj.steps else 0.0
        won = battle.won
        stats = {}
        if self.learning_enabled:
            try:
                stats = self.policy.learn(traj, won)
            except Exception:
                log.exception("learning step failed")
        self._log_battle(battle, won, stats)

    def _log_battle(self, battle, won, stats) -> None:
        rec = {
            "t": time.time(), "tag": battle.battle_tag, "format": battle.format,
            "won": won, "turns": battle.turn,
            "my_team": [m.species for m in battle.team.values()],
            "opp_team": [m.species for m in battle.opponent_team.values()],
            "learn": stats,
            "running_winrate": round(self.policy.wins / max(1, self.policy.battles_seen), 3),
            "battles_seen": self.policy.battles_seen,
        }
        with open(self.data_dir / "battles.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("%s %s in %d turns | total %d battles, %.0f%% wins", battle.battle_tag,
                 "WON" if won else "LOST" if won is False else "TIED", battle.turn,
                 self.policy.battles_seen, 100 * rec["running_winrate"])