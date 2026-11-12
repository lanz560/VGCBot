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
    """Plays VGC doubles and queues completed battles for on-policy updates."""

    def __init__(
        self,
        policy: Policy,
        data_dir: str | Path = "data",
        learn: bool = True,
        prioritize_trick_room: bool = False,
        lead_synergy: bool = True,
        team_id: str | None = None,
        team_file: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy
        self.learning_enabled = learn
        self.prioritize_trick_room = prioritize_trick_room
        self.lead_synergy = lead_synergy
        self.team_id = team_id
        self.team_file = team_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._traj: dict[str, Trajectory] = {}
        self._lead_picks: dict[str, list[str]] = {}

    def teampreview(self, battle) -> str:
        try:
            rows = F.teampreview_features(battle)
            mons = list(battle.team.values())
            k = min(battle.max_team_size or len(mons), len(mons))
            if battle.format and "vgc" in battle.format:
                k = min(k, 4)
            pair_synergy = F.teampreview_pair_synergy(battle) if self.lead_synergy else None
            picks = self.policy.pick_team(rows, k, pair_synergy)
            self._lead_picks[battle.battle_tag] = [mons[i].species for i in picks[:2]]
            log.info(
                "Team roles: %s | leads: %s",
                ", ".join(f"{mon.species}={F.classify_role(mon)}" for mon in mons),
                ", ".join(self._lead_picks[battle.battle_tag]),
            )
            for i in picks:
                mons[i]._selected_in_teampreview = True
            if self.learning_enabled:
                traj = self._traj.setdefault(battle.battle_tag, Trajectory())
                traj.tp_rows, traj.tp_picks = rows, picks
                traj.tp_pair_synergy = pair_synergy
            return "/team " + "".join(str(i + 1) for i in picks)
        except Exception:
            log.exception("teampreview failed, falling back to random")
            return self.random_teampreview(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        ots_timeout = (
            self.accept_open_team_sheet
            and not self.format_is_bestof
            and any(
                part.endswith(" has 30 seconds left.")
                for message in split_messages
                for part in message
            )
            and not any(
                len(message) > 2
                and message[1] == "showteam"
                and message[2] == "p2"
                for message in split_messages
            )
        )
        await super()._handle_battle_message(split_messages)
        if ots_timeout and split_messages and split_messages[0]:
            room_header = split_messages[0][0]
            if room_header.startswith(">battle-"):
                battle = await self._get_battle(room_header)
                if battle.teampreview:
                    log.info(
                        "%s: OTS wait timed out; selecting team",
                        battle.battle_tag,
                    )
                    await self._handle_battle_request(battle)

    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        try:
            if battle.wait:
                return DoubleBattleOrder(DefaultBattleOrder(), DefaultBattleOrder())
            orders, phi = F.build_candidates(battle)
            if not orders:
                return self.choose_random_move(battle)
            bias = np.zeros(len(orders))
            if self.prioritize_trick_room and Field.TRICK_ROOM not in battle.fields:
                counts = np.array([
                    sum(getattr(single.order, "id", None) == "trickroom"
                        for single in (order.first_order, order.second_order))
                    for order in orders
                ])
                eligible = np.flatnonzero(counts <= 1)
                if len(eligible):
                    orders = [orders[i] for i in eligible]
                    phi, counts = phi[eligible], counts[eligible]
                gain = phi[:, F.F_S + F.P["speed_control_gain"]]
                bias = np.where(counts == 1, 3.0 * gain, 0.0)
            idx = self.policy.act(phi, bias)
            if self.learning_enabled:
                summ = F.state_summary(battle)
                self._traj.setdefault(battle.battle_tag, Trajectory()).steps.append(
                    Step(phi=phi, chosen=idx, psi=summ["psi"], pot=summ["pot"],
                         bias=bias)
                )
            return orders[idx]
        except Exception:
            log.exception("choose_move failed, falling back to random")
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle) -> None:
        traj = self._traj.pop(battle.battle_tag, Trajectory())
        won = battle.won
        stats = {}
        if self.learning_enabled:
            try:
                stats = self.policy.learn(traj, won)
            except Exception:
                log.exception("learning step failed")
        leads = self._lead_picks.pop(battle.battle_tag, [])
        self._log_battle(battle, won, stats, leads)

    def _log_battle(self, battle, won, stats, leads: list[str]) -> None:
        rec = {
            "t": time.time(), "tag": battle.battle_tag, "format": battle.format,
            "team_id": self.team_id,
            "team_file": self.team_file,
            "won": won, "turns": battle.turn,
            "my_team": [m.species for m in battle.team.values()],
            "leads": leads,
            "lead_synergy": self.lead_synergy,
            "my_team_roles": [
                {"species": mon.species, "role": F.classify_role(mon)}
                for mon in battle.team.values()
            ],
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
