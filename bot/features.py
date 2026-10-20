"""Turn a poke-env DoubleBattle into numeric features.

Everything here is a cheap *estimate* (no full damage calc, because opponent
EVs/items are usually unknown on ladder). The policy learns how much to trust
each signal, so rough is fine as long as it is consistent.
"""
from __future__ import annotations

import numpy as np
from poke_env.battle import Pokemon
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

# ---------------------------------------------------------------- feature names
SINGLE_FEATURES = [
    "dmg_frac",          # expected damage as a fraction of target's *current* HP (capped at 1 per target)
    "ko",                # number of foes this move likely KOs
    "ally_dmg",          # fraction of ally's current HP we would remove
    "wasted_attack",     # attack that does ~nothing (immune / no target)
    "super_effective",
    "priority_ko",
    "outspeed_dmg",      # damage dealt by a move whose user outspeeds the target
    "status_move",
    "protect",
    "protect_threat",    # protect while the user is under threat
    "protect_repeat",    # protect again right after protecting (usually fails)
    "fake_out_ok",
    "fake_out_bad",
    "speed_control",     # tailwind / trick room / icy wind / thunder wave ...
    "redirect",          # follow me / rage powder
    "ally_support",      # helping hand and friends
    "self_destruct",
    "mega",
    "tera",
    "switch",
    "switch_threat",     # switching out a mon that is under threat
    "switch_matchup",    # quality of the incoming mon vs. current foes
    "forced_switch_matchup",
    "heal_when_hurt",
]
PAIR_FEATURES = [
    "focus_fire",        # both attack the same foe
    "overkill",          # both attack the same foe and one alone already KOs
    "combo_ko",          # two attacks together KO a foe neither KOs alone
    "double_protect",
    "double_switch",
    "double_status",
]
F_S, F_P = len(SINGLE_FEATURES), len(PAIR_FEATURES)
D_MOVE = F_S + F_P
S = {n: i for i, n in enumerate(SINGLE_FEATURES)}
P = {n: i for i, n in enumerate(PAIR_FEATURES)}

TEAMPREVIEW_FEATURES = [
    "offense", "danger", "outspeeds", "bulk", "power",
    "fake_out", "speed_control", "redirect", "protect",
]
D_TP = len(TEAMPREVIEW_FEATURES)

# Sensible starting point so the bot is competent from battle #1. Learning
# moves the weights away from this; a small L2 pull keeps them from running off.
PRIOR_MOVE = np.zeros(D_MOVE)
for k, v in {
    "dmg_frac": 3.0, "ko": 2.0, "ally_dmg": -3.5, "wasted_attack": -4.0,
    "super_effective": 0.4, "priority_ko": 0.8, "outspeed_dmg": 0.5,
    "status_move": -0.4, "protect": -0.2, "protect_threat": 2.0,
    "protect_repeat": -3.5, "fake_out_ok": 1.8, "fake_out_bad": -4.0,
    "speed_control": 1.0, "redirect": 0.6, "ally_support": 0.3,
    "self_destruct": -1.5, "mega": 0.3, "tera": 0.2, "switch": -1.2,
    "switch_threat": 2.0, "switch_matchup": 1.0, "forced_switch_matchup": 1.5,
    "heal_when_hurt": 0.8,
}.items():
    PRIOR_MOVE[S[k]] = v
for k, v in {
    "focus_fire": 0.4, "overkill": -1.0, "combo_ko": 1.0,
    "double_protect": -2.0, "double_switch": -1.0, "double_status": -0.6,
}.items():
    PRIOR_MOVE[F_S + P[k]] = v

PRIOR_TP = np.array([1.5, -1.5, 0.8, 0.5, 0.5, 0.6, 0.6, 0.4, 0.1])

SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave",
                 "glare", "scaryface", "stringshot", "bulldoze", "snarl"}
REDIRECT = {"followme", "ragepowder"}
SUPPORT = {"helpinghand", "coaching", "lifedew", "decorate", "afteryou"}

# --------------------------------------------------------------------- helpers
def _boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def est_stat(mon: Pokemon, key: str) -> float:
    """Exact for our mons (from the request), assumed spread for opponents."""
    known = (mon.stats or {}).get(key)
    if known:
        return float(known)
    base = mon.base_stats[key]
    lvl = mon.level or 50
    core = ((2 * base + 31 + 21) * lvl) // 100
    return float(core + lvl + 10) if key == "hp" else float(core + 5)


def _hp_abs(mon: Pokemon) -> float:
    if mon.max_hp and mon.max_hp > 100:        # our own mons carry real HP
        return float(mon.max_hp)
    return est_stat(mon, "hp")


def speed(mon: Pokemon, battle, mine: bool) -> float:
    s = est_stat(mon, "spe") * _boost_mult(mon.boosts.get("spe", 0))
    sc = battle.side_conditions if mine else battle.opponent_side_conditions
    if SideCondition.TAILWIND in sc:
        s *= 2
    if mon.status == Status.PAR:
        s *= 0.5
    if Field.TRICK_ROOM in battle.fields:
        s = -s
    return s


_ABILITY_IMMUNE = {
    "levitate": "ground", "flashfire": "fire", "waterabsorb": "water",
    "stormdrain": "water", "dryskin": "water", "voltabsorb": "electric",
    "lightningrod": "electric", "motordrive": "electric", "sapsipper": "grass",
    "wellbakedbody": "fire", "eartheater": "ground",
}


def calc_frac(att: Pokemon, mtype, cat: MoveCategory, bp: float, dfn: Pokemon) -> float:
    """Average-roll damage as a fraction of the defender's MAX hp."""
    if bp <= 0 or cat == MoveCategory.STATUS:
        return 0.0
    ak, dk = ("atk", "def") if cat == MoveCategory.PHYSICAL else ("spa", "spd")
    a = est_stat(att, ak) * _boost_mult(att.boosts.get(ak, 0))
    d = est_stat(dfn, dk) * _boost_mult(dfn.boosts.get(dk, 0))
    lvl = att.level or 50
    dmg = ((2 * lvl / 5 + 2) * bp * a / max(d, 1)) / 50 + 2
    if mtype in att.types:
        dmg *= 1.5
    if dfn.ability and _ABILITY_IMMUNE.get(dfn.ability) == mtype.name.lower():
        return 0.0
    try:
        dmg *= dfn.damage_multiplier(mtype)
    except Exception:
        pass
    if att.status == Status.BRN and cat == MoveCategory.PHYSICAL:
        dmg *= 0.5
    if att.item in ("lifeorb",):
        dmg *= 1.3
    elif att.item in ("choiceband", "choicespecs"):
        dmg *= 1.5
    return 0.925 * dmg / _hp_abs(dfn)


def _grounded(mon: Pokemon, battle) -> bool:
    if Field.GRAVITY in battle.fields:
        return True
    types = {t.name.lower() for t in mon.types if t is not None}
    return (
        "flying" not in types
        and mon.ability != "levitate"
        and mon.item != "airballoon"
    )


def _expanding_force_boosted(mon: Pokemon, move: Move, battle) -> bool:
    return (
        move.id == "expandingforce"
        and Field.PSYCHIC_TERRAIN in battle.fields
        and _grounded(mon, battle)
    )


def move_frac(att: Pokemon, move: Move, dfn: Pokemon, battle) -> float:
    bp = move.base_power or (60 if move.category != MoveCategory.STATUS else 0)
    multiplier = 1.0
    if (
        move.type.name.lower() == "psychic"
        and Field.PSYCHIC_TERRAIN in battle.fields
        and _grounded(att, battle)
    ):
        multiplier *= 1.3
        if move.id == "expandingforce":
            bp *= 1.5
    return calc_frac(att, move.type, move.category, bp, dfn) * multiplier


def incoming_threat(attacker: Pokemon, defender: Pokemon, battle) -> float:
    """Best expected hit (fraction of defender's max HP) from known moves or STAB guesses."""
    best = 0.0
    for m in (attacker.moves or {}).values():
        best = max(best, move_frac(attacker, m, defender, battle))
    ak = "atk" if attacker.base_stats["atk"] >= attacker.base_stats["spa"] else "spa"
    cat = MoveCategory.PHYSICAL if ak == "atk" else MoveCategory.SPECIAL
    for t in attacker.types:
        if t is not None:
            best = max(best, calc_frac(attacker, t, cat, 85, defender))
    return best


def _alive(mon):  # noqa: ANN001
    return mon is not None and not mon.fainted


def threat_to(battle, mon: Pokemon) -> float:
    """Summed expected damage from both opposing actives (fraction of mon's current HP, capped)."""
    tot = sum(
        incoming_threat(o, mon, battle)
        for o in battle.opponent_active_pokemon
        if _alive(o)
    )
    return float(min(1.0, tot / max(mon.current_hp_fraction, 0.05)))


def matchup(battle, mon: Pokemon) -> float:
    foes = [o for o in battle.opponent_active_pokemon if _alive(o)]
    if not foes:
        return 0.0
    off = np.mean([min(1.0, incoming_threat(mon, o, battle)) for o in foes])
    dfn = np.mean([min(1.0, incoming_threat(o, mon, battle)) for o in foes])
    return float(off - dfn)


# ------------------------------------------------------------ per-action features
def single_features(battle, slot: int, order) -> tuple[np.ndarray, dict]:
    v = np.zeros(F_S)
    aux = {"kind": "other", "tgt": None, "ko": 0, "dmg": 0.0, "protect": False}
    if not isinstance(order, SingleBattleOrder) or isinstance(order, (PassBattleOrder, DefaultBattleOrder)):
        return v, aux
    me = battle.active_pokemon[slot]
    thing = order.order

    # ------------------------------ switch
    if isinstance(thing, Pokemon):
        aux["kind"] = "switch"
        m = matchup(battle, thing)
        if battle.force_switch[slot]:
            v[S["forced_switch_matchup"]] = m
        else:
            v[S["switch"]] = 1.0
            v[S["switch_matchup"]] = m
            if me is not None:
                v[S["switch_threat"]] = threat_to(battle, me)
        return v, aux

    if not isinstance(thing, Move) or me is None:
        return v, aux

    move: Move = thing
    mid = move.id
    kind = move.entry.get("target", "normal")
    expanding_force = _expanding_force_boosted(me, move, battle)
    if expanding_force:
        kind = "allAdjacentFoes"
    foes = battle.opponent_active_pokemon
    ally = battle.active_pokemon[1 - slot]
    my_spe = speed(me, battle, True)
    attacking = move.category != MoveCategory.STATUS

    if order.mega:
        v[S["mega"]] = 1.0
    if order.terastallize:
        v[S["tera"]] = 1.0

    targets: list[tuple[Pokemon | None, float, bool]] = []   # (mon, spread mult, is_foe)
    if attacking:
        if kind == "allAdjacentFoes":
            targets = [(f, 0.75, True) for f in foes if _alive(f)]
        elif kind == "allAdjacent":
            targets = [(f, 0.75, True) for f in foes if _alive(f)]
            if _alive(ally):
                targets.append((ally, 0.75, False))
        elif kind in ("normal", "any", "adjacentFoe", "randomNormal", "scripted"):
            t = order.move_target
            if t > 0:
                targets = [(foes[t - 1], 1.0, True)]
            elif t < 0:
                targets = [(battle.active_pokemon[-t - 1], 1.0, False)]
        if not targets or all(t[0] is None or t[0].fainted for t in targets):
            v[S["wasted_attack"]] = 1.0

    acc = 1.0 if move.accuracy is True else float(move.accuracy or 1.0)
    for tmon, mult, is_foe in targets:
        if tmon is None or tmon.fainted:
            continue
        frac = move_frac(me, move, tmon, battle) * mult
        cur = max(tmon.current_hp_fraction, 0.02)
        part = min(1.0, frac / cur)
        if is_foe:
            v[S["dmg_frac"]] += acc * part
            if frac >= 0.95 * cur and acc >= 0.85:
                v[S["ko"]] += 1
                aux["ko"] += 1
                if move.priority > 0:
                    v[S["priority_ko"]] = 1.0
            if frac <= 0.01 and len(targets) == 1:
                v[S["wasted_attack"]] = 1.0
            try:
                if tmon.damage_multiplier(move) >= 2:
                    v[S["super_effective"]] = 1.0
            except Exception:
                pass
            if my_spe > speed(tmon, battle, False):
                v[S["outspeed_dmg"]] += part
            aux["dmg"] += frac / cur
            if len(targets) == 1 or kind == "normal":
                aux["tgt"] = order.move_target
        else:
            v[S["ally_dmg"]] += part
    aux["kind"] = "attack" if attacking else "status"

    # ------------------------------ non-damage behaviour
    if not attacking:
        v[S["status_move"]] = 1.0
    if move.is_protect_move:
        v[S["protect"]] = 1.0
        v[S["protect_threat"]] = threat_to(battle, me)
        if getattr(me, "protect_counter", 0) > 0:
            v[S["protect_repeat"]] = 1.0
        aux["protect"] = True
    if mid == "fakeout":
        v[S["fake_out_ok" if me.first_turn else "fake_out_bad"]] = 1.0
    if mid in SPEED_CONTROL:
        v[S["speed_control"]] = 1.0
    if mid in REDIRECT:
        v[S["redirect"]] = 1.0
    if mid in SUPPORT:
        v[S["ally_support"]] = 1.0
    if move.self_destruct:
        v[S["self_destruct"]] = 1.0
    if move.heal and move.heal > 0:
        v[S["heal_when_hurt"]] = (1.0 - me.current_hp_fraction) * float(move.heal) * 2
    return v, aux


def pair_features(a1: dict, a2: dict) -> np.ndarray:
    p = np.zeros(F_P)
    if a1["kind"] == "attack" and a2["kind"] == "attack" and a1["tgt"] and a1["tgt"] == a2["tgt"]:
        p[P["focus_fire"]] = 1.0
        if a1["ko"] or a2["ko"]:
            p[P["overkill"]] = 1.0
        elif a1["dmg"] + a2["dmg"] >= 0.95:
            p[P["combo_ko"]] = 1.0
    if a1["protect"] and a2["protect"]:
        p[P["double_protect"]] = 1.0
    if a1["kind"] == "switch" and a2["kind"] == "switch":
        p[P["double_switch"]] = 1.0
    if a1["kind"] == "status" and a2["kind"] == "status":
        p[P["double_status"]] = 1.0
    return p


def build_candidates(battle):
    """All legal joint orders with their feature rows.

    Returns (orders, Phi) where Phi has shape (N, D_MOVE).
    """
    lists = battle.valid_orders
    singles = []
    for slot in (0, 1):
        opts = lists[slot] or [PassBattleOrder()]
        singles.append([(o, *single_features(battle, slot, o)) for o in opts])

    orders, rows = [], []
    for o1, v1, a1 in singles[0]:
        for o2, v2, a2 in singles[1]:
            # same legality filters as poke-env's DoubleBattleOrder.join_orders
            if isinstance(o1, PassBattleOrder) and isinstance(o2, PassBattleOrder):
                continue
            if o1.mega and o2.mega or o1.terastallize and o2.terastallize:
                continue
            if o1.z_move and o2.z_move or o1.dynamax and o2.dynamax:
                continue
            if isinstance(o1.order, Pokemon) and o1.order is o2.order:
                continue
            orders.append(DoubleBattleOrder(first_order=o1, second_order=o2))
            rows.append(np.concatenate([v1 + v2, pair_features(a1, a2)]))
    if not orders:
        return [], np.zeros((0, D_MOVE))
    return orders, np.vstack(rows)


# --------------------------------------------------------------- team preview
def teampreview_features(battle) -> np.ndarray:
    mine = list(battle.team.values())
    theirs = battle.teampreview_opponent_team or list(battle.opponent_team.values())
    rows = []
    for m in mine:
        mtypes = [t for t in m.types if t is not None]
        off, dng, outs = [], [], []
        for o in theirs:
            try:
                off.append(max(o.damage_multiplier(t) for t in mtypes))
                dng.append(max(m.damage_multiplier(t) for t in o.types if t is not None))
            except Exception:
                off.append(1.0)
                dng.append(1.0)
            outs.append(1.0 if est_stat(m, "spe") > est_stat(o, "spe") else 0.0)
        move_ids = set((m.moves or {}).keys())
        bulk = est_stat(m, "hp") * (est_stat(m, "def") + est_stat(m, "spd")) / 60000
        power = max(est_stat(m, "atk"), est_stat(m, "spa")) / 150
        rows.append([
            np.mean(off) - 1 if off else 0.0,
            np.mean(dng) - 1 if dng else 0.0,
            np.mean(outs) if outs else 0.5,
            bulk,
            power,
            float("fakeout" in move_ids),
            float(bool(move_ids & SPEED_CONTROL)),
            float(bool(move_ids & REDIRECT)),
            float(any(i in move_ids for i in ("protect", "detect", "spikyshield", "kingsshield", "banefulbunker"))),
        ])
    return np.array(rows) if rows else np.zeros((0, D_TP))


# ------------------------------------------------------ battle state / shaping
def _team_mons(battle) -> list[Pokemon]:
    mons = list(battle.team.values())
    chosen = [m for m in mons if getattr(m, "_selected_in_teampreview", False)]
    return chosen or mons


def state_summary(battle) -> dict:
    """Potential function for reward shaping + features for the value baseline."""
    n = max(1, min(battle.max_team_size or 4, 6))
    mine = _team_mons(battle)
    my_lost = sum(1 - m.current_hp_fraction for m in mine)
    opp_lost = sum(1 - m.current_hp_fraction for m in battle.opponent_team.values())
    my_f = sum(m.fainted for m in mine)
    opp_f = sum(m.fainted for m in battle.opponent_team.values())
    pot = (opp_lost - my_lost) / n
    psi = np.array([1.0, pot, (n - my_f) / n, (n - opp_f) / n, min(battle.turn, 30) / 30])
    return {"pot": float(pot), "psi": psi}