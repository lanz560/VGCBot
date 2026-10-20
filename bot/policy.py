"""A small, inspectable policy that learns after every battle.

Move policy:   pi(a | s) = softmax(Phi(s, a) @ w_move)      over all legal joint actions
Preview policy: Plackett-Luce over our team members using w_tp
Baseline:      V(s) = v @ psi(s), fit by Monte-Carlo regression

After each battle we do one policy-gradient step (REINFORCE with advantage
G_t - V(s_t)). Weights are pulled gently back toward the hand-written prior so
a few unlucky ladder streaks cannot wreck the bot.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import D_MOVE, D_TP, PRIOR_MOVE, PRIOR_TP, SINGLE_FEATURES, PAIR_FEATURES, TEAMPREVIEW_FEATURES


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Step:
    phi: np.ndarray          # (N, D_MOVE) feature rows of every legal action
    chosen: int
    psi: np.ndarray          # state features for baseline
    pot: float               # shaping potential before acting
    bias: np.ndarray         # per-action logit bias applied while choosing


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    tp_rows: np.ndarray | None = None        # (n_mons, D_TP)
    tp_picks: list[int] = field(default_factory=list)
    final_pot: float = 0.0


class Policy:
    def __init__(self, path: str | Path = "data/weights.json", lr: float = 0.05,
                 lr_value: float = 0.05, gamma: float = 0.97, prior_pull: float = 0.01,
                 terminal_reward: float = 1.0, seed: int | None = None):
        self.path = Path(path)
        self.lr, self.lr_v, self.gamma, self.prior_pull = lr, lr_value, gamma, prior_pull
        self.terminal_reward = terminal_reward
        self.rng = np.random.default_rng(seed)
        self.w = PRIOR_MOVE.copy()
        self.w_tp = PRIOR_TP.copy()
        self.v = np.zeros(5)
        self.tp_baseline = 0.0
        self.battles_seen = 0
        self.wins = 0
        self._load()

    # ------------------------------------------------------------ acting
    def act(self, phi: np.ndarray, bias: np.ndarray | None = None) -> int:
        logits = phi @ self.w
        if bias is not None:
            logits = logits + bias
        p = softmax(logits)
        return int(self.rng.choice(len(p), p=p))

    def pick_team(self, rows: np.ndarray, k: int) -> list[int]:
        """Sample k members without replacement (first two are the leads)."""
        remaining = list(range(len(rows)))
        picks: list[int] = []
        for _ in range(min(k, len(rows))):
            p = softmax(rows[remaining] @ self.w_tp)
            j = int(self.rng.choice(len(remaining), p=p))
            picks.append(remaining.pop(j))
        return picks

    # ---------------------------------------------------------- learning
    def learn(self, traj: Trajectory, won: bool | None) -> dict:
        T = len(traj.steps)
        outcome = 0.0 if won is None else (self.terminal_reward if won else -self.terminal_reward)
        pots = [s.pot for s in traj.steps] + [traj.final_pot]
        rewards = np.array([pots[t + 1] - pots[t] for t in range(T)]) if T else np.zeros(0)
        if T:
            rewards[-1] += outcome

        G = np.zeros(T)
        run = 0.0
        for t in reversed(range(T)):
            run = rewards[t] + self.gamma * run
            G[t] = run

        stats = {"turns": T, "return0": float(G[0]) if T else float(outcome)}
        if T:
            V = np.array([self.v @ s.psi for s in traj.steps])
            adv = G - V
            for s, g in zip(traj.steps, G):          # value regression
                self.v += self.lr_v * (g - self.v @ s.psi) * s.psi / T
            if T >= 3 and adv.std() > 1e-6:
                adv = adv / (adv.std() + 1e-6)
            adv = np.clip(adv, -2.5, 2.5)

            grad = np.zeros_like(self.w)
            for s, a in zip(traj.steps, adv):
                logits = s.phi @ self.w + s.bias
                p = softmax(logits)
                grad += a * (s.phi[s.chosen] - p @ s.phi)
            grad /= T
            step = self.lr * grad - self.prior_pull * (self.w - PRIOR_MOVE)
            n = np.linalg.norm(step)
            if n > 0.5:                              # trust-region-ish cap
                step *= 0.5 / n
            self.w += step
            stats["grad_norm"] = float(np.linalg.norm(grad))

        # team preview gets the whole-battle return as its signal
        if traj.tp_rows is not None and traj.tp_picks:
            g0 = float(G[0]) if T else float(outcome)
            adv_tp = float(np.clip(g0 - self.tp_baseline, -2.5, 2.5))
            self.tp_baseline += 0.1 * (g0 - self.tp_baseline)
            grad = np.zeros_like(self.w_tp)
            remaining = list(range(len(traj.tp_rows)))
            for pick in traj.tp_picks:
                p = softmax(traj.tp_rows[remaining] @ self.w_tp)
                grad += traj.tp_rows[pick] - p @ traj.tp_rows[remaining]
                remaining.remove(pick)
            step = 0.5 * self.lr * adv_tp * grad / len(traj.tp_picks) \
                - self.prior_pull * (self.w_tp - PRIOR_TP)
            self.w_tp += step

        self.battles_seen += 1
        self.wins += bool(won)
        self.save()
        return stats

    # ------------------------------------------------------------- disk
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "saved_at": time.time(), "battles_seen": self.battles_seen, "wins": self.wins,
            "w_move": self.w.tolist(), "w_tp": self.w_tp.tolist(), "v": self.v.tolist(),
            "tp_baseline": self.tp_baseline,
            "feature_names": SINGLE_FEATURES + PAIR_FEATURES, "tp_feature_names": TEAMPREVIEW_FEATURES,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=1))
        tmp.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        b = json.loads(self.path.read_text())
        if len(b.get("w_move", [])) == D_MOVE and len(b.get("w_tp", [])) == D_TP:
            self.w, self.w_tp = np.array(b["w_move"]), np.array(b["w_tp"])
            self.v, self.tp_baseline = np.array(b["v"]), b.get("tp_baseline", 0.0)
            self.battles_seen, self.wins = b.get("battles_seen", 0), b.get("wins", 0)