"""Linear softmax policy with on-policy batches and a learned value baseline.

Parameters stay fixed while a batch is collected. Each batch gets one gradient
update; old trajectories are discarded rather than replayed off-policy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import (
    D_LEAD_SYNERGY,
    D_MOVE,
    D_TP,
    D_VALUE,
    VALUE_FEATURES,
    PAIR_FEATURES,
    PRIOR_MOVE,
    PRIOR_TP,
    SINGLE_FEATURES,
    TEAMPREVIEW_FEATURES,
)

LEAD_SYNERGY_WEIGHTS = np.array([0.65, 0.8, 1.1])


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
    tp_pair_synergy: np.ndarray | None = None
    final_pot: float = 0.0


class Policy:
    def __init__(self, path: str | Path = "data/weights.json", lr: float = 0.05,
                 lr_value: float = 0.05, gamma: float = 0.99, prior_pull: float = 0.01,
                 terminal_reward: float = 1.0, seed: int | None = None,
                 batch_size: int = 16, shaping_scale: float = 0.25):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= gamma <= 1 or not np.isfinite(shaping_scale) or shaping_scale < 0:
            raise ValueError("gamma must be in [0, 1] and shaping_scale nonnegative")
        self.batch_size = batch_size
        self.shaping_scale = shaping_scale
        self._pending: list[tuple[Trajectory, bool | None]] = []
        self.updates = 0
        self.path = Path(path)
        self.lr, self.lr_v, self.gamma, self.prior_pull = lr, lr_value, gamma, prior_pull
        self.terminal_reward = terminal_reward
        self.rng = np.random.default_rng(seed)
        self.w = PRIOR_MOVE.copy()
        self.anchor = PRIOR_MOVE.copy()
        self.w_tp = PRIOR_TP.copy()
        self.v = np.zeros(D_VALUE)
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

    def pick_team(
        self,
        rows: np.ndarray,
        k: int,
        pair_synergy: np.ndarray | None = None,
    ) -> list[int]:
        """Sample a team, using pair synergy only to choose the second lead."""
        remaining = list(range(len(rows)))
        picks: list[int] = []
        for _ in range(min(k, len(rows))):
            scores = rows[remaining] @ self.w_tp
            if len(picks) == 1 and pair_synergy is not None:
                if pair_synergy.shape != (len(rows), len(rows), D_LEAD_SYNERGY):
                    raise ValueError(
                        "pair_synergy must have shape "
                        f"({len(rows)}, {len(rows)}, {D_LEAD_SYNERGY})"
                    )
                scores = scores + (
                    pair_synergy[picks[0], remaining] @ LEAD_SYNERGY_WEIGHTS
                )
            p = softmax(scores)
            j = int(self.rng.choice(len(remaining), p=p))
            picks.append(remaining.pop(j))
        return picks

    # ---------------------------------------------------------- learning
    def returns(self, traj: Trajectory, won: bool | None) -> np.ndarray:
        """Discount-consistent shaping, with zero potential at terminal states."""
        outcome = (0.0 if won is None else
                   self.terminal_reward if won else -self.terminal_reward)
        rewards = np.zeros(len(traj.steps))
        for t, step in enumerate(traj.steps):
            next_pot = traj.steps[t + 1].pot if t + 1 < len(traj.steps) else 0.0
            rewards[t] = self.shaping_scale * (self.gamma * next_pot - step.pot)
        if len(rewards):
            rewards[-1] += outcome
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    @property
    def pending_battles(self) -> int:
        return len(self._pending)

    def learn(self, traj: Trajectory, won: bool | None) -> dict:
        """Queue one completed battle, updating only when the batch is ready."""
        self._pending.append((traj, won))
        self.battles_seen += 1
        self.wins += won is True
        returns = self.returns(traj, won)
        outcome = 0.0 if won is None else self.terminal_reward if won else -self.terminal_reward
        stats = {"turns": len(traj.steps),
                 "return0": float(returns[0]) if len(returns) else float(outcome),
                 "updated": False, "pending_battles": len(self._pending)}
        if len(self._pending) >= self.batch_size:
            stats.update(self.flush())
        return stats

    def flush(self) -> dict:
        """Train on a full or final partial batch. No-op if nothing is queued."""
        if not self._pending:
            return {}
        samples, targets, preview = [], [], []
        for traj, won in self._pending:
            returns = self.returns(traj, won)
            samples.extend(traj.steps)
            targets.extend(returns)
            if traj.tp_rows is not None and traj.tp_picks:
                outcome = 0.0 if won is None else self.terminal_reward if won else -self.terminal_reward
                preview.append((traj, float(returns[0]) if len(returns) else outcome))

        old_w = self.w.copy()
        new_w, new_v, new_tp = self.w.copy(), self.v.copy(), self.w_tp.copy()
        new_baseline = self.tp_baseline
        stats = {"updated": True, "batch_battles": len(self._pending),
                 "batch_steps": len(samples), "pending_battles": 0}
        if samples:
            psi = np.vstack([s.psi for s in samples])
            targets = np.asarray(targets)
            values = psi @ self.v
            advantages = targets - values
            # Very small or constant batches do not have a useful variance estimate.
            if len(samples) >= 3 and advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            advantages = np.clip(advantages, -2.5, 2.5)
            grad = np.zeros_like(self.w)
            probabilities = []
            for sample, advantage in zip(samples, advantages):
                prob = softmax(sample.phi @ old_w + sample.bias)
                probabilities.append(prob)
                grad += advantage * (sample.phi[sample.chosen] - prob @ sample.phi)
            grad /= len(samples)
            delta = self.lr * grad - self.prior_pull * (old_w - self.anchor)
            norm = np.linalg.norm(delta)
            if norm > 0.5:
                delta *= 0.5 / norm
            new_w += delta
            new_v += self.lr_v * (psi.T @ (targets - values)) / len(samples)
            kl, entropy = [], []
            for sample, before in zip(samples, probabilities):
                after = softmax(sample.phi @ new_w + sample.bias)
                log_before = np.log(np.maximum(before, 1e-300))
                kl.append(float(before @ (log_before - np.log(np.maximum(after, 1e-300)))))
                entropy.append(float(-before @ log_before))
            stats.update(grad_norm=float(np.linalg.norm(grad)),
                         update_norm=float(np.linalg.norm(delta)),
                         mean_kl=float(np.mean(kl)), mean_entropy=float(np.mean(entropy)),
                         value_mse=float(np.mean((targets - values) ** 2)))

        if preview:
            gradient = np.zeros_like(self.w_tp)
            for traj, target in preview:
                advantage = float(np.clip(target - self.tp_baseline, -2.5, 2.5))
                remaining = list(range(len(traj.tp_rows)))
                grad = np.zeros_like(self.w_tp)
                for pick_index, pick in enumerate(traj.tp_picks):
                    scores = traj.tp_rows[remaining] @ self.w_tp
                    if pick_index == 1 and traj.tp_pair_synergy is not None:
                        expected = (len(traj.tp_rows), len(traj.tp_rows), D_LEAD_SYNERGY)
                        if traj.tp_pair_synergy.shape != expected:
                            raise ValueError(f"tp_pair_synergy must have shape {expected}")
                        scores += (traj.tp_pair_synergy[traj.tp_picks[0], remaining]
                                   @ LEAD_SYNERGY_WEIGHTS)
                    prob = softmax(scores)
                    grad += traj.tp_rows[pick] - prob @ traj.tp_rows[remaining]
                    remaining.remove(pick)
                gradient += advantage * grad / len(traj.tp_picks)
            new_tp += (0.5 * self.lr * gradient / len(preview)
                       - self.prior_pull * (self.w_tp - PRIOR_TP))
            new_baseline += 0.1 * (np.mean([target for _, target in preview]) - self.tp_baseline)

        if not all(np.all(np.isfinite(x)) for x in (new_w, new_v, new_tp, new_baseline)):
            raise ValueError("non-finite batch update; parameters unchanged")
        old_state = (self.w, self.v, self.w_tp, self.tp_baseline, self.updates)
        self.w, self.v, self.w_tp = new_w, new_v, new_tp
        self.tp_baseline = float(new_baseline)
        self.updates += 1
        try:
            self.save()
        except Exception:
            self.w, self.v, self.w_tp, self.tp_baseline, self.updates = old_state
            raise
        self._pending.clear()
        stats["updates"] = self.updates
        return stats

    # ------------------------------------------------------------- disk
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "schema_version": 2, "updates": self.updates,
            "value_feature_names": VALUE_FEATURES,
            "training": {"batch_size": self.batch_size, "gamma": self.gamma,
                         "shaping_scale": self.shaping_scale},
            "saved_at": time.time(), "battles_seen": self.battles_seen, "wins": self.wins,
            "w_move": self.w.tolist(), "w_tp": self.w_tp.tolist(), "v": self.v.tolist(),
            "tp_baseline": self.tp_baseline, "anchor_move": self.anchor.tolist(),
            "feature_names": SINGLE_FEATURES + PAIR_FEATURES, "tp_feature_names": TEAMPREVIEW_FEATURES,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=1))
        tmp.replace(self.path)
 
    def _load(self) -> None:
        if not self.path.exists():
            return
        b = json.loads(self.path.read_text())
        names = SINGLE_FEATURES + PAIR_FEATURES
        # Legacy policies had the first 30 move features and first five values.
        # Map by name when available so adding/reordering features is safe.
        def restore(key, defaults, current_names, legacy_names):
            values = b.get(key, [])
            if not values:
                return defaults.copy()
            saved_names = b.get({"w_move": "feature_names", "anchor_move": "feature_names",
                                 "w_tp": "tp_feature_names", "v": "value_feature_names"}[key])
            if saved_names is None:
                if len(values) == len(defaults):
                    saved_names = current_names
                elif len(values) == len(legacy_names):
                    saved_names = legacy_names
                else:
                    raise ValueError(f"Cannot migrate {key}: missing feature names")
            if len(values) != len(saved_names) or len(set(saved_names)) != len(saved_names):
                raise ValueError(f"Invalid feature schema for {key}")
            mapping = dict(zip(saved_names, values))
            result = np.array([mapping.get(name, defaults[i])
                               for i, name in enumerate(current_names)], dtype=float)
            if not np.all(np.isfinite(result)):
                raise ValueError(f"Non-finite weights in {key}")
            return result

        self.w = restore("w_move", self.w, names, names[:30])
        self.anchor = restore("anchor_move", self.anchor, names, names[:30])
        self.w_tp = restore("w_tp", self.w_tp, TEAMPREVIEW_FEATURES, TEAMPREVIEW_FEATURES)
        self.v = restore("v", self.v, VALUE_FEATURES, VALUE_FEATURES[:5])
        self.tp_baseline = float(b.get("tp_baseline", 0.0))
        self.battles_seen, self.wins = b.get("battles_seen", 0), b.get("wins", 0)
        self.updates = b.get("updates", 0)
