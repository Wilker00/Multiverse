"""
tools/validate_sf_transfer.py

Validation harness for "Multiverse V2" transfer hypothesis:
1) Universal perceptual interface via egocentric local occupancy grid.
2) Successor Features (SF) transfer: dynamics (psi) transferred separately from reward weights (w).
3) Semantic bridge style task preference vector as w initialization.

This tool is intentionally self-contained and does not modify runtime agents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.types import VerseSpec
from verses.registry import create_verse, register_builtin


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No valid seeds parsed")
    return uniq


def _parse_int_grid(raw: str, *, default: Sequence[int]) -> List[int]:
    txt = str(raw or "").strip()
    if not txt:
        vals = [int(x) for x in default]
        return sorted(set(vals))
    out: List[int] = []
    for part in txt.replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No valid integer grid values parsed")
    return uniq


def _parse_str_list(raw: str, *, default: Sequence[str]) -> List[str]:
    txt = str(raw or "").strip()
    if not txt:
        return [str(x).strip() for x in default if str(x).strip()]
    out = [str(x).strip() for x in txt.replace(";", ",").split(",") if str(x).strip()]
    if not out:
        raise ValueError("No valid string list values parsed")
    return out


@dataclass
class EgoObservation:
    occupancy: np.ndarray  # [K, K] in {0,1}
    goal: np.ndarray  # [K, K] in {0,1}


class EgoGridAdapter:
    def __init__(self, size: int = 5):
        if size < 3 or size % 2 == 0:
            raise ValueError("ego grid size must be odd and >=3")
        self.size = int(size)
        self.radius = self.size // 2
        self._ray_dirs: List[Tuple[int, int]] = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]

    def _blank(self) -> EgoObservation:
        occ = np.zeros((self.size, self.size), dtype=np.int8)
        goal = np.zeros((self.size, self.size), dtype=np.int8)
        return EgoObservation(occupancy=occ, goal=goal)

    def _mark_goal_direction(self, *, goal: np.ndarray, dgx: int, dgy: int) -> None:
        """
        Mark goal location if in-window; otherwise project direction to ego-grid edge.
        This preserves task-solving signal for long-range navigation.
        """
        if goal.size <= 0:
            return
        c = self.radius
        if -c <= dgx <= c and -c <= dgy <= c:
            goal[dgy + c, dgx + c] = 1
            return
        sx = 0 if dgx == 0 else (1 if dgx > 0 else -1)
        sy = 0 if dgy == 0 else (1 if dgy > 0 else -1)
        gx = c + (sx * c)
        gy = c + (sy * c)
        gx = max(0, min(self.size - 1, int(gx)))
        gy = max(0, min(self.size - 1, int(gy)))
        goal[gy, gx] = 1

    def from_grid_world(self, verse: Any, obs: Dict[str, Any]) -> EgoObservation:
        ego = self._blank()
        x = _safe_int(obs.get("x", 0), 0)
        y = _safe_int(obs.get("y", 0), 0)
        gx = _safe_int(obs.get("goal_x", 0), 0)
        gy = _safe_int(obs.get("goal_y", 0), 0)
        width = _safe_int(getattr(getattr(verse, "params", None), "width", 0), 0)
        height = _safe_int(getattr(getattr(verse, "params", None), "height", 0), 0)
        obstacles = getattr(verse, "_obstacles", set())

        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                lx = dx + self.radius
                ly = dy + self.radius
                wx = x + dx
                wy = y + dy
                blocked = False
                if wx < 0 or wx >= width or wy < 0 or wy >= height:
                    blocked = True
                elif (wx, wy) in obstacles:
                    blocked = True
                ego.occupancy[ly, lx] = 1 if blocked else 0

        dgx = gx - x
        dgy = gy - y
        self._mark_goal_direction(goal=ego.goal, dgx=int(dgx), dgy=int(dgy))
        return ego

    def from_warehouse_world(self, obs: Dict[str, Any]) -> EgoObservation:
        ego = self._blank()
        lidar_raw = obs.get("lidar", [])
        lidar: List[int] = []
        if isinstance(lidar_raw, list):
            for i in range(8):
                v = lidar_raw[i] if i < len(lidar_raw) else self.radius + 1
                lidar.append(max(1, _safe_int(v, self.radius + 1)))
        else:
            lidar = [self.radius + 1] * 8

        # Approximate occupancy from lidar rays.
        for i, (dx, dy) in enumerate(self._ray_dirs):
            dist = lidar[i]
            for k in range(1, self.radius + 1):
                lx = self.radius + dx * k
                ly = self.radius + dy * k
                if lx < 0 or lx >= self.size or ly < 0 or ly >= self.size:
                    continue
                if k >= dist:
                    ego.occupancy[ly, lx] = 1
                    break
                ego.occupancy[ly, lx] = 0

        x = _safe_int(obs.get("x", 0), 0)
        y = _safe_int(obs.get("y", 0), 0)
        gx = _safe_int(obs.get("goal_x", 0), 0)
        gy = _safe_int(obs.get("goal_y", 0), 0)
        dgx = gx - x
        dgy = gy - y
        self._mark_goal_direction(goal=ego.goal, dgx=int(dgx), dgy=int(dgy))
        return ego

    def from_maze_world(self, obs: Dict[str, Any]) -> EgoObservation:
        ego = self._blank()
        c = self.radius
        # Local wall sensors map to immediate blocked neighbors around agent.
        if _safe_int(obs.get("wall_n", 0), 0) != 0 and c - 1 >= 0:
            ego.occupancy[c - 1, c] = 1
        if _safe_int(obs.get("wall_s", 0), 0) != 0 and c + 1 < self.size:
            ego.occupancy[c + 1, c] = 1
        if _safe_int(obs.get("wall_w", 0), 0) != 0 and c - 1 >= 0:
            ego.occupancy[c, c - 1] = 1
        if _safe_int(obs.get("wall_e", 0), 0) != 0 and c + 1 < self.size:
            ego.occupancy[c, c + 1] = 1

        x = _safe_int(obs.get("x", 0), 0)
        y = _safe_int(obs.get("y", 0), 0)
        gx = _safe_int(obs.get("exit_x", obs.get("goal_x", 0)), 0)
        gy = _safe_int(obs.get("exit_y", obs.get("goal_y", 0)), 0)
        dgx = gx - x
        dgy = gy - y
        self._mark_goal_direction(goal=ego.goal, dgx=int(dgx), dgy=int(dgy))
        return ego

    def extract(self, *, verse_name: str, verse: Any, obs: Dict[str, Any]) -> EgoObservation:
        v = str(verse_name).strip().lower()
        if v == "grid_world":
            return self.from_grid_world(verse, obs)
        if v == "warehouse_world":
            return self.from_warehouse_world(obs)
        if v == "maze_world":
            return self.from_maze_world(obs)
        raise ValueError(f"Unsupported verse for EgoGridAdapter: {verse_name}")

    def phi(self, ego: EgoObservation) -> np.ndarray:
        # Shared feature basis for SF:
        # [bias, occupancy_flat, goal_flat]
        occ = ego.occupancy.astype(np.float32).reshape(-1)
        goal = ego.goal.astype(np.float32).reshape(-1)
        return np.concatenate([np.array([1.0], dtype=np.float32), occ, goal], axis=0)

    def state_key(self, ego: EgoObservation) -> str:
        occ = "".join("1" if int(v) else "0" for v in ego.occupancy.reshape(-1).tolist())
        goal = "".join("1" if int(v) else "0" for v in ego.goal.reshape(-1).tolist())
        return f"{occ}|{goal}"


class TabularSFAgent:
    def __init__(
        self,
        *,
        n_actions: int,
        feature_dim: int,
        gamma: float = 0.97,
        psi_lr: float = 0.22,
        w_lr: float = 0.05,
        fwd_lr: float = 0.02,
        allowed_actions: Optional[Sequence[int]] = None,
    ):
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.psi_lr = float(psi_lr)
        self.w_lr = float(w_lr)
        self.fwd_lr = float(fwd_lr)
        self.allowed_actions = list(allowed_actions) if allowed_actions else list(range(self.n_actions))

        self.psi_table: Dict[str, np.ndarray] = {}
        self.w = np.zeros((self.feature_dim,), dtype=np.float32)
        # Linear auxiliary head per action: phi_next ~= F[a] @ phi
        self.forward_model = np.zeros((self.n_actions, self.feature_dim, self.feature_dim), dtype=np.float32)
        self._gpi_psi_banks: List[Dict[str, np.ndarray]] = []

    def copy_psi_from(self, src: "TabularSFAgent") -> None:
        copy_actions = min(self.n_actions, src.n_actions)
        for k, arr in src.psi_table.items():
            dst = np.zeros((self.n_actions, self.feature_dim), dtype=np.float32)
            dst[:copy_actions, :] = arr[:copy_actions, :]
            self.psi_table[k] = dst

    def set_w(self, w: np.ndarray) -> None:
        self.w = np.asarray(w, dtype=np.float32).copy()

    def clone(self) -> "TabularSFAgent":
        out = TabularSFAgent(
            n_actions=int(self.n_actions),
            feature_dim=int(self.feature_dim),
            gamma=float(self.gamma),
            psi_lr=float(self.psi_lr),
            w_lr=float(self.w_lr),
            fwd_lr=float(self.fwd_lr),
            allowed_actions=list(self.allowed_actions),
        )
        out.psi_table = {k: np.asarray(v, dtype=np.float32).copy() for k, v in self.psi_table.items()}
        out.w = np.asarray(self.w, dtype=np.float32).copy()
        out.forward_model = np.asarray(self.forward_model, dtype=np.float32).copy()
        return out

    def set_gpi_banks(self, banks: Sequence[Dict[str, np.ndarray]]) -> None:
        self._gpi_psi_banks = []
        for bank in banks:
            copied: Dict[str, np.ndarray] = {}
            for k, v in bank.items():
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim != 2:
                    continue
                if int(arr.shape[1]) != int(self.feature_dim):
                    continue
                if int(arr.shape[0]) != int(self.n_actions):
                    continue
                copied[str(k)] = arr.copy()
            self._gpi_psi_banks.append(copied)

    def _psi_state(self, key: str) -> np.ndarray:
        arr = self.psi_table.get(key)
        if arr is None:
            arr = np.zeros((self.n_actions, self.feature_dim), dtype=np.float32)
            self.psi_table[key] = arr
        return arr

    def _q_values_self(self, key: str) -> np.ndarray:
        psi = self._psi_state(key)
        return psi @ self.w

    def q_values(self, key: str) -> np.ndarray:
        q = self._q_values_self(key)
        if not self._gpi_psi_banks:
            return q
        best = q.copy()
        for bank in self._gpi_psi_banks:
            arr = bank.get(key)
            if arr is None:
                continue
            qb = arr @ self.w
            best = np.maximum(best, qb.astype(np.float32))
        return best

    def select_action(self, key: str, epsilon: float, rng: random.Random) -> int:
        if rng.random() < float(epsilon):
            return int(rng.choice(self.allowed_actions))
        q = self.q_values(key)
        best = self.allowed_actions[0]
        best_q = float(q[best])
        for a in self.allowed_actions[1:]:
            qa = float(q[a])
            if qa > best_q:
                best = int(a)
                best_q = qa
        return int(best)

    def update(
        self,
        *,
        s_key: str,
        a: int,
        phi_s: np.ndarray,
        reward: float,
        sp_key: str,
        phi_sp: np.ndarray,
        done: bool,
        learn_psi: bool,
        learn_w: bool,
    ) -> Dict[str, float]:
        psi_s = self._psi_state(s_key)
        psi_sp = self._psi_state(sp_key)

        # Use next-state features as the immediate SF basis because environment rewards
        # are primarily emitted from transition outcomes (goal reached, wall bump, etc.).
        if done:
            target_vec = phi_sp
        else:
            q_sp = self.q_values(sp_key)
            a_next = max(self.allowed_actions, key=lambda idx: float(q_sp[idx]))
            target_vec = phi_sp + self.gamma * psi_sp[a_next]

        td_vec = target_vec - psi_s[a]
        if learn_psi:
            psi_s[a] = psi_s[a] + self.psi_lr * td_vec
            psi_s[a] = np.clip(psi_s[a], -200.0, 200.0)

        pred_r = float(np.dot(self.w, phi_sp))
        err_r = float(reward) - pred_r
        if learn_w:
            self.w = self.w + self.w_lr * err_r * phi_sp
            self.w = np.clip(self.w, -20.0, 20.0)

        # Auxiliary dynamics objective: one-step next feature prediction.
        # This is the explicit "predict next local grid state" path.
        pred_phi_sp = self.forward_model[a] @ phi_s
        fwd_err_vec = phi_sp - pred_phi_sp
        self.forward_model[a] = self.forward_model[a] + self.fwd_lr * np.outer(fwd_err_vec, phi_s)
        self.forward_model[a] = np.clip(self.forward_model[a], -20.0, 20.0)

        fwd_mse = float(np.mean((fwd_err_vec) ** 2))
        return {
            "td_abs_mean": float(np.mean(np.abs(td_vec))),
            "reward_pred_err_abs": abs(err_r),
            "forward_mse": fwd_mse,
        }

    def predict_forward_mse(self, *, a: int, phi_s: np.ndarray, phi_sp: np.ndarray) -> float:
        pred = self.forward_model[a] @ phi_s
        return float(np.mean((phi_sp - pred) ** 2))


@dataclass
class EpisodeStats:
    return_sum: float
    success: bool
    steps: int
    hazards: int
    mean_fwd_mse: float


def _semantic_reward_weights(*, feature_dim: int, grid_size: int) -> np.ndarray:
    # Step-3 style semantic initialization:
    # high penalty for blocked occupancy; high reward for goal indicators.
    w = np.zeros((feature_dim,), dtype=np.float32)
    occ_start = 1
    occ_end = 1 + grid_size * grid_size
    goal_start = occ_end
    goal_end = goal_start + grid_size * grid_size
    w[0] = -0.03  # small living cost
    w[occ_start:occ_end] = -1.2
    w[goal_start:goal_end] = 2.2
    return w


def _project_psi_table_to_actions(
    *,
    src_table: Dict[str, np.ndarray],
    src_n_actions: int,
    dst_n_actions: int,
    feature_dim: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    copy_actions = min(int(src_n_actions), int(dst_n_actions))
    for k, v in src_table.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            continue
        if int(arr.shape[1]) != int(feature_dim):
            continue
        dst = np.zeros((int(dst_n_actions), int(feature_dim)), dtype=np.float32)
        dst[:copy_actions, :] = arr[:copy_actions, :]
        out[str(k)] = dst
    return out


def _ridge_reward_weights(
    *,
    feature_dim: int,
    features: List[np.ndarray],
    rewards: List[float],
    l2: float = 1.0,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not features or not rewards or len(features) != len(rewards):
        if fallback is None:
            return np.zeros((feature_dim,), dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32).copy()
    x = np.asarray([np.asarray(f, dtype=np.float32) for f in features], dtype=np.float32)
    y = np.asarray([float(r) for r in rewards], dtype=np.float32)
    d = int(feature_dim)
    try:
        xtx = (x.T @ x).astype(np.float32)
        reg = np.eye(d, dtype=np.float32) * float(max(1e-6, l2))
        xty = (x.T @ y).astype(np.float32)
        w = np.linalg.solve(xtx + reg, xty).astype(np.float32)
        return np.clip(w, -20.0, 20.0)
    except Exception:
        if fallback is None:
            return np.zeros((feature_dim,), dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32).copy()


def _estimate_target_reward_weights(
    *,
    verse: Any,
    verse_name: str,
    adapter: EgoGridAdapter,
    agent: TabularSFAgent,
    steps_budget: int,
    max_steps_per_episode: int,
    rng: random.Random,
    epsilon: float = 0.35,
    ridge_l2: float = 1.0,
) -> Dict[str, Any]:
    budget = max(0, int(steps_budget))
    if budget <= 0:
        return {"used_steps": 0, "episodes": 0, "w_estimated": False}
    feats: List[np.ndarray] = []
    rews: List[float] = []
    used_steps = 0
    episodes = 0
    while used_steps < budget:
        rr = verse.reset()
        obs = rr.obs if isinstance(rr.obs, dict) else {}
        episodes += 1
        for _ in range(max(1, int(max_steps_per_episode))):
            ego = adapter.extract(verse_name=verse_name, verse=verse, obs=obs)
            s_key = adapter.state_key(ego)
            phi_s = adapter.phi(ego)
            a = agent.select_action(s_key, epsilon=float(epsilon), rng=rng)
            sr = verse.step(a)
            feats.append(phi_s.astype(np.float32, copy=True))
            rews.append(float(sr.reward))
            used_steps += 1
            obs = sr.obs if isinstance(sr.obs, dict) else {}
            if bool(sr.done or sr.truncated) or used_steps >= budget:
                break
    prev_w = np.asarray(agent.w, dtype=np.float32).copy()
    w_hat = _ridge_reward_weights(
        feature_dim=int(agent.feature_dim),
        features=feats,
        rewards=rews,
        l2=float(ridge_l2),
        fallback=prev_w,
    )
    agent.set_w(w_hat)
    return {
        "used_steps": int(used_steps),
        "episodes": int(episodes),
        "w_estimated": bool(used_steps > 0),
        "w_l2_norm": float(np.linalg.norm(w_hat)),
        "w_delta_l2": float(np.linalg.norm(w_hat - prev_w)),
        "ridge_l2": float(ridge_l2),
    }


def _build_verse(
    *,
    verse_name: str,
    seed: int,
    params: Dict[str, Any],
) -> Any:
    spec = VerseSpec(
        spec_version="v1",
        verse_name=verse_name,
        verse_version="0.1",
        seed=int(seed),
        params=dict(params),
    )
    verse = create_verse(spec)
    verse.seed(int(seed))
    return verse


def _hazard_count(info: Dict[str, Any]) -> int:
    keys = [
        "hit_wall",
        "bumped_wall",
        "hit_obstacle",
        "hit_patrol",
        "battery_death",
        "battery_depleted",
        "hit_hazard",
        "fell_pit",
        "hit_laser",
    ]
    count = 0
    for k in keys:
        if bool(info.get(k, False)):
            count += 1
    return int(count)


def _epsilon_linear(ep_idx: int, total_episodes: int, start: float, end: float) -> float:
    if total_episodes <= 1:
        return float(end)
    frac = float(ep_idx) / float(max(1, total_episodes - 1))
    return float(start + (end - start) * frac)


def _run_episode(
    *,
    verse: Any,
    verse_name: str,
    adapter: EgoGridAdapter,
    agent: TabularSFAgent,
    max_steps: int,
    train: bool,
    epsilon: float,
    rng: random.Random,
    learn_psi: bool = True,
    learn_w: bool = True,
) -> EpisodeStats:
    rr = verse.reset()
    obs = rr.obs if isinstance(rr.obs, dict) else {}
    total = 0.0
    success = False
    hazards = 0
    fwd_mses: List[float] = []

    for step in range(int(max_steps)):
        ego = adapter.extract(verse_name=verse_name, verse=verse, obs=obs)
        s_key = adapter.state_key(ego)
        phi_s = adapter.phi(ego)
        a = agent.select_action(s_key, epsilon if train else 0.0, rng)

        sr = verse.step(a)
        next_obs = sr.obs if isinstance(sr.obs, dict) else {}
        total += float(sr.reward)
        info = sr.info if isinstance(sr.info, dict) else {}
        hazards += _hazard_count(info)
        if bool(info.get("reached_goal", False)):
            success = True

        ego_sp = adapter.extract(verse_name=verse_name, verse=verse, obs=next_obs)
        sp_key = adapter.state_key(ego_sp)
        phi_sp = adapter.phi(ego_sp)
        done = bool(sr.done or sr.truncated)

        if train:
            upd = agent.update(
                s_key=s_key,
                a=int(a),
                phi_s=phi_s,
                reward=float(sr.reward),
                sp_key=sp_key,
                phi_sp=phi_sp,
                done=done,
                learn_psi=bool(learn_psi),
                learn_w=bool(learn_w),
            )
            fwd_mses.append(float(upd["forward_mse"]))
        else:
            fwd_mses.append(float(agent.predict_forward_mse(a=int(a), phi_s=phi_s, phi_sp=phi_sp)))

        obs = next_obs
        if done:
            return EpisodeStats(
                return_sum=float(total),
                success=bool(success),
                steps=int(step + 1),
                hazards=int(hazards),
                mean_fwd_mse=float(sum(fwd_mses) / float(max(1, len(fwd_mses)))),
            )

    return EpisodeStats(
        return_sum=float(total),
        success=bool(success),
        steps=int(max_steps),
        hazards=int(hazards),
        mean_fwd_mse=float(sum(fwd_mses) / float(max(1, len(fwd_mses)))),
    )


def _summarize(stats: List[EpisodeStats]) -> Dict[str, Any]:
    if not stats:
        return {
            "episodes": 0,
            "mean_return": 0.0,
            "success_rate": 0.0,
            "hazard_per_1k": 0.0,
            "mean_steps": 0.0,
            "mean_forward_mse": 0.0,
        }
    returns = [s.return_sum for s in stats]
    wins = [1.0 if s.success else 0.0 for s in stats]
    steps = [float(s.steps) for s in stats]
    hazards = [float(s.hazards) for s in stats]
    fwd = [float(s.mean_fwd_mse) for s in stats]
    total_steps = max(1.0, float(sum(steps)))
    total_hazards = float(sum(hazards))
    return {
        "episodes": int(len(stats)),
        "mean_return": float(sum(returns) / float(len(returns))),
        "median_return": float(statistics.median(returns)),
        "success_rate": float(sum(wins) / float(len(wins))),
        "hazard_per_1k": float(1000.0 * total_hazards / total_steps),
        "mean_steps": float(sum(steps) / float(len(steps))),
        "mean_forward_mse": float(sum(fwd) / float(len(fwd))),
    }


def _slope(vals: Sequence[float]) -> float:
    n = int(len(vals))
    if n <= 1:
        return 0.0
    xs = [float(i) for i in range(n)]
    ys = [float(v) for v in vals]
    mx = float(sum(xs) / float(n))
    my = float(sum(ys) / float(n))
    varx = float(sum((x - mx) ** 2 for x in xs))
    if varx <= 1e-12:
        return 0.0
    cov = float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)))
    return float(cov / varx)


def _episode_trace(stats: List[EpisodeStats], *, max_points: int = 10) -> Dict[str, Any]:
    k = max(0, min(int(max_points), len(stats)))
    if k <= 0:
        return {
            "episodes": 0,
            "return": [],
            "success": [],
            "hazard_per_1k": [],
            "forward_mse": [],
            "steps": [],
        }
    out_ret: List[float] = []
    out_succ: List[float] = []
    out_haz: List[float] = []
    out_fwd: List[float] = []
    out_steps: List[float] = []
    for s in stats[:k]:
        steps = float(max(1, int(s.steps)))
        out_ret.append(float(s.return_sum))
        out_succ.append(1.0 if bool(s.success) else 0.0)
        out_haz.append(float(1000.0 * float(s.hazards) / steps))
        out_fwd.append(float(s.mean_fwd_mse))
        out_steps.append(float(steps))
    return {
        "episodes": int(k),
        "return": out_ret,
        "success": out_succ,
        "hazard_per_1k": out_haz,
        "forward_mse": out_fwd,
        "steps": out_steps,
    }


def _trace_diagnostics(trace: Dict[str, Any]) -> Dict[str, Any]:
    def _arr(name: str) -> List[float]:
        raw = trace.get(name, [])
        if not isinstance(raw, list):
            return []
        return [float(x) for x in raw if isinstance(x, (int, float))]

    ret = _arr("return")
    suc = _arr("success")
    haz = _arr("hazard_per_1k")
    fwd = _arr("forward_mse")
    n = int(min(len(ret), len(suc), len(haz), len(fwd))) if ret and suc and haz and fwd else int(max(len(ret), len(suc), len(haz), len(fwd)))

    def _first_last_delta(vals: List[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        return float(vals[-1] - vals[0])

    return {
        "episodes": int(trace.get("episodes", n) or n),
        "return_slope": _slope(ret),
        "success_slope": _slope(suc),
        "hazard_slope": _slope(haz),
        "forward_mse_slope": _slope(fwd),
        "return_delta_first_last": _first_last_delta(ret),
        "success_delta_first_last": _first_last_delta(suc),
        "hazard_delta_first_last": _first_last_delta(haz),
        "forward_mse_delta_first_last": _first_last_delta(fwd),
        "return_mean": (0.0 if not ret else float(sum(ret) / float(len(ret)))),
        "success_mean": (0.0 if not suc else float(sum(suc) / float(len(suc)))),
        "hazard_mean": (0.0 if not haz else float(sum(haz) / float(len(haz)))),
        "forward_mse_mean": (0.0 if not fwd else float(sum(fwd) / float(len(fwd)))),
    }


def _trace_delta(a: Dict[str, Any], b: Dict[str, Any], *, max_points: int = 10) -> Dict[str, Any]:
    # delta = a - b on aligned early episodes
    n = min(
        int(a.get("episodes", 0) or 0),
        int(b.get("episodes", 0) or 0),
        int(max_points),
    )
    if n <= 0:
        empty = {"episodes": 0, "return": [], "success": [], "hazard_per_1k": [], "forward_mse": []}
        return {"trace": empty, "diagnostics": _trace_diagnostics(empty)}

    out = {"episodes": int(n), "return": [], "success": [], "hazard_per_1k": [], "forward_mse": []}
    for key in ("return", "success", "hazard_per_1k", "forward_mse"):
        xa = a.get(key, [])
        xb = b.get(key, [])
        if not isinstance(xa, list) or not isinstance(xb, list):
            out[key] = [0.0] * int(n)
            continue
        vals: List[float] = []
        for i in range(int(n)):
            va = float(xa[i]) if i < len(xa) and isinstance(xa[i], (int, float)) else 0.0
            vb = float(xb[i]) if i < len(xb) and isinstance(xb[i], (int, float)) else 0.0
            vals.append(float(va - vb))
        out[key] = vals
    return {"trace": out, "diagnostics": _trace_diagnostics(out)}


def _collect_probe_egos(
    *,
    verse_name: str,
    verse: Any,
    adapter: EgoGridAdapter,
    max_steps: int,
    rng: random.Random,
    probe_count: int = 24,
) -> List[EgoObservation]:
    probes: List[EgoObservation] = []
    tries = 0
    while len(probes) < int(probe_count) and tries < int(max(1, probe_count) * 4):
        tries += 1
        rr = verse.reset()
        obs = rr.obs if isinstance(rr.obs, dict) else {}
        for _ in range(max(1, int(max_steps))):
            try:
                probes.append(adapter.extract(verse_name=verse_name, verse=verse, obs=obs))
            except Exception:
                break
            if len(probes) >= int(probe_count):
                break
            n_actions = int(getattr(getattr(verse, "action_space", None), "n", 0) or 0)
            if n_actions <= 0:
                break
            a = int(rng.randrange(n_actions))
            sr = verse.step(a)
            obs = sr.obs if isinstance(sr.obs, dict) else {}
            if bool(sr.done or sr.truncated):
                break
    return probes


def _policy_bank_agreement_diag(
    *,
    snapshots: List["TabularSFAgent"],
    probes: List[EgoObservation],
    adapter: EgoGridAdapter,
) -> Dict[str, Any]:
    if not snapshots or not probes:
        return {
            "num_snapshots": int(len(snapshots)),
            "num_probes": int(len(probes)),
            "evaluated_probes": 0,
            "mean_majority_fraction": 0.0,
            "mean_unique_actions": 0.0,
            "mean_vote_margin": 0.0,
        }
    probe_majority: List[float] = []
    probe_unique: List[float] = []
    probe_margin: List[float] = []
    local_rng = random.Random(0)
    for ego in probes:
        key = adapter.state_key(ego)
        votes: Dict[int, int] = {}
        for snap in snapshots:
            a = int(snap.select_action(key, 0.0, local_rng))
            votes[a] = int(votes.get(a, 0) + 1)
        counts = sorted(votes.values(), reverse=True)
        if not counts:
            continue
        top = int(counts[0])
        second = int(counts[1]) if len(counts) > 1 else 0
        n = int(sum(counts))
        probe_majority.append(float(top / float(max(1, n))))
        probe_unique.append(float(len(votes)))
        probe_margin.append(float((top - second) / float(max(1, n))))
    if not probe_majority:
        return {
            "num_snapshots": int(len(snapshots)),
            "num_probes": int(len(probes)),
            "evaluated_probes": 0,
            "mean_majority_fraction": 0.0,
            "mean_unique_actions": 0.0,
            "mean_vote_margin": 0.0,
        }
    return {
        "num_snapshots": int(len(snapshots)),
        "num_probes": int(len(probes)),
        "evaluated_probes": int(len(probe_majority)),
        "mean_majority_fraction": float(sum(probe_majority) / float(len(probe_majority))),
        "mean_unique_actions": float(sum(probe_unique) / float(len(probe_unique))),
        "mean_vote_margin": float(sum(probe_margin) / float(len(probe_margin))),
    }


def _train_then_eval(
    *,
    seed: int,
    adapter: EgoGridAdapter,
    source_verse_name: str,
    target_verse_name: str,
    source_params: Dict[str, Any],
    target_params: Dict[str, Any],
    source_train_episodes: int,
    target_train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    warmup_psi_episodes: int,
    source_allowed_actions: Optional[Sequence[int]] = None,
    target_allowed_actions: Optional[Sequence[int]] = None,
    target_w_estimation_steps: int = 0,
    source_policy_snapshots: int = 3,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    np.random.seed(int(seed))
    src_name = str(source_verse_name).strip().lower()
    trg_name = str(target_verse_name).strip().lower()

    # 1) Source SF pretraining on shared navigation dynamics.
    src_verse = _build_verse(verse_name=src_name, seed=int(seed), params=source_params)
    probe = src_verse.reset()
    probe_obs = probe.obs if isinstance(probe.obs, dict) else {}
    probe_ego = adapter.extract(verse_name=src_name, verse=src_verse, obs=probe_obs)
    feat_dim = int(adapter.phi(probe_ego).shape[0])
    src_n_actions = int(getattr(getattr(src_verse, "action_space", None), "n", 0) or 0)
    if src_n_actions <= 0:
        raise ValueError(f"Unsupported source action space for SF transfer: {src_name}")
    src_allowed = (
        [int(a) for a in source_allowed_actions if 0 <= int(a) < src_n_actions]
        if source_allowed_actions is not None
        else list(range(src_n_actions))
    )
    if not src_allowed:
        src_allowed = list(range(src_n_actions))

    src_agent = TabularSFAgent(
        n_actions=src_n_actions,
        feature_dim=feat_dim,
        gamma=0.97,
        psi_lr=0.24,
        w_lr=0.06,
        fwd_lr=0.02,
        allowed_actions=src_allowed,
    )
    src_agent.set_w(_semantic_reward_weights(feature_dim=feat_dim, grid_size=adapter.size))
    src_train_stats: List[EpisodeStats] = []
    src_policy_bank: List[TabularSFAgent] = []
    num_snaps = max(1, int(source_policy_snapshots))
    snap_eps = set()
    for i in range(1, num_snaps + 1):
        frac_ep = int(round(float(i) * float(max(1, source_train_episodes)) / float(num_snaps)))
        snap_eps.add(max(1, min(int(source_train_episodes), frac_ep)))
    for ep in range(int(source_train_episodes)):
        eps = _epsilon_linear(ep, source_train_episodes, start=0.40, end=0.05)
        src_train_stats.append(
            _run_episode(
                verse=src_verse,
                verse_name=src_name,
                adapter=adapter,
                agent=src_agent,
                max_steps=max_steps,
                train=True,
                epsilon=eps,
                rng=rng,
                learn_psi=True,
                learn_w=True,
            )
        )
        ep1 = int(ep + 1)
        if ep1 in snap_eps:
            src_policy_bank.append(src_agent.clone())
    if not src_policy_bank:
        src_policy_bank.append(src_agent.clone())
    probe_rng = random.Random((int(seed) * 1000003) ^ 0x5A17)
    source_probe_egos = _collect_probe_egos(
        verse_name=src_name,
        verse=src_verse,
        adapter=adapter,
        max_steps=max_steps,
        rng=probe_rng,
        probe_count=24,
    )
    source_policy_bank_agreement = _policy_bank_agreement_diag(
        snapshots=src_policy_bank,
        probes=source_probe_egos,
        adapter=adapter,
    )
    src_verse.close()
    source_summary = _summarize(src_train_stats)
    source_early_k = max(1, min(10, len(src_train_stats)))
    source_early_trace = _episode_trace(src_train_stats, max_points=source_early_k)
    source_early_diagnostics = _trace_diagnostics(source_early_trace)

    # 2) Target conditions.
    def _make_target_agent(*, transferred: bool, freeze_psi_episodes: int) -> Dict[str, Any]:
        trg_verse = _build_verse(verse_name=trg_name, seed=int(seed), params=target_params)
        trg_n_actions = int(getattr(getattr(trg_verse, "action_space", None), "n", 0) or 0)
        if trg_n_actions <= 0:
            raise ValueError(f"Unsupported target action space for SF transfer: {trg_name}")
        trg_allowed = (
            [int(a) for a in target_allowed_actions if 0 <= int(a) < trg_n_actions]
            if target_allowed_actions is not None
            else list(range(trg_n_actions))
        )
        if not trg_allowed:
            trg_allowed = list(range(trg_n_actions))
        ag = TabularSFAgent(
            n_actions=trg_n_actions,
            feature_dim=feat_dim,
            gamma=0.97,
            psi_lr=0.20,
            w_lr=0.07,
            fwd_lr=0.02,
            allowed_actions=trg_allowed,
        )
        if transferred:
            ag.copy_psi_from(src_agent)
            ag.set_w(_semantic_reward_weights(feature_dim=feat_dim, grid_size=adapter.size))
            projected_banks = [
                _project_psi_table_to_actions(
                    src_table=snap.psi_table,
                    src_n_actions=int(snap.n_actions),
                    dst_n_actions=int(trg_n_actions),
                    feature_dim=int(feat_dim),
                )
                for snap in src_policy_bank
            ]
            ag.set_gpi_banks(projected_banks)

        # Zero-shot snapshot before any target updates.
        zero_shot: Optional[Dict[str, Any]] = None
        w_est_diag: Optional[Dict[str, Any]] = None
        if transferred and int(target_w_estimation_steps) > 0:
            w_est_diag = _estimate_target_reward_weights(
                verse=trg_verse,
                verse_name=trg_name,
                adapter=adapter,
                agent=ag,
                steps_budget=int(target_w_estimation_steps),
                max_steps_per_episode=max_steps,
                rng=rng,
                epsilon=0.40,
                ridge_l2=1.0,
            )
        if transferred:
            eval_stats = []
            for _ in range(int(eval_episodes)):
                eval_stats.append(
                    _run_episode(
                        verse=trg_verse,
                        verse_name=trg_name,
                        adapter=adapter,
                        agent=ag,
                        max_steps=max_steps,
                        train=False,
                        epsilon=0.0,
                        rng=rng,
                    )
                )
            zero_shot = _summarize(eval_stats)

        train_stats: List[EpisodeStats] = []
        for ep in range(int(target_train_episodes)):
            eps = _epsilon_linear(ep, target_train_episodes, start=0.35, end=0.05)
            learn_psi_flag = bool(ep >= int(freeze_psi_episodes))
            train_stats.append(
                _run_episode(
                    verse=trg_verse,
                    verse_name=trg_name,
                    adapter=adapter,
                    agent=ag,
                    max_steps=max_steps,
                    train=True,
                    epsilon=eps,
                    rng=rng,
                    learn_psi=learn_psi_flag,
                    learn_w=True,
                )
            )

        eval_stats_post = []
        for _ in range(int(eval_episodes)):
            eval_stats_post.append(
                _run_episode(
                    verse=trg_verse,
                    verse_name=trg_name,
                    adapter=adapter,
                    agent=ag,
                    max_steps=max_steps,
                    train=False,
                    epsilon=0.0,
                    rng=rng,
                )
            )
        trg_verse.close()

        early_k = max(1, min(10, len(train_stats)))
        early_summary = _summarize(train_stats[:early_k])
        early_trace = _episode_trace(train_stats, max_points=early_k)
        early_diagnostics = _trace_diagnostics(early_trace)
        return {
            "zero_shot_eval": zero_shot,
            "w_estimation": w_est_diag,
            "train_summary": _summarize(train_stats),
            "early_train_summary": early_summary,
            "early_train_trace": early_trace,
            "early_train_diagnostics": early_diagnostics,
            "eval_summary": _summarize(eval_stats_post),
        }

    scratch = _make_target_agent(transferred=False, freeze_psi_episodes=0)
    transfer = _make_target_agent(transferred=True, freeze_psi_episodes=0)
    transfer_warmup = _make_target_agent(
        transferred=True,
        freeze_psi_episodes=max(0, int(warmup_psi_episodes)),
    )
    # Short canary trajectory deltas for adaptive gating/diagnostics.
    canary_transfer_minus_scratch = _trace_delta(
        transfer.get("early_train_trace", {}) if isinstance(transfer, dict) else {},
        scratch.get("early_train_trace", {}) if isinstance(scratch, dict) else {},
        max_points=10,
    )
    canary_transfer_warmup_minus_scratch = _trace_delta(
        transfer_warmup.get("early_train_trace", {}) if isinstance(transfer_warmup, dict) else {},
        scratch.get("early_train_trace", {}) if isinstance(scratch, dict) else {},
        max_points=10,
    )
    if isinstance(transfer, dict):
        transfer["canary_vs_scratch_early"] = canary_transfer_minus_scratch
    if isinstance(transfer_warmup, dict):
        transfer_warmup["canary_vs_scratch_early"] = canary_transfer_warmup_minus_scratch

    return {
        "seed": int(seed),
        "source_verse_name": src_name,
        "target_verse_name": trg_name,
        "source_pretrain": source_summary,
        "source_early_train_trace": source_early_trace,
        "source_early_train_diagnostics": source_early_diagnostics,
        "source_policy_bank_agreement": source_policy_bank_agreement,
        "source_policy_bank_size": int(len(src_policy_bank)),
        "target_w_estimation_steps": int(max(0, int(target_w_estimation_steps))),
        "target_conditions": {
            "sf_scratch": scratch,
            "sf_transfer": transfer,
            "sf_transfer_warmup": transfer_warmup,
        },
    }


def _adaptive_gate_cfg(
    *,
    enabled: bool,
    scratch_early_success_max: float,
    scratch_early_return_max: float,
    scratch_early_hazard_min: float,
    scratch_eval_success_max: float,
    scratch_eval_return_max: float,
    scratch_eval_hazard_min: float,
    transfer_gate_condition_key: str,
    transfer_early_success_min: float,
    transfer_early_return_min: float,
    transfer_early_hazard_max: float,
    transfer_early_forward_mse_max: float,
    transfer_minus_scratch_early_success_min: float,
    transfer_minus_scratch_early_return_min: float,
    scratch_minus_transfer_early_hazard_max: float,
    scratch_minus_transfer_early_forward_mse_max: float,
    transfer_early_return_slope_min: float,
    transfer_early_success_slope_min: float,
    transfer_early_hazard_slope_max: float,
    transfer_early_forward_mse_slope_max: float,
    canary_delta_return_slope_min: float,
    canary_delta_success_slope_min: float,
    canary_delta_hazard_slope_max: float,
    canary_delta_forward_mse_slope_max: float,
    source_policy_bank_majority_min: float,
    logic: str,
) -> Dict[str, Any]:
    logic_n = str(logic or "any").strip().lower()
    if logic_n not in {"any", "all"}:
        logic_n = "any"
    return {
        "enabled": bool(enabled),
        "kind": "hybrid_hardness_quality_gate",
        "logic": logic_n,
        "transfer_gate_condition_key": str(transfer_gate_condition_key or "sf_transfer_warmup"),
        "thresholds": {
            "scratch_early_success_max": float(scratch_early_success_max),
            "scratch_early_return_max": float(scratch_early_return_max),
            "scratch_early_hazard_min": float(scratch_early_hazard_min),
            "scratch_eval_success_max": float(scratch_eval_success_max),
            "scratch_eval_return_max": float(scratch_eval_return_max),
            "scratch_eval_hazard_min": float(scratch_eval_hazard_min),
            "transfer_early_success_min": float(transfer_early_success_min),
            "transfer_early_return_min": float(transfer_early_return_min),
            "transfer_early_hazard_max": float(transfer_early_hazard_max),
            "transfer_early_forward_mse_max": float(transfer_early_forward_mse_max),
            "transfer_minus_scratch_early_success_min": float(transfer_minus_scratch_early_success_min),
            "transfer_minus_scratch_early_return_min": float(transfer_minus_scratch_early_return_min),
            "scratch_minus_transfer_early_hazard_max": float(scratch_minus_transfer_early_hazard_max),
            "scratch_minus_transfer_early_forward_mse_max": float(scratch_minus_transfer_early_forward_mse_max),
            "transfer_early_return_slope_min": float(transfer_early_return_slope_min),
            "transfer_early_success_slope_min": float(transfer_early_success_slope_min),
            "transfer_early_hazard_slope_max": float(transfer_early_hazard_slope_max),
            "transfer_early_forward_mse_slope_max": float(transfer_early_forward_mse_slope_max),
            "canary_delta_return_slope_min": float(canary_delta_return_slope_min),
            "canary_delta_success_slope_min": float(canary_delta_success_slope_min),
            "canary_delta_hazard_slope_max": float(canary_delta_hazard_slope_max),
            "canary_delta_forward_mse_slope_max": float(canary_delta_forward_mse_slope_max),
            "source_policy_bank_majority_min": float(source_policy_bank_majority_min),
        },
    }


def _adaptive_gate_model_cfg(
    *,
    enabled: bool,
    transfer_gate_condition_key: str,
    model_payload: Dict[str, Any],
    accept_prob_min: float,
    warmup_prob_min: float,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "kind": "learned_gate_model",
        "logic": "probability_policy",
        "transfer_gate_condition_key": str(transfer_gate_condition_key or "sf_transfer_warmup"),
        "thresholds": {},
        "model": dict(model_payload) if isinstance(model_payload, dict) else {},
        "model_policy": {
            "accept_prob_min": float(accept_prob_min),
            "warmup_prob_min": float(warmup_prob_min),
            "enable_triage": False,
        },
    }


def _adaptive_gate_model_features(
    row: Dict[str, Any],
    *,
    transfer_gate_condition_key: str,
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer_cond = tc.get(str(transfer_gate_condition_key), {})
    if not isinstance(transfer_cond, dict):
        transfer_cond = {}
    early = scratch.get("early_train_summary", {}) if isinstance(scratch.get("early_train_summary"), dict) else {}
    eval_sum = scratch.get("eval_summary", {}) if isinstance(scratch.get("eval_summary"), dict) else {}
    transfer_early = (
        transfer_cond.get("early_train_summary", {})
        if isinstance(transfer_cond.get("early_train_summary"), dict)
        else {}
    )
    transfer_early_diag = (
        transfer_cond.get("early_train_diagnostics", {})
        if isinstance(transfer_cond.get("early_train_diagnostics"), dict)
        else {}
    )
    canary_delta = (
        (transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics", {})
        if isinstance(transfer_cond.get("canary_vs_scratch_early"), dict)
        and isinstance((transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics"), dict)
        else {}
    )
    src_bank_agreement = (
        row.get("source_policy_bank_agreement", {})
        if isinstance(row.get("source_policy_bank_agreement"), dict)
        else {}
    )
    early_sr = _safe_float(early.get("success_rate", 0.0), 0.0)
    early_ret = _safe_float(early.get("mean_return", 0.0), 0.0)
    early_haz = _safe_float(early.get("hazard_per_1k", 0.0), 0.0)
    eval_sr = _safe_float(eval_sum.get("success_rate", 0.0), 0.0)
    eval_ret = _safe_float(eval_sum.get("mean_return", 0.0), 0.0)
    eval_haz = _safe_float(eval_sum.get("hazard_per_1k", 0.0), 0.0)
    transfer_early_sr = _safe_float(transfer_early.get("success_rate", 0.0), 0.0)
    transfer_early_ret = _safe_float(transfer_early.get("mean_return", 0.0), 0.0)
    transfer_early_haz = _safe_float(transfer_early.get("hazard_per_1k", 0.0), 0.0)
    scratch_early_fwd = _safe_float(early.get("mean_forward_mse", 0.0), 0.0)
    transfer_early_fwd = _safe_float(transfer_early.get("mean_forward_mse", 0.0), 0.0)
    d_early_success = float(transfer_early_sr - early_sr)
    d_early_return = float(transfer_early_ret - early_ret)
    d_early_hazard_gain = float(early_haz - transfer_early_haz)
    d_early_forward_mse_gain = float(scratch_early_fwd - transfer_early_fwd)
    transfer_early_return_slope = _safe_float(transfer_early_diag.get("return_slope", 0.0), 0.0)
    transfer_early_success_slope = _safe_float(transfer_early_diag.get("success_slope", 0.0), 0.0)
    transfer_early_hazard_slope = _safe_float(transfer_early_diag.get("hazard_slope", 0.0), 0.0)
    transfer_early_forward_mse_slope = _safe_float(transfer_early_diag.get("forward_mse_slope", 0.0), 0.0)
    canary_return_slope = _safe_float(canary_delta.get("return_slope", 0.0), 0.0)
    canary_success_slope = _safe_float(canary_delta.get("success_slope", 0.0), 0.0)
    canary_hazard_slope = _safe_float(canary_delta.get("hazard_slope", 0.0), 0.0)
    canary_forward_mse_slope = _safe_float(canary_delta.get("forward_mse_slope", 0.0), 0.0)
    source_bank_majority = _safe_float(src_bank_agreement.get("mean_majority_fraction", 0.0), 0.0)
    source_bank_vote_margin = _safe_float(src_bank_agreement.get("mean_vote_margin", 0.0), 0.0)
    source_bank_unique_actions = _safe_float(src_bank_agreement.get("mean_unique_actions", 0.0), 0.0)
    features = {
        "scratch_early_success_rate": float(early_sr),
        "scratch_early_return": float(early_ret),
        "scratch_early_hazard_per_1k": float(early_haz),
        "scratch_early_forward_mse": float(scratch_early_fwd),
        "scratch_eval_success_rate": float(eval_sr),
        "scratch_eval_return": float(eval_ret),
        "scratch_eval_hazard_per_1k": float(eval_haz),
        "transfer_early_success_rate": float(transfer_early_sr),
        "transfer_early_return": float(transfer_early_ret),
        "transfer_early_hazard_per_1k": float(transfer_early_haz),
        "transfer_early_forward_mse": float(transfer_early_fwd),
        "transfer_minus_scratch_early_success_rate": float(d_early_success),
        "transfer_minus_scratch_early_return": float(d_early_return),
        "scratch_minus_transfer_early_hazard_per_1k": float(d_early_hazard_gain),
        "scratch_minus_transfer_early_forward_mse": float(d_early_forward_mse_gain),
        "transfer_early_return_slope": float(transfer_early_return_slope),
        "transfer_early_success_slope": float(transfer_early_success_slope),
        "transfer_early_hazard_slope": float(transfer_early_hazard_slope),
        "transfer_early_forward_mse_slope": float(transfer_early_forward_mse_slope),
        "canary_delta_return_slope": float(canary_return_slope),
        "canary_delta_success_slope": float(canary_success_slope),
        "canary_delta_hazard_slope": float(canary_hazard_slope),
        "canary_delta_forward_mse_slope": float(canary_forward_mse_slope),
        "source_policy_bank_majority_fraction": float(source_bank_majority),
        "source_policy_bank_vote_margin": float(source_bank_vote_margin),
        "source_policy_bank_unique_actions": float(source_bank_unique_actions),
    }
    return {
        "features": features,
        "scratch_early": {
            "success_rate": float(early_sr),
            "mean_return": float(early_ret),
            "hazard_per_1k": float(early_haz),
        },
        "scratch_eval": {
            "success_rate": float(eval_sr),
            "mean_return": float(eval_ret),
            "hazard_per_1k": float(eval_haz),
        },
        "transfer_early": {
            "success_rate": float(transfer_early_sr),
            "mean_return": float(transfer_early_ret),
            "hazard_per_1k": float(transfer_early_haz),
            "mean_forward_mse": float(transfer_early_fwd),
        },
        "canary_early_deltas": {
            "transfer_minus_scratch_success_rate": float(d_early_success),
            "transfer_minus_scratch_mean_return": float(d_early_return),
            "scratch_minus_transfer_hazard_per_1k": float(d_early_hazard_gain),
            "scratch_minus_transfer_forward_mse": float(d_early_forward_mse_gain),
        },
        "transfer_early_trends": {
            "return_slope": float(transfer_early_return_slope),
            "success_slope": float(transfer_early_success_slope),
            "hazard_slope": float(transfer_early_hazard_slope),
            "forward_mse_slope": float(transfer_early_forward_mse_slope),
        },
        "canary_early_delta_trends": {
            "return_slope": float(canary_return_slope),
            "success_slope": float(canary_success_slope),
            "hazard_slope": float(canary_hazard_slope),
            "forward_mse_slope": float(canary_forward_mse_slope),
        },
        "source_policy_bank_agreement": {
            "mean_majority_fraction": float(source_bank_majority),
            "mean_unique_actions": float(source_bank_unique_actions),
            "mean_vote_margin": float(source_bank_vote_margin),
            "num_snapshots": int(src_bank_agreement.get("num_snapshots", 0) or 0),
            "evaluated_probes": int(src_bank_agreement.get("evaluated_probes", 0) or 0),
        },
    }


def _sigmoid(x: float) -> float:
    z = max(-60.0, min(60.0, float(x)))
    return float(1.0 / (1.0 + math.exp(-z)))


def _score_learned_linear_model(
    *,
    features: Dict[str, Any],
    model_block: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(model_block, dict):
        return {"probability": 0.5, "logit_score": 0.0, "feature_names": [], "feature_count": 0}
    feature_names = [str(x) for x in model_block.get("feature_names", [])]
    weights = [float(x) for x in model_block.get("weights", [])]
    bias = _safe_float(model_block.get("bias", 0.0), 0.0)
    norm = model_block.get("normalization", {}) if isinstance(model_block.get("normalization"), dict) else {}
    means = [float(x) for x in norm.get("mean", [])] if isinstance(norm.get("mean"), list) else []
    scales = [float(x) for x in norm.get("scale", [])] if isinstance(norm.get("scale"), list) else []
    z = float(bias)
    used_features: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        x = _safe_float(features.get(name, 0.0), 0.0)
        mu = means[i] if i < len(means) else 0.0
        sc = scales[i] if i < len(scales) and abs(scales[i]) > 1e-12 else 1.0
        xn = float((x - mu) / sc)
        w = weights[i] if i < len(weights) else 0.0
        z += float(w * xn)
        used_features[name] = float(x)
    return {
        "probability": float(_sigmoid(z)),
        "logit_score": float(z),
        "feature_names": feature_names,
        "feature_count": int(len(feature_names)),
        "used_features": used_features,
    }


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(float(x) for x in scores)
    exps = [math.exp(max(-60.0, min(60.0, float(x) - m))) for x in scores]
    s = sum(exps)
    if s <= 0.0:
        return [1.0 / float(len(scores)) for _ in scores]
    return [float(e / s) for e in exps]


def _score_learned_softmax_model(
    *,
    features: Dict[str, Any],
    model_block: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(model_block, dict):
        return {"class_names": ["sf_scratch", "sf_transfer", "sf_transfer_warmup"], "probs": [1 / 3, 1 / 3, 1 / 3]}
    feature_names = [str(x) for x in model_block.get("feature_names", [])]
    class_names = [str(x) for x in model_block.get("class_names", ["sf_scratch", "sf_transfer", "sf_transfer_warmup"])]
    weights = np.asarray(model_block.get("weights", []), dtype=np.float64)
    bias = np.asarray(model_block.get("bias", []), dtype=np.float64)
    norm = model_block.get("normalization", {}) if isinstance(model_block.get("normalization"), dict) else {}
    means = np.asarray(norm.get("mean", []), dtype=np.float64) if isinstance(norm.get("mean"), list) else np.zeros((0,), dtype=np.float64)
    scales = np.asarray(norm.get("scale", []), dtype=np.float64) if isinstance(norm.get("scale"), list) else np.ones((0,), dtype=np.float64)
    x_raw = np.asarray([_safe_float(features.get(name, 0.0), 0.0) for name in feature_names], dtype=np.float64)
    if means.size != x_raw.size:
        means = np.zeros_like(x_raw)
    if scales.size != x_raw.size:
        scales = np.ones_like(x_raw)
    scales = np.where(np.abs(scales) < 1e-8, 1.0, scales)
    xn = (x_raw - means) / scales
    if weights.ndim != 2 or weights.shape[0] != x_raw.size:
        k = max(1, int(bias.size) if bias.size else len(class_names))
        scores = [0.0] * k
    else:
        if bias.size != weights.shape[1]:
            bias = np.zeros((weights.shape[1],), dtype=np.float64)
        scores = (xn @ weights + bias).astype(np.float64).tolist()
    probs = _softmax([float(s) for s in scores])
    used_features = {str(name): float(x_raw[i]) for i, name in enumerate(feature_names)}
    return {
        "class_names": class_names,
        "scores": [float(s) for s in scores],
        "probs": [float(p) for p in probs],
        "feature_names": feature_names,
        "feature_count": int(len(feature_names)),
        "used_features": used_features,
    }


def _adaptive_triage_direct_features(row: Dict[str, Any]) -> Dict[str, float]:
    f_full = _adaptive_gate_model_features(row, transfer_gate_condition_key="sf_transfer")
    f_warm = _adaptive_gate_model_features(row, transfer_gate_condition_key="sf_transfer_warmup")
    ff = f_full.get("features", {}) if isinstance(f_full.get("features"), dict) else {}
    fw = f_warm.get("features", {}) if isinstance(f_warm.get("features"), dict) else {}
    out: Dict[str, float] = {}
    for k, v in ff.items():
        out[f"full::{k}"] = _safe_float(v, 0.0)
    for k, v in fw.items():
        out[f"warm::{k}"] = _safe_float(v, 0.0)
    # Pairwise preference signals
    for k in sorted(set(ff.keys()) | set(fw.keys())):
        out[f"full_minus_warm::{k}"] = float(_safe_float(ff.get(k, 0.0), 0.0) - _safe_float(fw.get(k, 0.0), 0.0))
    return out


def _triage_canary_probe_summary(row: Dict[str, Any], cond_key: str) -> Dict[str, float]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    cond = tc.get(str(cond_key), {}) if isinstance(tc.get(str(cond_key)), dict) else {}
    cvs = cond.get("canary_vs_scratch_early", {}) if isinstance(cond.get("canary_vs_scratch_early"), dict) else {}
    diag = cvs.get("diagnostics", {}) if isinstance(cvs.get("diagnostics"), dict) else {}
    # hazard_mean is transfer-minus-scratch hazard delta (positive = worse hazard); convert to gain.
    hazard_delta = _safe_float(diag.get("hazard_mean", 0.0), 0.0)
    hazard_gain = float(-hazard_delta)
    return {
        "episodes": _safe_float(diag.get("episodes", 0.0), 0.0),
        "success_delta_mean": _safe_float(diag.get("success_mean", 0.0), 0.0),
        "return_delta_mean": _safe_float(diag.get("return_mean", 0.0), 0.0),
        "hazard_delta_mean": float(hazard_delta),
        "hazard_gain_mean": float(hazard_gain),
        "success_delta_slope": _safe_float(diag.get("success_slope", 0.0), 0.0),
        "return_delta_slope": _safe_float(diag.get("return_slope", 0.0), 0.0),
        "hazard_delta_slope": _safe_float(diag.get("hazard_slope", 0.0), 0.0),
    }


def _canary_triad_override(
    *,
    row: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    selected: str,
    decision_reason: str,
) -> Tuple[str, str, Dict[str, Any]]:
    if not bool(runtime_policy.get("enable_canary_triad_override", False)):
        return str(selected), str(decision_reason), {"enabled": False}
    full = _triage_canary_probe_summary(row, "sf_transfer")
    warm = _triage_canary_probe_summary(row, "sf_transfer_warmup")
    base_policy = {
        "name": "competence",
        "success_weight": _safe_float(runtime_policy.get("canary_success_weight", 100.0), 100.0),
        "return_weight": _safe_float(runtime_policy.get("canary_return_weight", 1.0), 1.0),
        "hazard_weight": _safe_float(runtime_policy.get("canary_hazard_weight", 0.02), 0.02),
        "min_utility": _safe_float(runtime_policy.get("canary_min_utility", 0.0), 0.0),
        "min_hazard_gain": _safe_float(runtime_policy.get("canary_min_hazard_gain", -1e9), -1e9),
        "min_episodes": _safe_float(runtime_policy.get("canary_min_episodes", 1.0), 1.0),
    }
    safety_policy = {
        "name": "safety",
        "success_weight": _safe_float(runtime_policy.get("canary_safety_success_weight", base_policy["success_weight"]), base_policy["success_weight"]),
        "return_weight": _safe_float(runtime_policy.get("canary_safety_return_weight", base_policy["return_weight"]), base_policy["return_weight"]),
        "hazard_weight": _safe_float(runtime_policy.get("canary_safety_hazard_weight", max(base_policy["hazard_weight"], 0.05)), max(base_policy["hazard_weight"], 0.05)),
        "min_utility": _safe_float(runtime_policy.get("canary_safety_min_utility", base_policy["min_utility"]), base_policy["min_utility"]),
        "min_hazard_gain": _safe_float(runtime_policy.get("canary_safety_min_hazard_gain", max(0.0, base_policy["min_hazard_gain"])), max(0.0, base_policy["min_hazard_gain"])),
        "min_episodes": _safe_float(runtime_policy.get("canary_safety_min_episodes", base_policy["min_episodes"]), base_policy["min_episodes"]),
    }

    dual_enabled = bool(runtime_policy.get("enable_canary_dual_policy", False))
    dual_hg_split = _safe_float(runtime_policy.get("canary_dual_select_hazard_gain_threshold", 0.0), 0.0)
    dual_sd_split = _safe_float(runtime_policy.get("canary_dual_select_success_delta_threshold", -1e9), -1e9)
    max_probe_hg = max(_safe_float(full.get("hazard_gain_mean", 0.0), 0.0), _safe_float(warm.get("hazard_gain_mean", 0.0), 0.0))
    max_probe_sd = max(_safe_float(full.get("success_delta_mean", 0.0), 0.0), _safe_float(warm.get("success_delta_mean", 0.0), 0.0))
    if dual_enabled and (max_probe_hg < dual_hg_split or max_probe_sd < dual_sd_split):
        active_policy = safety_policy
        policy_reason = "dual_policy_safety"
    else:
        active_policy = base_policy
        policy_reason = "dual_policy_competence" if dual_enabled else "single_policy"

    sw = float(active_policy["success_weight"])
    rw = float(active_policy["return_weight"])
    hw = float(active_policy["hazard_weight"])
    min_u = float(active_policy["min_utility"])
    min_hg = float(active_policy["min_hazard_gain"])
    min_eps = float(active_policy["min_episodes"])

    def _utility(c: Dict[str, float]) -> float:
        return float(
            sw * _safe_float(c.get("success_delta_mean", 0.0), 0.0)
            + rw * _safe_float(c.get("return_delta_mean", 0.0), 0.0)
            + hw * _safe_float(c.get("hazard_gain_mean", 0.0), 0.0)
        )

    full_u = _utility(full)
    warm_u = _utility(warm)
    full_pass = bool(
        _safe_float(full.get("episodes", 0.0), 0.0) >= min_eps
        and full_u >= min_u
        and _safe_float(full.get("hazard_gain_mean", 0.0), 0.0) >= min_hg
    )
    warm_pass = bool(
        _safe_float(warm.get("episodes", 0.0), 0.0) >= min_eps
        and warm_u >= min_u
        and _safe_float(warm.get("hazard_gain_mean", 0.0), 0.0) >= min_hg
    )
    # Optional veto: require minimum canary success lift for full transfer,
    # unless hazard gain is very strong (safety override).
    full_success_floor = _safe_float(runtime_policy.get("canary_full_success_floor", -1e9), -1e9)
    full_hazard_override = _safe_float(runtime_policy.get("canary_full_hazard_gain_override", 25.0), 25.0)
    full_success_floor_applied = False
    full_success_floor_pass = True
    if float(full_success_floor) > -1e8:
        full_success_floor_applied = True
        full_success_floor_pass = bool(
            _safe_float(full.get("success_delta_mean", 0.0), 0.0) >= float(full_success_floor)
            or _safe_float(full.get("hazard_gain_mean", 0.0), 0.0) >= float(full_hazard_override)
        )
        if not full_success_floor_pass:
            full_pass = False
    prior = str(selected)
    reason = str(decision_reason)
    if bool(runtime_policy.get("canary_two_stage_global_select", False)):
        # Stage A: hard veto already encoded in full_pass/warm_pass.
        # Stage B: rank surviving modes by canary utility, with scratch utility baseline = 0.
        scratch_u = _safe_float(runtime_policy.get("canary_scratch_utility", 0.0), 0.0)
        candidates: List[Tuple[str, float]] = [("sf_scratch", float(scratch_u))]
        if full_pass:
            candidates.append(("sf_transfer", float(full_u)))
        if warm_pass:
            candidates.append(("sf_transfer_warmup", float(warm_u)))
        # Tie-break toward prior if utilities match closely.
        best_sel = "sf_scratch"
        best_u = float("-inf")
        for name, u in candidates:
            if (u > best_u) or (abs(u - best_u) <= 1e-12 and name == prior):
                best_sel = str(name)
                best_u = float(u)
        selected = str(best_sel)
        if selected == prior:
            reason = "canary_two_stage_keep_prior"
        elif selected == "sf_scratch":
            reason = "canary_two_stage_global_to_scratch"
        elif selected == "sf_transfer":
            reason = "canary_two_stage_global_to_full"
        else:
            reason = "canary_two_stage_global_to_warmup"
    elif prior == "sf_transfer" and not full_pass:
        if warm_pass and warm_u > full_u:
            selected = "sf_transfer_warmup"
            reason = "canary_override_full_to_warmup"
        else:
            selected = "sf_scratch"
            reason = "canary_override_full_to_scratch"
    elif prior == "sf_transfer_warmup" and not warm_pass:
        if full_pass and full_u > warm_u:
            selected = "sf_transfer"
            reason = "canary_override_warmup_to_full"
        else:
            selected = "sf_scratch"
            reason = "canary_override_warmup_to_scratch"
    elif prior == "sf_scratch":
        if full_pass or warm_pass:
            if full_pass and (not warm_pass or full_u >= warm_u):
                selected = "sf_transfer"
                reason = "canary_override_scratch_to_full"
            elif warm_pass:
                selected = "sf_transfer_warmup"
                reason = "canary_override_scratch_to_warmup"
    return str(selected), str(reason), {
        "enabled": True,
        "prior_selected_condition": str(prior),
        "prior_decision_reason": str(decision_reason),
        "weights": {
            "success_weight": float(sw),
            "return_weight": float(rw),
            "hazard_gain_weight": float(hw),
        },
        "active_policy": str(active_policy.get("name", "competence")),
        "policy_choice_reason": str(policy_reason),
        "dual_policy": {
            "enabled": bool(dual_enabled),
            "hazard_gain_threshold": float(dual_hg_split),
            "success_delta_threshold": float(dual_sd_split),
            "max_probe_hazard_gain": float(max_probe_hg),
            "max_probe_success_delta": float(max_probe_sd),
            "competence_policy": dict(base_policy),
            "safety_policy": dict(safety_policy),
        },
        "thresholds": {
            "min_utility": float(min_u),
            "min_hazard_gain": float(min_hg),
            "min_episodes": float(min_eps),
            "two_stage_global_select": bool(runtime_policy.get("canary_two_stage_global_select", False)),
            "scratch_utility": float(_safe_float(runtime_policy.get("canary_scratch_utility", 0.0), 0.0)),
            "full_success_floor": float(full_success_floor),
            "full_hazard_gain_override": float(full_hazard_override),
        },
        "full_probe": {
            **full,
            "utility": float(full_u),
            "pass": bool(full_pass),
            "success_floor_applied": bool(full_success_floor_applied),
            "success_floor_pass": bool(full_success_floor_pass),
        },
        "warmup_probe": {**warm, "utility": float(warm_u), "pass": bool(warm_pass)},
        "overrode": bool(str(prior) != str(selected)),
    }


def _adaptive_gate_model_decision(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool((cfg or {}).get("enabled", False))
    transfer_gate_condition_key = str((cfg or {}).get("transfer_gate_condition_key", "sf_transfer_warmup") or "sf_transfer_warmup")
    payload = (cfg or {}).get("model", {})
    if not isinstance(payload, dict):
        payload = {}
    condition_models = payload.get("condition_models", {})
    model_block: Dict[str, Any] = {}
    if isinstance(condition_models, dict) and isinstance(condition_models.get(transfer_gate_condition_key), dict):
        model_block = dict(condition_models.get(transfer_gate_condition_key) or {})
    elif isinstance(payload.get("model"), dict):
        model_block = dict(payload.get("model") or {})
    elif isinstance(payload.get("default_model"), dict):
        model_block = dict(payload.get("default_model") or {})
    feature_payload = _adaptive_gate_model_features(row, transfer_gate_condition_key=transfer_gate_condition_key)
    feats = feature_payload.get("features", {}) if isinstance(feature_payload.get("features"), dict) else {}

    model_gate_details: Dict[str, Any] = {}
    if not enabled:
        accept = True
        reason = "adaptive_disabled"
        prob_help = 1.0
        z = 0.0
    elif not model_block:
        accept = True
        reason = "missing_learned_model"
        prob_help = 1.0
        z = 0.0
    else:
        effect_score = _score_learned_linear_model(features=feats, model_block=model_block)
        prob_help = _safe_float(effect_score.get("probability", 0.5), 0.5)
        z = _safe_float(effect_score.get("logit_score", 0.0), 0.0)
        policy_cfg = (cfg or {}).get("model_policy", {})
        payload_policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
        model_policy = model_block.get("policy", {}) if isinstance(model_block.get("policy"), dict) else {}
        accept_prob_min = _safe_float(
            (policy_cfg.get("accept_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("accept_prob_min", payload_policy.get("accept_prob_min", 0.5)), 0.5),
        )
        warmup_prob_min = _safe_float(
            (policy_cfg.get("warmup_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("warmup_prob_min", payload_policy.get("warmup_prob_min", 0.35)), 0.35),
        )
        hard_model_block = (
            model_block.get("hardness_aux_model", {})
            if isinstance(model_block.get("hardness_aux_model"), dict)
            else {}
        )
        prob_hard = None
        hard_z = None
        hard_prob_min = _safe_float(
            (policy_cfg.get("hard_prob_min") if isinstance(policy_cfg, dict) else None),
            _safe_float(model_policy.get("hard_prob_min", payload_policy.get("hard_prob_min", -1.0)), -1.0),
        )
        hardness_pass = True
        if isinstance(hard_model_block, dict) and hard_model_block:
            hard_score = _score_learned_linear_model(features=feats, model_block=hard_model_block)
            prob_hard = _safe_float(hard_score.get("probability", 0.5), 0.5)
            hard_z = _safe_float(hard_score.get("logit_score", 0.0), 0.0)
            if hard_prob_min >= 0.0:
                hardness_pass = bool(prob_hard >= hard_prob_min)
        model_gate_details = {
            "effect_model": {
                "probability_help": float(prob_help),
                "logit_score": float(z),
                "feature_names": list(effect_score.get("feature_names", [])),
                "feature_count": int(effect_score.get("feature_count", 0)),
            }
        }
        if prob_hard is not None:
            model_gate_details["hardness_model"] = {
                "probability_hard": float(prob_hard),
                "logit_score": float(hard_z if hard_z is not None else 0.0),
                "hard_prob_min": float(hard_prob_min),
                "pass": bool(hardness_pass),
                "label_definition": str(hard_model_block.get("label_definition", "scratch_eval_success_leq")),
            }
        if transfer_gate_condition_key == "sf_transfer_warmup":
            accept = bool(hardness_pass and (prob_help >= warmup_prob_min))
            reason = "learned_two_stage_warmup" if accept else "learned_two_stage_scratch"
            recommendation = "accept_transfer" if accept else "fallback_scratch"
        else:
            accept = bool(hardness_pass and (prob_help >= accept_prob_min))
            if accept:
                recommendation = "accept_transfer"
                reason = "learned_two_stage_transfer"
            elif hardness_pass and (prob_help >= warmup_prob_min):
                recommendation = "warmup_only_recommended"
                reason = "learned_two_stage_warmup_band"
            else:
                recommendation = "fallback_scratch"
                reason = "learned_two_stage_scratch"
    if "recommendation" not in locals():
        recommendation = "accept_transfer" if accept else "fallback_scratch"
    policy_cfg = (cfg or {}).get("model_policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    accept_prob_min = _safe_float(policy_cfg.get("accept_prob_min", 0.5), 0.5)
    warmup_prob_min = _safe_float(policy_cfg.get("warmup_prob_min", 0.35), 0.35)
    checks: Dict[str, Optional[bool]] = {
        "model_probability_above_accept": bool(prob_help >= accept_prob_min),
        "model_probability_above_warmup": bool(prob_help >= warmup_prob_min),
    }
    if isinstance(model_gate_details.get("hardness_model"), dict):
        checks["hardness_probability_above_threshold"] = bool(
            (model_gate_details.get("hardness_model") or {}).get("pass", True)
        )
    out = {
        "enabled": bool(enabled),
        "accept_transfer": bool(accept),
        "decision_reason": str(reason),
        "logic": "probability_policy",
        "thresholds": {},
        "scratch_early": dict(feature_payload.get("scratch_early", {})),
        "scratch_eval": dict(feature_payload.get("scratch_eval", {})),
        "transfer_gate_condition_key": transfer_gate_condition_key,
        "transfer_early": dict(feature_payload.get("transfer_early", {})),
        "canary_early_deltas": dict(feature_payload.get("canary_early_deltas", {})),
        "transfer_early_trends": dict(feature_payload.get("transfer_early_trends", {})),
        "canary_early_delta_trends": dict(feature_payload.get("canary_early_delta_trends", {})),
        "source_policy_bank_agreement": dict(feature_payload.get("source_policy_bank_agreement", {})),
        "checks": checks,
        "model_gate": {
            "model_schema_version": str(payload.get("schema_version", "")),
            "model_type": str(payload.get("model_type", "logistic_linear")),
            "transfer_gate_condition_key": transfer_gate_condition_key,
            "probability_help": float(prob_help),
            "logit_score": float(z),
            "recommendation": str(recommendation),
            "policy": {
                "accept_prob_min": float(accept_prob_min),
                "warmup_prob_min": float(warmup_prob_min),
                "hard_prob_min": float(
                    _safe_float(
                        (policy_cfg.get("hard_prob_min") if isinstance(policy_cfg, dict) else None),
                        _safe_float(
                            (model_block.get("policy", {}) if isinstance(model_block.get("policy"), dict) else {}).get(
                                "hard_prob_min",
                                (payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}).get(
                                    "hard_prob_min",
                                    -1.0,
                                ),
                            ),
                            -1.0,
                        ),
                    )
                ),
            },
        },
    }
    if model_gate_details:
        out["model_gate"].update(model_gate_details)
    if isinstance(model_block, dict):
        if "feature_names" not in out["model_gate"]:
            out["model_gate"]["feature_names"] = [str(x) for x in model_block.get("feature_names", [])]
            out["model_gate"]["feature_count"] = int(len(out["model_gate"]["feature_names"]))
    return out


def _seed_condition_summary(cond: Dict[str, Any]) -> Dict[str, Optional[float]]:
    eval_sum = cond.get("eval_summary", {}) if isinstance(cond.get("eval_summary"), dict) else {}
    early_sum = cond.get("early_train_summary", {}) if isinstance(cond.get("early_train_summary"), dict) else {}
    return {
        "eval_return": (
            None if not isinstance(eval_sum.get("mean_return"), (int, float)) else float(eval_sum.get("mean_return"))
        ),
        "eval_success_rate": (
            None
            if not isinstance(eval_sum.get("success_rate"), (int, float))
            else float(eval_sum.get("success_rate"))
        ),
        "eval_hazard_per_1k": (
            None
            if not isinstance(eval_sum.get("hazard_per_1k"), (int, float))
            else float(eval_sum.get("hazard_per_1k"))
        ),
        "eval_forward_mse": (
            None
            if not isinstance(eval_sum.get("mean_forward_mse"), (int, float))
            else float(eval_sum.get("mean_forward_mse"))
        ),
        "early_success_rate": (
            None
            if not isinstance(early_sum.get("success_rate"), (int, float))
            else float(early_sum.get("success_rate"))
        ),
        "early_return": (
            None if not isinstance(early_sum.get("mean_return"), (int, float)) else float(early_sum.get("mean_return"))
        ),
        "early_hazard_per_1k": (
            None
            if not isinstance(early_sum.get("hazard_per_1k"), (int, float))
            else float(early_sum.get("hazard_per_1k"))
        ),
        "early_forward_mse": (
            None
            if not isinstance(early_sum.get("mean_forward_mse"), (int, float))
            else float(early_sum.get("mean_forward_mse"))
        ),
    }


def _adaptive_gate_decision(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    transfer_gate_condition_key_override: Optional[str] = None,
) -> Dict[str, Any]:
    if transfer_gate_condition_key_override is not None:
        cfg_local = dict(cfg) if isinstance(cfg, dict) else {}
        cfg_local["transfer_gate_condition_key"] = str(transfer_gate_condition_key_override)
    else:
        cfg_local = cfg
    if str((cfg_local or {}).get("kind", "")).strip().lower() == "learned_gate_model":
        return _adaptive_gate_model_decision(row, cfg_local)
    enabled = bool((cfg_local or {}).get("enabled", False))
    thresholds = dict((cfg_local or {}).get("thresholds", {})) if isinstance((cfg_local or {}).get("thresholds", {}), dict) else {}
    logic = str((cfg_local or {}).get("logic", "any")).strip().lower()
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer_gate_condition_key = str((cfg_local or {}).get("transfer_gate_condition_key", "sf_transfer_warmup") or "sf_transfer_warmup")
    transfer_cond = tc.get(transfer_gate_condition_key, {}) if isinstance(tc.get(transfer_gate_condition_key), dict) else {}
    early = scratch.get("early_train_summary", {}) if isinstance(scratch.get("early_train_summary"), dict) else {}
    eval_sum = scratch.get("eval_summary", {}) if isinstance(scratch.get("eval_summary"), dict) else {}
    transfer_early = (
        transfer_cond.get("early_train_summary", {})
        if isinstance(transfer_cond.get("early_train_summary"), dict)
        else {}
    )
    transfer_early_diag = (
        transfer_cond.get("early_train_diagnostics", {})
        if isinstance(transfer_cond.get("early_train_diagnostics"), dict)
        else {}
    )
    canary_delta = (
        (transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics", {})
        if isinstance(transfer_cond.get("canary_vs_scratch_early"), dict)
        and isinstance((transfer_cond.get("canary_vs_scratch_early") or {}).get("diagnostics"), dict)
        else {}
    )
    src_bank_agreement = (
        row.get("source_policy_bank_agreement", {})
        if isinstance(row.get("source_policy_bank_agreement"), dict)
        else {}
    )
    early_sr = _safe_float(early.get("success_rate", 0.0), 0.0)
    early_ret = _safe_float(early.get("mean_return", 0.0), 0.0)
    early_haz = _safe_float(early.get("hazard_per_1k", 0.0), 0.0)
    eval_sr = _safe_float(eval_sum.get("success_rate", 0.0), 0.0)
    eval_ret = _safe_float(eval_sum.get("mean_return", 0.0), 0.0)
    eval_haz = _safe_float(eval_sum.get("hazard_per_1k", 0.0), 0.0)
    transfer_early_sr = _safe_float(transfer_early.get("success_rate", 0.0), 0.0)
    transfer_early_ret = _safe_float(transfer_early.get("mean_return", 0.0), 0.0)
    transfer_early_haz = _safe_float(transfer_early.get("hazard_per_1k", 0.0), 0.0)
    scratch_early_fwd = _safe_float(early.get("mean_forward_mse", 0.0), 0.0)
    transfer_early_fwd = _safe_float(transfer_early.get("mean_forward_mse", 0.0), 0.0)
    d_early_success = float(transfer_early_sr - early_sr)
    d_early_return = float(transfer_early_ret - early_ret)
    d_early_hazard_gain = float(early_haz - transfer_early_haz)
    d_early_forward_mse_gain = float(scratch_early_fwd - transfer_early_fwd)
    transfer_early_return_slope = _safe_float(transfer_early_diag.get("return_slope", 0.0), 0.0)
    transfer_early_success_slope = _safe_float(transfer_early_diag.get("success_slope", 0.0), 0.0)
    transfer_early_hazard_slope = _safe_float(transfer_early_diag.get("hazard_slope", 0.0), 0.0)
    transfer_early_forward_mse_slope = _safe_float(transfer_early_diag.get("forward_mse_slope", 0.0), 0.0)
    canary_return_slope = _safe_float(canary_delta.get("return_slope", 0.0), 0.0)
    canary_success_slope = _safe_float(canary_delta.get("success_slope", 0.0), 0.0)
    canary_hazard_slope = _safe_float(canary_delta.get("hazard_slope", 0.0), 0.0)
    canary_forward_mse_slope = _safe_float(canary_delta.get("forward_mse_slope", 0.0), 0.0)
    source_bank_majority = _safe_float(src_bank_agreement.get("mean_majority_fraction", 0.0), 0.0)

    sr_max = _safe_float(thresholds.get("scratch_early_success_max", -1.0), -1.0)
    ret_max = _safe_float(thresholds.get("scratch_early_return_max", -1e9), -1e9)
    haz_min = _safe_float(thresholds.get("scratch_early_hazard_min", -1.0), -1.0)
    eval_sr_max = _safe_float(thresholds.get("scratch_eval_success_max", -1.0), -1.0)
    eval_ret_max = _safe_float(thresholds.get("scratch_eval_return_max", -1e9), -1e9)
    eval_haz_min = _safe_float(thresholds.get("scratch_eval_hazard_min", -1.0), -1.0)
    tr_early_sr_min = _safe_float(thresholds.get("transfer_early_success_min", -1.0), -1.0)
    tr_early_ret_min = _safe_float(thresholds.get("transfer_early_return_min", -1e9), -1e9)
    tr_early_haz_max = _safe_float(thresholds.get("transfer_early_hazard_max", -1.0), -1.0)
    tr_early_fwd_max = _safe_float(thresholds.get("transfer_early_forward_mse_max", -1.0), -1.0)
    d_tr_sc_early_sr_min = _safe_float(thresholds.get("transfer_minus_scratch_early_success_min", -1e9), -1e9)
    d_tr_sc_early_ret_min = _safe_float(thresholds.get("transfer_minus_scratch_early_return_min", -1e9), -1e9)
    d_sc_tr_early_haz_max = _safe_float(thresholds.get("scratch_minus_transfer_early_hazard_max", -1e9), -1e9)
    d_sc_tr_early_fwd_max = _safe_float(
        thresholds.get("scratch_minus_transfer_early_forward_mse_max", -1e9), -1e9
    )
    tr_ret_slope_min = _safe_float(thresholds.get("transfer_early_return_slope_min", -1e9), -1e9)
    tr_succ_slope_min = _safe_float(thresholds.get("transfer_early_success_slope_min", -1e9), -1e9)
    tr_haz_slope_max = _safe_float(thresholds.get("transfer_early_hazard_slope_max", 1e9), 1e9)
    tr_fwd_slope_max = _safe_float(thresholds.get("transfer_early_forward_mse_slope_max", 1e9), 1e9)
    can_ret_slope_min = _safe_float(thresholds.get("canary_delta_return_slope_min", -1e9), -1e9)
    can_succ_slope_min = _safe_float(thresholds.get("canary_delta_success_slope_min", -1e9), -1e9)
    can_haz_slope_max = _safe_float(thresholds.get("canary_delta_hazard_slope_max", 1e9), 1e9)
    can_fwd_slope_max = _safe_float(thresholds.get("canary_delta_forward_mse_slope_max", 1e9), 1e9)
    src_bank_majority_min = _safe_float(thresholds.get("source_policy_bank_majority_min", -1.0), -1.0)
    checks: Dict[str, Optional[bool]] = {
        "scratch_early_success_hard": (None if sr_max < 0.0 else bool(early_sr <= sr_max)),
        "scratch_early_return_hard": (None if ret_max <= -1e8 else bool(early_ret <= ret_max)),
        "scratch_early_hazard_hard": (None if haz_min < 0.0 else bool(early_haz >= haz_min)),
        "scratch_eval_success_hard": (None if eval_sr_max < 0.0 else bool(eval_sr <= eval_sr_max)),
        "scratch_eval_return_hard": (None if eval_ret_max <= -1e8 else bool(eval_ret <= eval_ret_max)),
        "scratch_eval_hazard_hard": (None if eval_haz_min < 0.0 else bool(eval_haz >= eval_haz_min)),
        "transfer_early_success_good": (None if tr_early_sr_min < 0.0 else bool(transfer_early_sr >= tr_early_sr_min)),
        "transfer_early_return_good": (None if tr_early_ret_min <= -1e8 else bool(transfer_early_ret >= tr_early_ret_min)),
        "transfer_early_hazard_good": (None if tr_early_haz_max < 0.0 else bool(transfer_early_haz <= tr_early_haz_max)),
        "transfer_early_forward_mse_good": (None if tr_early_fwd_max < 0.0 else bool(transfer_early_fwd <= tr_early_fwd_max)),
        "canary_delta_early_success_good": (
            None if d_tr_sc_early_sr_min <= -1e8 else bool(d_early_success >= d_tr_sc_early_sr_min)
        ),
        "canary_delta_early_return_good": (
            None if d_tr_sc_early_ret_min <= -1e8 else bool(d_early_return >= d_tr_sc_early_ret_min)
        ),
        "canary_delta_early_hazard_not_too_large": (
            None if d_sc_tr_early_haz_max <= -1e8 else bool(d_early_hazard_gain <= d_sc_tr_early_haz_max)
        ),
        "canary_delta_early_forward_mse_not_too_large": (
            None if d_sc_tr_early_fwd_max <= -1e8 else bool(d_early_forward_mse_gain <= d_sc_tr_early_fwd_max)
        ),
        "transfer_early_return_slope_good": (
            None if tr_ret_slope_min <= -1e8 else bool(transfer_early_return_slope >= tr_ret_slope_min)
        ),
        "transfer_early_success_slope_good": (
            None if tr_succ_slope_min <= -1e8 else bool(transfer_early_success_slope >= tr_succ_slope_min)
        ),
        "transfer_early_hazard_slope_good": (
            None if tr_haz_slope_max >= 1e8 else bool(transfer_early_hazard_slope <= tr_haz_slope_max)
        ),
        "transfer_early_forward_mse_slope_good": (
            None if tr_fwd_slope_max >= 1e8 else bool(transfer_early_forward_mse_slope <= tr_fwd_slope_max)
        ),
        "canary_delta_return_slope_good": (
            None if can_ret_slope_min <= -1e8 else bool(canary_return_slope >= can_ret_slope_min)
        ),
        "canary_delta_success_slope_good": (
            None if can_succ_slope_min <= -1e8 else bool(canary_success_slope >= can_succ_slope_min)
        ),
        "canary_delta_hazard_slope_good": (
            None if can_haz_slope_max >= 1e8 else bool(canary_hazard_slope <= can_haz_slope_max)
        ),
        "canary_delta_forward_mse_slope_good": (
            None if can_fwd_slope_max >= 1e8 else bool(canary_forward_mse_slope <= can_fwd_slope_max)
        ),
        "source_policy_bank_agreement_good": (
            None if src_bank_majority_min < 0.0 else bool(source_bank_majority >= src_bank_majority_min)
        ),
    }
    active = [bool(v) for v in checks.values() if isinstance(v, bool)]
    if not enabled:
        accept = True
        reason = "adaptive_disabled"
    elif not active:
        accept = True
        reason = "no_active_thresholds"
    else:
        if logic == "all":
            accept = bool(all(active))
            reason = "all_hardness_checks"
        else:
            accept = bool(any(active))
            reason = "any_hardness_check"
    return {
        "enabled": bool(enabled),
        "accept_transfer": bool(accept),
        "decision_reason": str(reason),
        "logic": str(logic),
        "thresholds": {
            "scratch_early_success_max": float(sr_max),
            "scratch_early_return_max": float(ret_max),
            "scratch_early_hazard_min": float(haz_min),
            "scratch_eval_success_max": float(eval_sr_max),
            "scratch_eval_return_max": float(eval_ret_max),
            "scratch_eval_hazard_min": float(eval_haz_min),
            "transfer_early_success_min": float(tr_early_sr_min),
            "transfer_early_return_min": float(tr_early_ret_min),
            "transfer_early_hazard_max": float(tr_early_haz_max),
            "transfer_early_forward_mse_max": float(tr_early_fwd_max),
            "transfer_minus_scratch_early_success_min": float(d_tr_sc_early_sr_min),
            "transfer_minus_scratch_early_return_min": float(d_tr_sc_early_ret_min),
            "scratch_minus_transfer_early_hazard_max": float(d_sc_tr_early_haz_max),
            "scratch_minus_transfer_early_forward_mse_max": float(d_sc_tr_early_fwd_max),
            "transfer_early_return_slope_min": float(tr_ret_slope_min),
            "transfer_early_success_slope_min": float(tr_succ_slope_min),
            "transfer_early_hazard_slope_max": float(tr_haz_slope_max),
            "transfer_early_forward_mse_slope_max": float(tr_fwd_slope_max),
            "canary_delta_return_slope_min": float(can_ret_slope_min),
            "canary_delta_success_slope_min": float(can_succ_slope_min),
            "canary_delta_hazard_slope_max": float(can_haz_slope_max),
            "canary_delta_forward_mse_slope_max": float(can_fwd_slope_max),
            "source_policy_bank_majority_min": float(src_bank_majority_min),
        },
        "scratch_early": {
            "success_rate": float(early_sr),
            "mean_return": float(early_ret),
            "hazard_per_1k": float(early_haz),
        },
        "scratch_eval": {
            "success_rate": float(eval_sr),
            "mean_return": float(eval_ret),
            "hazard_per_1k": float(eval_haz),
        },
        "transfer_gate_condition_key": transfer_gate_condition_key,
        "transfer_early": {
            "success_rate": float(transfer_early_sr),
            "mean_return": float(transfer_early_ret),
            "hazard_per_1k": float(transfer_early_haz),
            "mean_forward_mse": float(transfer_early_fwd),
        },
        "canary_early_deltas": {
            "transfer_minus_scratch_success_rate": float(d_early_success),
            "transfer_minus_scratch_mean_return": float(d_early_return),
            "scratch_minus_transfer_hazard_per_1k": float(d_early_hazard_gain),
            "scratch_minus_transfer_forward_mse": float(d_early_forward_mse_gain),
        },
        "transfer_early_trends": {
            "return_slope": float(transfer_early_return_slope),
            "success_slope": float(transfer_early_success_slope),
            "hazard_slope": float(transfer_early_hazard_slope),
            "forward_mse_slope": float(transfer_early_forward_mse_slope),
        },
        "canary_early_delta_trends": {
            "return_slope": float(canary_return_slope),
            "success_slope": float(canary_success_slope),
            "hazard_slope": float(canary_hazard_slope),
            "forward_mse_slope": float(canary_forward_mse_slope),
        },
        "source_policy_bank_agreement": {
            "mean_majority_fraction": float(source_bank_majority),
            "mean_unique_actions": _safe_float(src_bank_agreement.get("mean_unique_actions", 0.0), 0.0),
            "mean_vote_margin": _safe_float(src_bank_agreement.get("mean_vote_margin", 0.0), 0.0),
            "num_snapshots": int(src_bank_agreement.get("num_snapshots", 0) or 0),
            "evaluated_probes": int(src_bank_agreement.get("evaluated_probes", 0) or 0),
        },
        "checks": checks,
    }


def _sf_universe_relation_safe(source_verse_name: str, target_verse_name: str) -> str:
    try:
        from core.taxonomy import universe_relation

        return str(universe_relation(str(source_verse_name), str(target_verse_name)))
    except Exception:
        return "unknown"


def _sf_transfer_decision_record(
    *,
    row: Dict[str, Any],
    adaptive_cfg: Dict[str, Any],
    transfer_condition_key: str,
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
    transfer = tc.get(transfer_condition_key, {}) if isinstance(tc.get(transfer_condition_key), dict) else {}
    decision = _adaptive_gate_decision(
        row,
        adaptive_cfg,
        transfer_gate_condition_key_override=transfer_condition_key,
    )
    accept = bool(decision.get("accept_transfer", False))
    chosen_key = str(transfer_condition_key if accept else "sf_scratch")

    def _safe_block(cond: Dict[str, Any], k: str) -> Dict[str, Any]:
        v = cond.get(k, {})
        return v if isinstance(v, dict) else {}

    scratch_eval = _safe_block(scratch, "eval_summary")
    transfer_eval = _safe_block(transfer, "eval_summary")
    scratch_early = _safe_block(scratch, "early_train_summary")
    transfer_early = _safe_block(transfer, "early_train_summary")

    return {
        "schema_version": "transfer_decision_record.v1",
        "transfer_mode": "sf_transfer_adaptive_gate",
        "seed": int(row.get("seed", 0) or 0),
        "source_verse_name": str(row.get("source_verse_name", "")),
        "target_verse_name": str(row.get("target_verse_name", "")),
        "universe_relation": _sf_universe_relation_safe(
            str(row.get("source_verse_name", "")),
            str(row.get("target_verse_name", "")),
        ),
        "transfer_candidate_condition": str(transfer_condition_key),
        "decision": ("accept_transfer" if accept else "fallback_scratch"),
        "selected_condition": chosen_key,
        "decision_reason": str(decision.get("decision_reason", "")),
        "adaptive_gate": {
            "enabled": bool(adaptive_cfg.get("enabled", False)),
            "kind": str(adaptive_cfg.get("kind", "")),
            "logic": str(adaptive_cfg.get("logic", "")),
            "transfer_gate_condition_key": str(adaptive_cfg.get("transfer_gate_condition_key", "")),
            "thresholds": dict(adaptive_cfg.get("thresholds", {})) if isinstance(adaptive_cfg.get("thresholds"), dict) else {},
        },
        "gate_outcome": {
            "accept_transfer": bool(accept),
            "checks": dict(decision.get("checks", {})) if isinstance(decision.get("checks"), dict) else {},
            "model_gate": dict(decision.get("model_gate", {})) if isinstance(decision.get("model_gate"), dict) else {},
        },
        "gate_inputs": {
            "scratch_early": dict(decision.get("scratch_early", {})) if isinstance(decision.get("scratch_early"), dict) else {},
            "scratch_eval": dict(decision.get("scratch_eval", {})) if isinstance(decision.get("scratch_eval"), dict) else {},
            "transfer_early": dict(decision.get("transfer_early", {})) if isinstance(decision.get("transfer_early"), dict) else {},
            "canary_early_deltas": dict(decision.get("canary_early_deltas", {})) if isinstance(decision.get("canary_early_deltas"), dict) else {},
            "transfer_early_trends": dict(decision.get("transfer_early_trends", {})) if isinstance(decision.get("transfer_early_trends"), dict) else {},
            "canary_early_delta_trends": dict(decision.get("canary_early_delta_trends", {})) if isinstance(decision.get("canary_early_delta_trends"), dict) else {},
            "source_policy_bank_agreement": dict(decision.get("source_policy_bank_agreement", {})) if isinstance(decision.get("source_policy_bank_agreement"), dict) else {},
        },
        "counterfactual": {
            "scratch_eval_success_rate": _safe_float(scratch_eval.get("success_rate", 0.0), 0.0),
            "transfer_eval_success_rate": _safe_float(transfer_eval.get("success_rate", 0.0), 0.0),
            "transfer_minus_scratch_eval_success_rate": (
                _safe_float(transfer_eval.get("success_rate", 0.0), 0.0)
                - _safe_float(scratch_eval.get("success_rate", 0.0), 0.0)
            ),
            "scratch_eval_return": _safe_float(scratch_eval.get("mean_return", 0.0), 0.0),
            "transfer_eval_return": _safe_float(transfer_eval.get("mean_return", 0.0), 0.0),
            "transfer_minus_scratch_eval_return": (
                _safe_float(transfer_eval.get("mean_return", 0.0), 0.0)
                - _safe_float(scratch_eval.get("mean_return", 0.0), 0.0)
            ),
            "scratch_eval_hazard_per_1k": _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0),
            "transfer_eval_hazard_per_1k": _safe_float(transfer_eval.get("hazard_per_1k", 0.0), 0.0),
            "scratch_minus_transfer_eval_hazard_per_1k": (
                _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0)
                - _safe_float(transfer_eval.get("hazard_per_1k", 0.0), 0.0)
            ),
            "scratch_early_success_rate": _safe_float(scratch_early.get("success_rate", 0.0), 0.0),
            "transfer_early_success_rate": _safe_float(transfer_early.get("success_rate", 0.0), 0.0),
        },
    }


def _adaptive_gate_triage_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    if not (isinstance(cfg, dict) and bool(cfg.get("enabled", False))):
        return False
    if str(cfg.get("kind", "")).strip().lower() != "learned_gate_model":
        return False
    policy = cfg.get("model_policy", {}) if isinstance(cfg.get("model_policy"), dict) else {}
    return bool(policy.get("enable_triage", False))


def _adaptive_triage_decision(row: Dict[str, Any], adaptive_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg_full = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
    cfg_warm = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
    payload = cfg_full.get("model", {}) if isinstance(cfg_full.get("model"), dict) else {}
    runtime_policy = cfg_full.get("model_policy", {}) if isinstance(cfg_full.get("model_policy"), dict) else {}
    direct_model = payload.get("triage_direct_model", {}) if isinstance(payload.get("triage_direct_model"), dict) else {}
    if direct_model:
        feats = _adaptive_triage_direct_features(row)
        score = _score_learned_softmax_model(features=feats, model_block=direct_model)
        class_names = [str(x) for x in score.get("class_names", [])]
        probs = [float(x) for x in score.get("probs", [])]
        prob_map = {class_names[i]: probs[i] for i in range(min(len(class_names), len(probs)))}
        payload_policy = payload.get("triage_direct_policy", {}) if isinstance(payload.get("triage_direct_policy"), dict) else {}
        full_thr = _safe_float(runtime_policy.get("triage_full_accept_prob_min", payload_policy.get("full_accept_prob_min", 0.55)), 0.55)
        warm_thr = _safe_float(runtime_policy.get("triage_warmup_accept_prob_min", payload_policy.get("warmup_accept_prob_min", 0.50)), 0.50)
        prefer_higher = bool(payload_policy.get("prefer_higher_probability", True))
        p_s = _safe_float(prob_map.get("sf_scratch", 0.0), 0.0)
        p_f = _safe_float(prob_map.get("sf_transfer", 0.0), 0.0)
        p_w = _safe_float(prob_map.get("sf_transfer_warmup", 0.0), 0.0)
        full_ok = bool(p_f >= full_thr)
        warm_ok = bool(p_w >= warm_thr)
        if full_ok and warm_ok:
            if prefer_higher and p_f >= p_w:
                selected = "sf_transfer"
                reason = "direct_triad_both_accept_pick_full"
            elif prefer_higher:
                selected = "sf_transfer_warmup"
                reason = "direct_triad_both_accept_pick_warmup"
            else:
                # Conservative: prefer the better transfer mode only if it beats scratch confidence.
                if p_f >= p_s and p_f >= p_w:
                    selected = "sf_transfer"
                    reason = "direct_triad_both_accept_full_over_scratch"
                elif p_w >= p_s and p_w >= p_f:
                    selected = "sf_transfer_warmup"
                    reason = "direct_triad_both_accept_warmup_over_scratch"
                else:
                    selected = "sf_scratch"
                    reason = "direct_triad_both_accept_but_scratch_dominates"
        elif full_ok and (p_f >= p_s):
            selected = "sf_transfer"
            reason = "direct_triad_full_only"
        elif warm_ok and (p_w >= p_s):
            selected = "sf_transfer_warmup"
            reason = "direct_triad_warmup_only"
        else:
            selected = "sf_scratch"
            reason = "direct_triad_fallback_scratch"
        selected, reason, canary_override = _canary_triad_override(
            row=row,
            runtime_policy=runtime_policy,
            selected=str(selected),
            decision_reason=str(reason),
        )
        return {
            "selected_condition": str(selected),
            "decision_reason": str(reason),
            "full_decision": {
                "enabled": True,
                "accept_transfer": bool(selected == "sf_transfer"),
                "decision_reason": "direct_triad_selector",
                "model_gate": {
                    "model_type": "triage_softmax_linear",
                    "probability_help": float(p_f),
                    "probability_hard": None,
                },
            },
            "warm_decision": {
                "enabled": True,
                "accept_transfer": bool(selected == "sf_transfer_warmup"),
                "decision_reason": "direct_triad_selector",
                "model_gate": {
                    "model_type": "triage_softmax_linear",
                    "probability_help": float(p_w),
                    "probability_hard": None,
                },
            },
            "triage_policy": {
                "full_accept_prob_min": float(full_thr),
                "warmup_accept_prob_min": float(warm_thr),
                "prefer_higher_probability": bool(prefer_higher),
            },
            "direct_triad": {
                "class_names": class_names,
                "probs": [float(x) for x in probs],
                "probabilities": {
                    "sf_scratch": float(p_s),
                    "sf_transfer": float(p_f),
                    "sf_transfer_warmup": float(p_w),
                },
                "feature_count": int(score.get("feature_count", 0)),
            },
            "canary_override": canary_override,
        }
    triage_policy = payload.get("triage_policy", {}) if isinstance(payload.get("triage_policy"), dict) else {}
    if triage_policy:
        mp_full = dict(cfg_full.get("model_policy", {})) if isinstance(cfg_full.get("model_policy"), dict) else {}
        mp_warm = dict(cfg_warm.get("model_policy", {})) if isinstance(cfg_warm.get("model_policy"), dict) else {}
        if "hard_prob_min" in triage_policy:
            hp = _safe_float(triage_policy.get("hard_prob_min"), -1.0)
            mp_full["hard_prob_min"] = float(hp)
            mp_warm["hard_prob_min"] = float(hp)
        if "full_accept_prob_min" in triage_policy:
            fp = _safe_float(triage_policy.get("full_accept_prob_min"), 0.6)
            mp_full["accept_prob_min"] = float(fp)
            mp_full["warmup_prob_min"] = float(fp)
        if "warmup_accept_prob_min" in triage_policy:
            wp = _safe_float(triage_policy.get("warmup_accept_prob_min"), 0.4)
            mp_warm["accept_prob_min"] = float(wp)
            mp_warm["warmup_prob_min"] = float(wp)
        if "triage_full_accept_prob_min" in runtime_policy:
            fp_rt = _safe_float(runtime_policy.get("triage_full_accept_prob_min"), mp_full.get("accept_prob_min", 0.6))
            mp_full["accept_prob_min"] = float(fp_rt)
            mp_full["warmup_prob_min"] = float(fp_rt)
        if "triage_warmup_accept_prob_min" in runtime_policy:
            wp_rt = _safe_float(runtime_policy.get("triage_warmup_accept_prob_min"), mp_warm.get("accept_prob_min", 0.4))
            mp_warm["accept_prob_min"] = float(wp_rt)
            mp_warm["warmup_prob_min"] = float(wp_rt)
        cfg_full["model_policy"] = mp_full
        cfg_warm["model_policy"] = mp_warm
    d_full = _adaptive_gate_decision(row, cfg_full, transfer_gate_condition_key_override="sf_transfer")
    d_warm = _adaptive_gate_decision(row, cfg_warm, transfer_gate_condition_key_override="sf_transfer_warmup")
    mg_full = d_full.get("model_gate", {}) if isinstance(d_full.get("model_gate"), dict) else {}
    mg_warm = d_warm.get("model_gate", {}) if isinstance(d_warm.get("model_gate"), dict) else {}
    p_full = _safe_float(mg_full.get("probability_help", 0.0), 0.0)
    p_warm = _safe_float(mg_warm.get("probability_help", 0.0), 0.0)
    full_ok = bool(d_full.get("accept_transfer", False))
    warm_ok = bool(d_warm.get("accept_transfer", False))
    if full_ok and warm_ok:
        if p_full >= p_warm:
            selected = "sf_transfer"
            reason = "triage_both_accept_pick_full"
        else:
            selected = "sf_transfer_warmup"
            reason = "triage_both_accept_pick_warmup"
    elif full_ok:
        selected = "sf_transfer"
        reason = "triage_full_only"
    elif warm_ok:
        selected = "sf_transfer_warmup"
        reason = "triage_warmup_only"
    else:
        selected = "sf_scratch"
        reason = "triage_fallback_scratch"
    selected, reason, canary_override = _canary_triad_override(
        row=row,
        runtime_policy=runtime_policy,
        selected=str(selected),
        decision_reason=str(reason),
    )
    return {
        "enabled": True,
        "kind": "learned_gate_triad_selector",
        "selected_condition": str(selected),
        "decision_reason": str(reason),
        "triage_policy": dict(triage_policy) if isinstance(triage_policy, dict) else {},
        "canary_override": canary_override,
        "candidates": {
            "sf_transfer": {
                "accept_transfer": bool(full_ok),
                "probability_help": float(p_full),
                "model_gate": dict(mg_full),
                "checks": dict(d_full.get("checks", {})) if isinstance(d_full.get("checks"), dict) else {},
            },
            "sf_transfer_warmup": {
                "accept_transfer": bool(warm_ok),
                "probability_help": float(p_warm),
                "model_gate": dict(mg_warm),
                "checks": dict(d_warm.get("checks", {})) if isinstance(d_warm.get("checks"), dict) else {},
            },
        },
        "scratch_early": dict(d_full.get("scratch_early", {})) if isinstance(d_full.get("scratch_early"), dict) else {},
        "scratch_eval": dict(d_full.get("scratch_eval", {})) if isinstance(d_full.get("scratch_eval"), dict) else {},
    }


def _sf_transfer_triage_decision_record(
    *,
    row: Dict[str, Any],
    adaptive_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tc = row.get("target_conditions", {}) if isinstance(row.get("target_conditions"), dict) else {}
    tri = _adaptive_triage_decision(row, adaptive_cfg)
    sel = str(tri.get("selected_condition", "sf_scratch"))
    scratch_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
    full_eval = (((tc.get("sf_transfer") or {}) if isinstance(tc.get("sf_transfer"), dict) else {}).get("eval_summary") or {})
    warm_eval = (((tc.get("sf_transfer_warmup") or {}) if isinstance(tc.get("sf_transfer_warmup"), dict) else {}).get("eval_summary") or {})
    return {
        "schema_version": "transfer_decision_record.v1",
        "transfer_mode": "sf_transfer_adaptive_triad_gate",
        "seed": int(row.get("seed", 0) or 0),
        "source_verse_name": str(row.get("source_verse_name", "")),
        "target_verse_name": str(row.get("target_verse_name", "")),
        "universe_relation": _sf_universe_relation_safe(
            str(row.get("source_verse_name", "")),
            str(row.get("target_verse_name", "")),
        ),
        "decision": ("accept_transfer" if sel != "sf_scratch" else "fallback_scratch"),
        "selected_condition": sel,
        "decision_reason": str(tri.get("decision_reason", "")),
        "triage": tri,
        "counterfactual": {
            "scratch_eval_success_rate": _safe_float(scratch_eval.get("success_rate", 0.0), 0.0),
            "transfer_eval_success_rate": _safe_float(full_eval.get("success_rate", 0.0), 0.0),
            "warmup_eval_success_rate": _safe_float(warm_eval.get("success_rate", 0.0), 0.0),
            "scratch_eval_return": _safe_float(scratch_eval.get("mean_return", 0.0), 0.0),
            "transfer_eval_return": _safe_float(full_eval.get("mean_return", 0.0), 0.0),
            "warmup_eval_return": _safe_float(warm_eval.get("mean_return", 0.0), 0.0),
            "scratch_eval_hazard_per_1k": _safe_float(scratch_eval.get("hazard_per_1k", 0.0), 0.0),
            "transfer_eval_hazard_per_1k": _safe_float(full_eval.get("hazard_per_1k", 0.0), 0.0),
            "warmup_eval_hazard_per_1k": _safe_float(warm_eval.get("hazard_per_1k", 0.0), 0.0),
        },
    }


def _summarize_condition_from_rows(rows: List[Dict[str, Any]], condition_key: str) -> Dict[str, Any]:
    def _collect_metric(path: Tuple[str, ...]) -> List[float]:
        vals: List[float] = []
        for r in rows:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def _mean(vals: Iterable[float]) -> Optional[float]:
        arr = list(vals)
        if not arr:
            return None
        return float(sum(arr) / float(len(arr)))

    return {
        "mean_eval_return": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "mean_return"))),
        "mean_eval_success_rate": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "success_rate"))),
        "mean_eval_hazard_per_1k": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "hazard_per_1k"))),
        "mean_early_success_rate": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "success_rate"))),
        "mean_early_return": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "mean_return"))),
        "mean_early_hazard_per_1k": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "hazard_per_1k"))),
        "mean_early_forward_mse": _mean(_collect_metric(("target_conditions", condition_key, "early_train_summary", "mean_forward_mse"))),
        "mean_eval_forward_mse": _mean(_collect_metric(("target_conditions", condition_key, "eval_summary", "mean_forward_mse"))),
    }


def _summarize_adaptive_triage(
    rows: List[Dict[str, Any]],
    *,
    adaptive_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_metrics: List[Dict[str, Optional[float]]] = []
    tri_rows: List[Dict[str, Any]] = []
    counts = {"sf_scratch": 0, "sf_transfer": 0, "sf_transfer_warmup": 0}
    for r in rows:
        tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
        if not isinstance(tc, dict):
            continue
        tri = _adaptive_triage_decision(r, adaptive_cfg)
        tri_rows.append(tri)
        sel = str(tri.get("selected_condition", "sf_scratch"))
        if sel not in counts:
            sel = "sf_scratch"
        counts[sel] += 1
        chosen = tc.get(sel, {}) if isinstance(tc.get(sel), dict) else {}
        if not isinstance(chosen, dict):
            chosen = {}
        chosen_metrics.append(_seed_condition_summary(chosen))

    def _mean_key(k: str) -> Optional[float]:
        vals = [float(m[k]) for m in chosen_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    summary = {
        "mean_eval_return": _mean_key("eval_return"),
        "mean_eval_success_rate": _mean_key("eval_success_rate"),
        "mean_eval_hazard_per_1k": _mean_key("eval_hazard_per_1k"),
        "mean_early_success_rate": _mean_key("early_success_rate"),
        "mean_early_return": _mean_key("early_return"),
        "mean_early_hazard_per_1k": _mean_key("early_hazard_per_1k"),
        "mean_early_forward_mse": _mean_key("early_forward_mse"),
        "mean_eval_forward_mse": _mean_key("eval_forward_mse"),
    }
    n_eval = max(1, len(tri_rows))
    gate_info = {
        "enabled": True,
        "kind": "learned_gate_triad_selector",
        "num_seeds": int(len(rows)),
        "evaluated_seeds": int(len(tri_rows)),
        "selected_counts": {k: int(v) for k, v in counts.items()},
        "accept_transfer_count": int(counts["sf_transfer"] + counts["sf_transfer_warmup"]),
        "fallback_to_scratch_count": int(counts["sf_scratch"]),
        "accept_transfer_rate": float((counts["sf_transfer"] + counts["sf_transfer_warmup"]) / float(n_eval)),
        "full_transfer_rate": float(counts["sf_transfer"] / float(n_eval)),
        "warmup_transfer_rate": float(counts["sf_transfer_warmup"] / float(n_eval)),
    }
    return summary, gate_info


def _summarize_adaptive_condition(
    rows: List[Dict[str, Any]],
    *,
    transfer_condition_key: str,
    adaptive_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_metrics: List[Dict[str, Optional[float]]] = []
    accept_count = 0
    fallback_count = 0
    gate_rows: List[Dict[str, Any]] = []
    for r in rows:
        tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
        scratch = tc.get("sf_scratch", {}) if isinstance(tc.get("sf_scratch"), dict) else {}
        transfer = tc.get(transfer_condition_key, {}) if isinstance(tc.get(transfer_condition_key), dict) else {}
        if not isinstance(scratch, dict) or not isinstance(transfer, dict):
            continue
        decision = _adaptive_gate_decision(
            r,
            adaptive_cfg,
            transfer_gate_condition_key_override=transfer_condition_key,
        )
        gate_rows.append(decision)
        if bool(decision.get("accept_transfer", False)):
            chosen = transfer
            accept_count += 1
        else:
            chosen = scratch
            fallback_count += 1
        chosen_metrics.append(_seed_condition_summary(chosen))

    def _mean_key(k: str) -> Optional[float]:
        vals = [float(m[k]) for m in chosen_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    summary = {
        "mean_eval_return": _mean_key("eval_return"),
        "mean_eval_success_rate": _mean_key("eval_success_rate"),
        "mean_eval_hazard_per_1k": _mean_key("eval_hazard_per_1k"),
        "mean_early_success_rate": _mean_key("early_success_rate"),
        "mean_early_return": _mean_key("early_return"),
        "mean_early_hazard_per_1k": _mean_key("early_hazard_per_1k"),
        "mean_early_forward_mse": _mean_key("early_forward_mse"),
        "mean_eval_forward_mse": _mean_key("eval_forward_mse"),
    }

    gate_info = {
        "enabled": bool(adaptive_cfg.get("enabled", False)),
        "kind": str(adaptive_cfg.get("kind", "hybrid_hardness_quality_gate")),
        "logic": str(adaptive_cfg.get("logic", "any")),
        "transfer_gate_condition_key": str(transfer_condition_key),
        "thresholds": dict(adaptive_cfg.get("thresholds", {})) if isinstance(adaptive_cfg.get("thresholds"), dict) else {},
        "num_seeds": int(len(rows)),
        "evaluated_seeds": int(len(gate_rows)),
        "accept_transfer_count": int(accept_count),
        "fallback_to_scratch_count": int(fallback_count),
        "accept_transfer_rate": (
            float(accept_count / float(max(1, accept_count + fallback_count)))
            if (accept_count + fallback_count) > 0
            else 0.0
        ),
        "mean_scratch_early_success_rate": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_early_return": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_early_hazard_per_1k": (
            float(sum(_safe_float((g.get("scratch_early") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_success_rate": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_return": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_scratch_eval_hazard_per_1k": (
            float(sum(_safe_float((g.get("scratch_eval") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_success_rate": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_return": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_hazard_per_1k": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_transfer_early_forward_mse": (
            float(sum(_safe_float((g.get("transfer_early") or {}).get("mean_forward_mse", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_delta_success_rate": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("transfer_minus_scratch_success_rate", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_delta_return": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("transfer_minus_scratch_mean_return", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_hazard_gain_per_1k": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("scratch_minus_transfer_hazard_per_1k", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
        "mean_canary_early_forward_mse_gain": (
            float(sum(_safe_float((g.get("canary_early_deltas") or {}).get("scratch_minus_transfer_forward_mse", 0.0), 0.0) for g in gate_rows) / float(len(gate_rows)))
            if gate_rows else 0.0
        ),
    }
    return summary, gate_info


def _aggregate_seed_block(rows: List[Dict[str, Any]], adaptive_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    def _collect(path: Tuple[str, ...]) -> List[float]:
        vals: List[float] = []
        for r in rows:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def _mean(vals: Iterable[float]) -> Optional[float]:
        arr = list(vals)
        if not arr:
            return None
        return float(sum(arr) / float(len(arr)))

    out: Dict[str, Any] = {"num_seeds": int(len(rows))}
    for cond in ("sf_scratch", "sf_transfer", "sf_transfer_warmup"):
        out[cond] = _summarize_condition_from_rows(rows, cond)

    if isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False)):
        adap_full, gate_info_full = _summarize_adaptive_condition(
            rows,
            transfer_condition_key="sf_transfer",
            adaptive_cfg=adaptive_cfg,
        )
        adap_warm, gate_info_warm = _summarize_adaptive_condition(
            rows,
            transfer_condition_key="sf_transfer_warmup",
            adaptive_cfg=adaptive_cfg,
        )
        out["sf_adaptive_transfer"] = adap_full
        out["sf_adaptive_transfer_warmup"] = adap_warm
        out["adaptive_gate"] = {
            "sf_transfer": gate_info_full,
            "sf_transfer_warmup": gate_info_warm,
        }
        if _adaptive_gate_triage_enabled(adaptive_cfg):
            adap_triage, gate_info_triage = _summarize_adaptive_triage(rows, adaptive_cfg=adaptive_cfg)
            out["sf_adaptive_triad"] = adap_triage
            out["adaptive_gate"]["triage"] = gate_info_triage

    def _success_harm_rate(*, transfer_key: str) -> Optional[float]:
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            t_eval = (((tc.get(transfer_key) or {}) if isinstance(tc.get(transfer_key), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict) or not isinstance(t_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            t_sr = t_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)) or not isinstance(t_sr, (int, float)):
                continue
            total += 1
            if float(t_sr) < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _adaptive_success_harm_rate(*, transfer_key: str) -> Optional[float]:
        if not (isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False))):
            return None
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            t_eval = (((tc.get(transfer_key) or {}) if isinstance(tc.get(transfer_key), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict) or not isinstance(t_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            t_sr = t_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)) or not isinstance(t_sr, (int, float)):
                continue
            d = _adaptive_gate_decision(
                r,
                adaptive_cfg,
                transfer_gate_condition_key_override=transfer_key,
            )
            chosen_sr = float(t_sr) if bool(d.get("accept_transfer", False)) else float(s_sr)
            total += 1
            if chosen_sr < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _adaptive_triage_success_harm_rate() -> Optional[float]:
        if not (isinstance(adaptive_cfg, dict) and bool(adaptive_cfg.get("enabled", False)) and _adaptive_gate_triage_enabled(adaptive_cfg)):
            return None
        harms = 0
        total = 0
        for r in rows:
            tc = r.get("target_conditions", {}) if isinstance(r.get("target_conditions"), dict) else {}
            s_eval = (((tc.get("sf_scratch") or {}) if isinstance(tc.get("sf_scratch"), dict) else {}).get("eval_summary") or {})
            if not isinstance(s_eval, dict):
                continue
            s_sr = s_eval.get("success_rate")
            if not isinstance(s_sr, (int, float)):
                continue
            tri = _adaptive_triage_decision(r, adaptive_cfg)
            sel = str(tri.get("selected_condition", "sf_scratch"))
            t_eval = (((tc.get(sel) or {}) if isinstance(tc.get(sel), dict) else {}).get("eval_summary") or {})
            t_sr = t_eval.get("success_rate")
            if not isinstance(t_sr, (int, float)):
                continue
            total += 1
            if float(t_sr) < float(s_sr):
                harms += 1
        if total <= 0:
            return None
        return float(harms / float(total))

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    transfer_ret = out["sf_transfer"]["mean_eval_return"]
    scratch_ret = out["sf_scratch"]["mean_eval_return"]
    transfer_warmup_ret = out["sf_transfer_warmup"]["mean_eval_return"]
    transfer_haz = out["sf_transfer"]["mean_eval_hazard_per_1k"]
    scratch_haz = out["sf_scratch"]["mean_eval_hazard_per_1k"]
    transfer_warmup_haz = out["sf_transfer_warmup"]["mean_eval_hazard_per_1k"]
    transfer_sr = out["sf_transfer"]["mean_eval_success_rate"]
    scratch_sr = out["sf_scratch"]["mean_eval_success_rate"]
    transfer_warmup_sr = out["sf_transfer_warmup"]["mean_eval_success_rate"]

    out["comparison"] = {
        "transfer_minus_scratch_eval_return": _delta(transfer_ret, scratch_ret),
        "transfer_warmup_minus_scratch_eval_return": _delta(transfer_warmup_ret, scratch_ret),
        "transfer_minus_scratch_eval_success_rate": _delta(transfer_sr, scratch_sr),
        "transfer_warmup_minus_scratch_eval_success_rate": _delta(transfer_warmup_sr, scratch_sr),
        "scratch_minus_transfer_eval_hazard_per_1k": _delta(scratch_haz, transfer_haz),
        "scratch_minus_transfer_warmup_eval_hazard_per_1k": _delta(scratch_haz, transfer_warmup_haz),
        "negative_transfer_rate_success_sf_transfer": _success_harm_rate(transfer_key="sf_transfer"),
        "negative_transfer_rate_success_sf_transfer_warmup": _success_harm_rate(transfer_key="sf_transfer_warmup"),
    }
    if "sf_adaptive_transfer" in out:
        adap_ret = out["sf_adaptive_transfer"]["mean_eval_return"]
        adap_haz = out["sf_adaptive_transfer"]["mean_eval_hazard_per_1k"]
        adap_sr = out["sf_adaptive_transfer"]["mean_eval_success_rate"]
        adap_warm_ret = out["sf_adaptive_transfer_warmup"]["mean_eval_return"]
        adap_warm_haz = out["sf_adaptive_transfer_warmup"]["mean_eval_hazard_per_1k"]
        adap_warm_sr = out["sf_adaptive_transfer_warmup"]["mean_eval_success_rate"]
        out["comparison"].update(
            {
                "adaptive_transfer_minus_scratch_eval_return": _delta(adap_ret, scratch_ret),
                "adaptive_transfer_warmup_minus_scratch_eval_return": _delta(adap_warm_ret, scratch_ret),
                "adaptive_transfer_minus_scratch_eval_success_rate": _delta(adap_sr, scratch_sr),
                "adaptive_transfer_warmup_minus_scratch_eval_success_rate": _delta(adap_warm_sr, scratch_sr),
                "scratch_minus_adaptive_transfer_eval_hazard_per_1k": _delta(scratch_haz, adap_haz),
                "scratch_minus_adaptive_transfer_warmup_eval_hazard_per_1k": _delta(scratch_haz, adap_warm_haz),
                "negative_transfer_rate_success_sf_adaptive_transfer": _adaptive_success_harm_rate(
                    transfer_key="sf_transfer"
                ),
                "negative_transfer_rate_success_sf_adaptive_transfer_warmup": _adaptive_success_harm_rate(
                    transfer_key="sf_transfer_warmup"
                ),
            }
        )
    if "sf_adaptive_triad" in out:
        adap_tri_ret = out["sf_adaptive_triad"]["mean_eval_return"]
        adap_tri_haz = out["sf_adaptive_triad"]["mean_eval_hazard_per_1k"]
        adap_tri_sr = out["sf_adaptive_triad"]["mean_eval_success_rate"]
        out["comparison"].update(
            {
                "adaptive_triad_minus_scratch_eval_return": _delta(adap_tri_ret, scratch_ret),
                "adaptive_triad_minus_scratch_eval_success_rate": _delta(adap_tri_sr, scratch_sr),
                "scratch_minus_adaptive_triad_eval_hazard_per_1k": _delta(scratch_haz, adap_tri_haz),
                "negative_transfer_rate_success_sf_adaptive_triad": _adaptive_triage_success_harm_rate(),
            }
        )
    return out


def _profile_params(*, profile: str, max_steps: int) -> Dict[str, Any]:
    p = str(profile).strip().lower()
    if p == "near_transfer":
        return {
            "profile": p,
            "description": "Navigation-isolated warehouse profile (near-transfer).",
            "source_verse_name": "grid_world",
            "target_verse_name": "warehouse_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 32,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 10,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.02,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 14,
                "patrol_robot": False,
                "conveyor_count": 0,
                "battery_drain": 0,
                "lidar_range": 4,
                "step_penalty": -0.10,
                "adr_enabled": False,
            },
        }
    if p == "default_like":
        return {
            "profile": p,
            "description": "Warehouse default-like profile with patrol, conveyor, and battery costs.",
            "source_verse_name": "grid_world",
            "target_verse_name": "warehouse_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 32,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 12,
                "ice_count": 2,
                "teleporter_pairs": 1,
                "step_penalty": -0.02,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 8,
                "height": 8,
                "max_steps": int(max_steps),
                "obstacle_count": 14,
                "patrol_robot": True,
                "conveyor_count": 3,
                "battery_drain": 1,
                "lidar_range": 4,
                "step_penalty": -0.10,
                "adr_enabled": False,
            },
        }
    if p == "maze_near_transfer":
        return {
            "profile": p,
            "description": "Task-solving transfer curriculum stage: grid_world -> maze_world (5x5, no hazards).",
            "source_verse_name": "grid_world",
            "target_verse_name": "maze_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 24,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "step_penalty": -0.005,
                "bump_penalty": -0.02,
                "explore_bonus": 0.0,
                "hazard_count": 0,
                "fog_of_war": False,
                "adr_enabled": False,
            },
        }
    if p == "grid_same_transfer":
        return {
            "profile": p,
            "description": "Rung-1 task-solving transfer: grid_world -> harder held-out grid_world.",
            "source_verse_name": "grid_world",
            "target_verse_name": "grid_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 20,
            "source_policy_snapshots": 4,
            "source_params": {
                "width": 6,
                "height": 6,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 6,
                "height": 6,
                "max_steps": int(max_steps),
                "obstacle_count": 4,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
        }
    if p == "maze_to_grid_near_transfer":
        return {
            "profile": p,
            "description": "Same-universe reverse curriculum stage: maze_world -> grid_world (5x5 core navigation).",
            "source_verse_name": "maze_world",
            "target_verse_name": "grid_world",
            "source_allowed_actions": [0, 1, 2, 3],
            "target_allowed_actions": [0, 1, 2, 3],
            "target_w_estimation_steps": 24,
            "source_policy_snapshots": 3,
            "source_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "step_penalty": -0.005,
                "bump_penalty": -0.02,
                "explore_bonus": 0.0,
                "hazard_count": 0,
                "fog_of_war": False,
                "adr_enabled": False,
            },
            "target_params": {
                "width": 5,
                "height": 5,
                "max_steps": int(max_steps),
                "obstacle_count": 3,
                "ice_count": 0,
                "teleporter_pairs": 0,
                "step_penalty": -0.01,
                "adr_enabled": False,
            },
        }
    raise ValueError(
        "Unknown profile: "
        f"{profile}. Expected near_transfer, default_like, maze_near_transfer, grid_same_transfer, or maze_to_grid_near_transfer."
    )


def _phase2_score(summary: Dict[str, Any]) -> float:
    cmp = summary.get("comparison", {}) if isinstance(summary.get("comparison"), dict) else {}
    ret_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0)
    haz_gain = _safe_float(cmp.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0)
    sr_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0)
    # Return remains primary; hazard and success are tie-break style bonuses.
    return float(ret_delta + 0.02 * haz_gain + 20.0 * sr_delta)


def _transfer_pair_policy_check(
    *,
    source_verse_name: str,
    target_verse_name: str,
    disable_universe_policy: bool,
    allow_adjacent_universe_transfer: bool,
    allow_cross_universe_transfer: bool,
) -> Dict[str, Any]:
    src = str(source_verse_name or "").strip().lower()
    tgt = str(target_verse_name or "").strip().lower()
    relation = "unknown"
    try:
        from core.taxonomy import universe_relation

        relation = str(universe_relation(src, tgt))
    except Exception:
        relation = "unknown"

    if bool(disable_universe_policy):
        return {
            "enabled": False,
            "relation": relation,
            "allowed": True,
            "reason": "universe_policy_disabled",
        }

    allowed = False
    reason = "blocked_unknown_relation"
    if relation.startswith("same_universe:"):
        allowed = True
        reason = "same_universe_allowed"
    elif relation.startswith("adjacent_universe:"):
        allowed = bool(allow_adjacent_universe_transfer)
        reason = "adjacent_universe_override" if allowed else "adjacent_universe_blocked"
    elif relation.startswith("cross_universe:"):
        allowed = bool(allow_cross_universe_transfer)
        reason = "cross_universe_override" if allowed else "cross_universe_blocked"

    return {
        "enabled": True,
        "relation": relation,
        "allowed": bool(allowed),
        "reason": reason,
        "allow_adjacent_universe_transfer": bool(allow_adjacent_universe_transfer),
        "allow_cross_universe_transfer": bool(allow_cross_universe_transfer),
    }


def _gate_check_from_comparison(
    *,
    comparison: Dict[str, Any],
    min_return_delta: float,
    min_hazard_gain: float,
    min_success_delta: float,
) -> Dict[str, Any]:
    ret = _safe_float(comparison.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0)
    haz = _safe_float(comparison.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0)
    sr = _safe_float(comparison.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0)
    ok = bool(
        ret >= float(min_return_delta)
        and haz >= float(min_hazard_gain)
        and sr >= float(min_success_delta)
    )
    return {
        "ok": ok,
        "transfer_warmup_minus_scratch_eval_return": float(ret),
        "scratch_minus_transfer_warmup_eval_hazard_per_1k": float(haz),
        "transfer_warmup_minus_scratch_eval_success_rate": float(sr),
        "thresholds": {
            "min_return_delta": float(min_return_delta),
            "min_hazard_gain": float(min_hazard_gain),
            "min_success_delta": float(min_success_delta),
        },
    }


def _evaluate_config(
    *,
    seeds: Sequence[int],
    profile_params: Dict[str, Any],
    ego_size: int,
    source_train_episodes: int,
    target_train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    warmup_psi_episodes: int,
    target_w_estimation_steps: int = 0,
    source_policy_snapshots: int = 3,
    adaptive_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    adapter = EgoGridAdapter(size=int(ego_size))
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        row = _train_then_eval(
            seed=int(s),
            adapter=adapter,
            source_verse_name=str(profile_params.get("source_verse_name", "grid_world")),
            target_verse_name=str(profile_params.get("target_verse_name", "warehouse_world")),
            source_params=dict(profile_params["source_params"]),
            target_params=dict(profile_params["target_params"]),
            source_train_episodes=int(source_train_episodes),
            target_train_episodes=int(target_train_episodes),
            eval_episodes=int(eval_episodes),
            max_steps=int(max_steps),
            warmup_psi_episodes=int(warmup_psi_episodes),
            source_allowed_actions=profile_params.get("source_allowed_actions"),
            target_allowed_actions=profile_params.get("target_allowed_actions"),
            target_w_estimation_steps=int(
                profile_params.get("target_w_estimation_steps", target_w_estimation_steps)
            ),
            source_policy_snapshots=int(
                profile_params.get("source_policy_snapshots", source_policy_snapshots)
            ),
        )
        if isinstance(adaptive_gate, dict) and bool(adaptive_gate.get("enabled", False)):
            row["adaptive_transfer_decision_records"] = [
                _sf_transfer_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                    transfer_condition_key="sf_transfer",
                ),
                _sf_transfer_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                    transfer_condition_key="sf_transfer_warmup",
                ),
            ]
            if _adaptive_gate_triage_enabled(adaptive_gate):
                row["adaptive_transfer_triage_decision_record"] = _sf_transfer_triage_decision_record(
                    row=row,
                    adaptive_cfg=adaptive_gate,
                )
        per_seed.append(row)

    summary = _aggregate_seed_block(per_seed, adaptive_cfg=adaptive_gate)
    out = {
        "config": {
            "source_verse_name": str(profile_params.get("source_verse_name", "grid_world")),
            "target_verse_name": str(profile_params.get("target_verse_name", "warehouse_world")),
            "ego_size": int(ego_size),
            "source_train_episodes": int(source_train_episodes),
            "target_train_episodes": int(target_train_episodes),
            "eval_episodes": int(eval_episodes),
            "warmup_psi_episodes": int(warmup_psi_episodes),
            "max_steps": int(max_steps),
            "target_w_estimation_steps": int(
                profile_params.get("target_w_estimation_steps", target_w_estimation_steps)
            ),
            "source_policy_snapshots": int(
                profile_params.get("source_policy_snapshots", source_policy_snapshots)
            ),
        },
        "summary": summary,
        "score": _phase2_score(summary),
        "per_seed": per_seed,
    }
    if isinstance(adaptive_gate, dict) and bool(adaptive_gate.get("enabled", False)):
        out["adaptive_gate"] = dict(adaptive_gate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="phase2", choices=["single", "phase2"])
    ap.add_argument("--seeds", type=str, default="123,223,337")
    ap.add_argument("--profiles", type=str, default="near_transfer,default_like")
    ap.add_argument("--ego_size", type=int, default=5)
    ap.add_argument("--warmup_psi_episodes", type=int, default=8)
    ap.add_argument("--source_train_episodes", type=int, default=120)
    ap.add_argument("--target_train_episodes", type=int, default=80)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--target_w_estimation_steps", type=int, default=32)
    ap.add_argument("--source_policy_snapshots", type=int, default=3)
    ap.add_argument("--sweep_ego_sizes", type=str, default="3,5,7")
    ap.add_argument("--sweep_source_episodes", type=str, default="80,120")
    ap.add_argument("--sweep_warmup_episodes", type=str, default="0,8,16")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--strict_gate", action="store_true")
    ap.add_argument("--gate_min_return_delta", type=float, default=0.0)
    ap.add_argument("--gate_min_hazard_gain", type=float, default=0.0)
    ap.add_argument("--gate_min_success_delta", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_enabled", action="store_true")
    ap.add_argument("--adaptive_gate_logic", type=str, default="any", choices=["any", "all"])
    ap.add_argument("--adaptive_gate_scratch_early_success_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_early_return_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_early_hazard_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_eval_success_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_scratch_eval_return_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_eval_hazard_min", type=float, default=-1.0)
    ap.add_argument(
        "--adaptive_gate_transfer_condition",
        type=str,
        default="sf_transfer_warmup",
        choices=["sf_transfer", "sf_transfer_warmup"],
    )
    ap.add_argument("--adaptive_gate_transfer_early_success_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_early_return_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_hazard_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_early_forward_mse_max", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_transfer_minus_scratch_early_success_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_minus_scratch_early_return_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_minus_transfer_early_hazard_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_scratch_minus_transfer_early_forward_mse_max", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_return_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_success_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_transfer_early_hazard_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_transfer_early_forward_mse_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_canary_delta_return_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_canary_delta_success_slope_min", type=float, default=-1e9)
    ap.add_argument("--adaptive_gate_canary_delta_hazard_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_canary_delta_forward_mse_slope_max", type=float, default=1e9)
    ap.add_argument("--adaptive_gate_source_policy_bank_majority_min", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_json", type=str, default="")
    ap.add_argument("--adaptive_gate_model_accept_prob", type=float, default=0.5)
    ap.add_argument("--adaptive_gate_model_warmup_prob", type=float, default=0.35)
    ap.add_argument("--adaptive_gate_model_hard_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_triage_full_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_triage_warmup_prob", type=float, default=-2.0)
    ap.add_argument("--adaptive_gate_model_enable_canary_triad_override", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_success_weight", type=float, default=100.0)
    ap.add_argument("--adaptive_gate_model_canary_return_weight", type=float, default=1.0)
    ap.add_argument("--adaptive_gate_model_canary_hazard_weight", type=float, default=0.02)
    ap.add_argument("--adaptive_gate_model_canary_min_utility", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_canary_min_hazard_gain", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_min_episodes", type=float, default=1.0)
    ap.add_argument("--adaptive_gate_model_canary_two_stage_global_select", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_scratch_utility", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_enable_canary_dual_policy", action="store_true")
    ap.add_argument("--adaptive_gate_model_canary_dual_hazard_gain_threshold", type=float, default=0.0)
    ap.add_argument("--adaptive_gate_model_canary_dual_success_delta_threshold", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_success_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_return_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_hazard_weight", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_utility", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_hazard_gain", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_safety_min_episodes", type=float, default=-1.0)
    ap.add_argument("--adaptive_gate_model_canary_full_success_floor", type=float, default=-1000000000.0)
    ap.add_argument("--adaptive_gate_model_canary_full_hazard_gain_override", type=float, default=25.0)
    ap.add_argument("--adaptive_gate_model_enable_triage", action="store_true")
    ap.add_argument("--disable_universe_policy", action="store_true")
    ap.add_argument("--allow_adjacent_universe_transfer", action="store_true")
    ap.add_argument("--allow_cross_universe_transfer", action="store_true")
    ap.add_argument("--out_json", type=str, default="models/validation/sf_transfer_validation_v2_phase2.json")
    args = ap.parse_args()

    register_builtin()
    seeds = _parse_seed_list(args.seeds)
    profiles = _parse_str_list(args.profiles, default=["near_transfer", "default_like"])
    adaptive_gate_cfg = _adaptive_gate_cfg(
        enabled=bool(args.adaptive_gate_enabled),
        scratch_early_success_max=float(args.adaptive_gate_scratch_early_success_max),
        scratch_early_return_max=float(args.adaptive_gate_scratch_early_return_max),
        scratch_early_hazard_min=float(args.adaptive_gate_scratch_early_hazard_min),
        scratch_eval_success_max=float(args.adaptive_gate_scratch_eval_success_max),
        scratch_eval_return_max=float(args.adaptive_gate_scratch_eval_return_max),
        scratch_eval_hazard_min=float(args.adaptive_gate_scratch_eval_hazard_min),
        transfer_gate_condition_key=str(args.adaptive_gate_transfer_condition),
        transfer_early_success_min=float(args.adaptive_gate_transfer_early_success_min),
        transfer_early_return_min=float(args.adaptive_gate_transfer_early_return_min),
        transfer_early_hazard_max=float(args.adaptive_gate_transfer_early_hazard_max),
        transfer_early_forward_mse_max=float(args.adaptive_gate_transfer_early_forward_mse_max),
        transfer_minus_scratch_early_success_min=float(args.adaptive_gate_transfer_minus_scratch_early_success_min),
        transfer_minus_scratch_early_return_min=float(args.adaptive_gate_transfer_minus_scratch_early_return_min),
        scratch_minus_transfer_early_hazard_max=float(args.adaptive_gate_scratch_minus_transfer_early_hazard_max),
        scratch_minus_transfer_early_forward_mse_max=float(args.adaptive_gate_scratch_minus_transfer_early_forward_mse_max),
        transfer_early_return_slope_min=float(args.adaptive_gate_transfer_early_return_slope_min),
        transfer_early_success_slope_min=float(args.adaptive_gate_transfer_early_success_slope_min),
        transfer_early_hazard_slope_max=float(args.adaptive_gate_transfer_early_hazard_slope_max),
        transfer_early_forward_mse_slope_max=float(args.adaptive_gate_transfer_early_forward_mse_slope_max),
        canary_delta_return_slope_min=float(args.adaptive_gate_canary_delta_return_slope_min),
        canary_delta_success_slope_min=float(args.adaptive_gate_canary_delta_success_slope_min),
        canary_delta_hazard_slope_max=float(args.adaptive_gate_canary_delta_hazard_slope_max),
        canary_delta_forward_mse_slope_max=float(args.adaptive_gate_canary_delta_forward_mse_slope_max),
        source_policy_bank_majority_min=float(args.adaptive_gate_source_policy_bank_majority_min),
        logic=str(args.adaptive_gate_logic),
    )
    if str(args.adaptive_gate_model_json or "").strip():
        model_path = str(args.adaptive_gate_model_json).strip()
        with open(model_path, "r", encoding="utf-8") as f:
            model_payload = json.load(f)
        adaptive_gate_cfg = _adaptive_gate_model_cfg(
            enabled=bool(args.adaptive_gate_enabled or True),
            transfer_gate_condition_key=str(args.adaptive_gate_transfer_condition),
            model_payload=model_payload,
            accept_prob_min=float(args.adaptive_gate_model_accept_prob),
            warmup_prob_min=float(args.adaptive_gate_model_warmup_prob),
        )
        if float(args.adaptive_gate_model_hard_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["hard_prob_min"] = float(args.adaptive_gate_model_hard_prob)
        if float(args.adaptive_gate_model_triage_full_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["triage_full_accept_prob_min"] = float(args.adaptive_gate_model_triage_full_prob)
        if float(args.adaptive_gate_model_triage_warmup_prob) > -1.5:
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["triage_warmup_accept_prob_min"] = float(args.adaptive_gate_model_triage_warmup_prob)
        if bool(args.adaptive_gate_model_enable_canary_triad_override):
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["enable_canary_triad_override"] = True
            adaptive_gate_cfg["model_policy"]["canary_success_weight"] = float(args.adaptive_gate_model_canary_success_weight)
            adaptive_gate_cfg["model_policy"]["canary_return_weight"] = float(args.adaptive_gate_model_canary_return_weight)
            adaptive_gate_cfg["model_policy"]["canary_hazard_weight"] = float(args.adaptive_gate_model_canary_hazard_weight)
            adaptive_gate_cfg["model_policy"]["canary_min_utility"] = float(args.adaptive_gate_model_canary_min_utility)
            adaptive_gate_cfg["model_policy"]["canary_min_hazard_gain"] = float(args.adaptive_gate_model_canary_min_hazard_gain)
            adaptive_gate_cfg["model_policy"]["canary_min_episodes"] = float(args.adaptive_gate_model_canary_min_episodes)
            adaptive_gate_cfg["model_policy"]["canary_two_stage_global_select"] = bool(args.adaptive_gate_model_canary_two_stage_global_select)
            adaptive_gate_cfg["model_policy"]["canary_scratch_utility"] = float(args.adaptive_gate_model_canary_scratch_utility)
            adaptive_gate_cfg["model_policy"]["enable_canary_dual_policy"] = bool(args.adaptive_gate_model_enable_canary_dual_policy)
            adaptive_gate_cfg["model_policy"]["canary_dual_select_hazard_gain_threshold"] = float(args.adaptive_gate_model_canary_dual_hazard_gain_threshold)
            adaptive_gate_cfg["model_policy"]["canary_dual_select_success_delta_threshold"] = float(args.adaptive_gate_model_canary_dual_success_delta_threshold)
            adaptive_gate_cfg["model_policy"]["canary_full_success_floor"] = float(args.adaptive_gate_model_canary_full_success_floor)
            adaptive_gate_cfg["model_policy"]["canary_full_hazard_gain_override"] = float(args.adaptive_gate_model_canary_full_hazard_gain_override)
            if float(args.adaptive_gate_model_canary_safety_success_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_success_weight"] = float(args.adaptive_gate_model_canary_safety_success_weight)
            if float(args.adaptive_gate_model_canary_safety_return_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_return_weight"] = float(args.adaptive_gate_model_canary_safety_return_weight)
            if float(args.adaptive_gate_model_canary_safety_hazard_weight) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_hazard_weight"] = float(args.adaptive_gate_model_canary_safety_hazard_weight)
            if float(args.adaptive_gate_model_canary_safety_min_utility) > -1e8:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_utility"] = float(args.adaptive_gate_model_canary_safety_min_utility)
            if float(args.adaptive_gate_model_canary_safety_min_hazard_gain) > -1e8:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_hazard_gain"] = float(args.adaptive_gate_model_canary_safety_min_hazard_gain)
            if float(args.adaptive_gate_model_canary_safety_min_episodes) > -0.5:
                adaptive_gate_cfg["model_policy"]["canary_safety_min_episodes"] = float(args.adaptive_gate_model_canary_safety_min_episodes)
        if bool(args.adaptive_gate_model_enable_triage):
            if not isinstance(adaptive_gate_cfg.get("model_policy"), dict):
                adaptive_gate_cfg["model_policy"] = {}
            adaptive_gate_cfg["model_policy"]["enable_triage"] = True

    if args.mode == "single":
        if len(profiles) != 1:
            raise ValueError("Single mode requires exactly one profile in --profiles.")
        prof = _profile_params(profile=profiles[0], max_steps=int(args.max_steps))
        pair_policy = _transfer_pair_policy_check(
            source_verse_name=str(prof.get("source_verse_name", "")),
            target_verse_name=str(prof.get("target_verse_name", "")),
            disable_universe_policy=bool(args.disable_universe_policy),
            allow_adjacent_universe_transfer=bool(args.allow_adjacent_universe_transfer),
            allow_cross_universe_transfer=bool(args.allow_cross_universe_transfer),
        )
        if not bool(pair_policy.get("allowed", False)):
            raise ValueError(
                "SF competence transfer pair blocked by universe policy: "
                f"{prof.get('source_verse_name')} -> {prof.get('target_verse_name')} "
                f"({pair_policy.get('relation')}; {pair_policy.get('reason')}). "
                "Use --allow_adjacent_universe_transfer / --allow_cross_universe_transfer, "
                "or --disable_universe_policy to override."
            )
        run = _evaluate_config(
            seeds=seeds,
            profile_params=prof,
            ego_size=int(args.ego_size),
            source_train_episodes=int(args.source_train_episodes),
            target_train_episodes=int(args.target_train_episodes),
            eval_episodes=int(args.eval_episodes),
            max_steps=int(args.max_steps),
            warmup_psi_episodes=int(args.warmup_psi_episodes),
            target_w_estimation_steps=int(args.target_w_estimation_steps),
            source_policy_snapshots=int(args.source_policy_snapshots),
            adaptive_gate=adaptive_gate_cfg,
        )
        artifact = {
            "experiment": "sf_transfer_validation_v2_single",
            "notes": [
                "Ego-grid interface supports grid_world (global-map slice), warehouse_world (lidar approximation), and maze_world (wall-sensor local occupancy).",
                "Successor Features transfer: psi copied from source pretraining; target learns reward weights w.",
                "Auxiliary dynamics objective: one-step next-feature prediction (forward model) is always trained.",
            ],
            "config": {
                "mode": "single",
                "seeds": seeds,
                "profile": profiles[0],
                "profile_description": prof["description"],
                "source_verse_name": prof.get("source_verse_name"),
                "target_verse_name": prof.get("target_verse_name"),
                "run_config": run["config"],
                "source_params": prof["source_params"],
                "target_params": prof["target_params"],
                "transfer_pair_policy": pair_policy,
                "adaptive_gate": adaptive_gate_cfg,
            },
            "summary": run["summary"],
            "per_seed": run["per_seed"],
            "score": run["score"],
        }
        if bool(args.strict_gate):
            gate = _gate_check_from_comparison(
                comparison=artifact["summary"].get("comparison", {}),
                min_return_delta=float(args.gate_min_return_delta),
                min_hazard_gain=float(args.gate_min_hazard_gain),
                min_success_delta=float(args.gate_min_success_delta),
            )
            artifact["strict_gate"] = gate
            if not bool(gate.get("ok", False)):
                print("[gate] FAIL(single):", json.dumps(gate, ensure_ascii=False))
            else:
                print("[gate] PASS(single):", json.dumps(gate, ensure_ascii=False))
        print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    else:
        ego_grid = _parse_int_grid(args.sweep_ego_sizes, default=[3, 5, 7])
        source_grid = _parse_int_grid(args.sweep_source_episodes, default=[80, 120])
        warmup_grid = _parse_int_grid(args.sweep_warmup_episodes, default=[0, 8, 16])

        profile_results: Dict[str, Any] = {}
        for p in profiles:
            prof = _profile_params(profile=p, max_steps=int(args.max_steps))
            pair_policy = _transfer_pair_policy_check(
                source_verse_name=str(prof.get("source_verse_name", "")),
                target_verse_name=str(prof.get("target_verse_name", "")),
                disable_universe_policy=bool(args.disable_universe_policy),
                allow_adjacent_universe_transfer=bool(args.allow_adjacent_universe_transfer),
                allow_cross_universe_transfer=bool(args.allow_cross_universe_transfer),
            )
            if not bool(pair_policy.get("allowed", False)):
                raise ValueError(
                    "SF competence transfer pair blocked by universe policy: "
                    f"{prof.get('source_verse_name')} -> {prof.get('target_verse_name')} "
                    f"({pair_policy.get('relation')}; {pair_policy.get('reason')}). "
                    "Use --allow_adjacent_universe_transfer / --allow_cross_universe_transfer, "
                    "or --disable_universe_policy to override."
                )
            sweep_rows: List[Dict[str, Any]] = []
            for ego_size in ego_grid:
                for src_eps in source_grid:
                    for warmup in warmup_grid:
                        row = _evaluate_config(
                            seeds=seeds,
                            profile_params=prof,
                            ego_size=int(ego_size),
                            source_train_episodes=int(src_eps),
                            target_train_episodes=int(args.target_train_episodes),
                            eval_episodes=int(args.eval_episodes),
                            max_steps=int(args.max_steps),
                            warmup_psi_episodes=int(warmup),
                            target_w_estimation_steps=int(args.target_w_estimation_steps),
                            source_policy_snapshots=int(args.source_policy_snapshots),
                            adaptive_gate=adaptive_gate_cfg,
                        )
                        sweep_rows.append(row)

            ranked = sorted(sweep_rows, key=lambda x: float(x.get("score", -1e9)), reverse=True)
            top_k = max(1, int(args.top_k))
            top_rows = ranked[:top_k]
            table = []
            for r in top_rows:
                cmp = r["summary"]["comparison"]
                table.append(
                    {
                        "score": float(r["score"]),
                        "ego_size": int(r["config"]["ego_size"]),
                        "source_train_episodes": int(r["config"]["source_train_episodes"]),
                        "warmup_psi_episodes": int(r["config"]["warmup_psi_episodes"]),
                        "transfer_warmup_minus_scratch_eval_return": _safe_float(
                            cmp.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0
                        ),
                        "scratch_minus_transfer_warmup_eval_hazard_per_1k": _safe_float(
                            cmp.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0
                        ),
                        "transfer_warmup_minus_scratch_eval_success_rate": _safe_float(
                            cmp.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0
                        ),
                    }
                )

            profile_results[p] = {
                "profile_description": prof["description"],
                "transfer_pair_policy": pair_policy,
                "source_params": prof["source_params"],
                "target_params": prof["target_params"],
                "num_configs": len(sweep_rows),
                "best": ranked[0] if ranked else None,
                "top_table": table,
            }

        global_best: List[Dict[str, Any]] = []
        for p, block in profile_results.items():
            best = block.get("best")
            if isinstance(best, dict):
                global_best.append(
                    {
                        "profile": p,
                        "score": float(best.get("score", 0.0)),
                        "config": best.get("config", {}),
                        "comparison": (best.get("summary", {}) or {}).get("comparison", {}),
                    }
                )
        global_best = sorted(global_best, key=lambda x: float(x.get("score", -1e9)), reverse=True)

        artifact = {
            "experiment": "sf_transfer_validation_v2_phase2",
            "notes": [
                "Phase 2 adds multi-profile validation and hyperparameter sweep.",
                "Profiles can target warehouse or maze transfer pairs depending on --profiles.",
                "Sweep axes: ego_grid size, source SF pretrain episodes, transfer warmup freeze episodes.",
            ],
            "config": {
                "mode": "phase2",
                "seeds": seeds,
                "profiles": profiles,
                "sweep_ego_sizes": ego_grid,
                "sweep_source_episodes": source_grid,
                "sweep_warmup_episodes": warmup_grid,
                "target_train_episodes": int(args.target_train_episodes),
                "eval_episodes": int(args.eval_episodes),
                "max_steps": int(args.max_steps),
                "target_w_estimation_steps": int(args.target_w_estimation_steps),
                "source_policy_snapshots": int(args.source_policy_snapshots),
                "adaptive_gate": adaptive_gate_cfg,
                "universe_policy": {
                    "enabled": not bool(args.disable_universe_policy),
                    "allow_adjacent_universe_transfer": bool(args.allow_adjacent_universe_transfer),
                    "allow_cross_universe_transfer": bool(args.allow_cross_universe_transfer),
                },
                "top_k": int(args.top_k),
            },
            "phase2": {
                "by_profile": profile_results,
                "global_best": global_best,
            },
        }
        if bool(args.strict_gate):
            gate_by_profile: Dict[str, Any] = {}
            all_ok = True
            for p, block in profile_results.items():
                best = block.get("best")
                cmp = (
                    ((best.get("summary", {}) if isinstance(best, dict) else {}) or {}).get("comparison", {})
                    if isinstance(best, dict)
                    else {}
                )
                gate = _gate_check_from_comparison(
                    comparison=cmp if isinstance(cmp, dict) else {},
                    min_return_delta=float(args.gate_min_return_delta),
                    min_hazard_gain=float(args.gate_min_hazard_gain),
                    min_success_delta=float(args.gate_min_success_delta),
                )
                gate_by_profile[p] = gate
                if not bool(gate.get("ok", False)):
                    all_ok = False
            artifact["strict_gate"] = {
                "ok": bool(all_ok),
                "by_profile": gate_by_profile,
            }
            print("[gate] phase2:", json.dumps(artifact["strict_gate"], ensure_ascii=False))
            if not all_ok:
                out_path = str(args.out_json)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(artifact, f, ensure_ascii=False, indent=2)
                print(f"[ok] wrote: {out_path}")
                raise SystemExit(2)
        for p, block in profile_results.items():
            print(f"[profile={p}] top configs:")
            print(json.dumps(block.get("top_table", []), ensure_ascii=False, indent=2))

    out_path = str(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {out_path}")

    if bool(args.strict_gate) and args.mode == "single":
        gate = artifact.get("strict_gate", {})
        if isinstance(gate, dict) and not bool(gate.get("ok", False)):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
