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
        if -self.radius <= dgx <= self.radius and -self.radius <= dgy <= self.radius:
            ego.goal[dgy + self.radius, dgx + self.radius] = 1
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
        if -self.radius <= dgx <= self.radius and -self.radius <= dgy <= self.radius:
            ego.goal[dgy + self.radius, dgx + self.radius] = 1
        return ego

    def extract(self, *, verse_name: str, verse: Any, obs: Dict[str, Any]) -> EgoObservation:
        v = str(verse_name).strip().lower()
        if v == "grid_world":
            return self.from_grid_world(verse, obs)
        if v == "warehouse_world":
            return self.from_warehouse_world(obs)
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

    def copy_psi_from(self, src: "TabularSFAgent") -> None:
        copy_actions = min(self.n_actions, src.n_actions)
        for k, arr in src.psi_table.items():
            dst = np.zeros((self.n_actions, self.feature_dim), dtype=np.float32)
            dst[:copy_actions, :] = arr[:copy_actions, :]
            self.psi_table[k] = dst

    def set_w(self, w: np.ndarray) -> None:
        self.w = np.asarray(w, dtype=np.float32).copy()

    def _psi_state(self, key: str) -> np.ndarray:
        arr = self.psi_table.get(key)
        if arr is None:
            arr = np.zeros((self.n_actions, self.feature_dim), dtype=np.float32)
            self.psi_table[key] = arr
        return arr

    def q_values(self, key: str) -> np.ndarray:
        psi = self._psi_state(key)
        return psi @ self.w

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

        if done:
            target_vec = phi_s
        else:
            q_sp = psi_sp @ self.w
            a_next = max(self.allowed_actions, key=lambda idx: float(q_sp[idx]))
            target_vec = phi_s + self.gamma * psi_sp[a_next]

        td_vec = target_vec - psi_s[a]
        if learn_psi:
            psi_s[a] = psi_s[a] + self.psi_lr * td_vec
            psi_s[a] = np.clip(psi_s[a], -200.0, 200.0)

        pred_r = float(np.dot(self.w, phi_s))
        err_r = float(reward) - pred_r
        if learn_w:
            self.w = self.w + self.w_lr * err_r * phi_s
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
    keys = ["hit_wall", "hit_obstacle", "hit_patrol", "battery_death"]
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


def _train_then_eval(
    *,
    seed: int,
    adapter: EgoGridAdapter,
    source_params: Dict[str, Any],
    target_params: Dict[str, Any],
    source_train_episodes: int,
    target_train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    warmup_psi_episodes: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    np.random.seed(int(seed))

    # 1) Source SF pretraining on shared navigation dynamics.
    src_verse = _build_verse(verse_name="grid_world", seed=int(seed), params=source_params)
    probe = src_verse.reset()
    probe_obs = probe.obs if isinstance(probe.obs, dict) else {}
    probe_ego = adapter.extract(verse_name="grid_world", verse=src_verse, obs=probe_obs)
    feat_dim = int(adapter.phi(probe_ego).shape[0])

    src_agent = TabularSFAgent(
        n_actions=4,
        feature_dim=feat_dim,
        gamma=0.97,
        psi_lr=0.24,
        w_lr=0.06,
        fwd_lr=0.02,
        allowed_actions=[0, 1, 2, 3],
    )
    src_agent.set_w(_semantic_reward_weights(feature_dim=feat_dim, grid_size=adapter.size))
    src_train_stats: List[EpisodeStats] = []
    for ep in range(int(source_train_episodes)):
        eps = _epsilon_linear(ep, source_train_episodes, start=0.40, end=0.05)
        src_train_stats.append(
            _run_episode(
                verse=src_verse,
                verse_name="grid_world",
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
    src_verse.close()
    source_summary = _summarize(src_train_stats)

    # 2) Target conditions.
    def _make_target_agent(*, transferred: bool, freeze_psi_episodes: int) -> Dict[str, Any]:
        trg_verse = _build_verse(verse_name="warehouse_world", seed=int(seed), params=target_params)
        ag = TabularSFAgent(
            n_actions=5,
            feature_dim=feat_dim,
            gamma=0.97,
            psi_lr=0.20,
            w_lr=0.07,
            fwd_lr=0.02,
            allowed_actions=[0, 1, 2, 3],  # shared nav actions
        )
        if transferred:
            ag.copy_psi_from(src_agent)
            ag.set_w(_semantic_reward_weights(feature_dim=feat_dim, grid_size=adapter.size))

        # Zero-shot snapshot before any target updates.
        zero_shot: Optional[Dict[str, Any]] = None
        if transferred:
            eval_stats = []
            for _ in range(int(eval_episodes)):
                eval_stats.append(
                    _run_episode(
                        verse=trg_verse,
                        verse_name="warehouse_world",
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
                    verse_name="warehouse_world",
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
                    verse_name="warehouse_world",
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
        return {
            "zero_shot_eval": zero_shot,
            "train_summary": _summarize(train_stats),
            "early_train_summary": early_summary,
            "eval_summary": _summarize(eval_stats_post),
        }

    scratch = _make_target_agent(transferred=False, freeze_psi_episodes=0)
    transfer = _make_target_agent(transferred=True, freeze_psi_episodes=0)
    transfer_warmup = _make_target_agent(
        transferred=True,
        freeze_psi_episodes=max(0, int(warmup_psi_episodes)),
    )

    return {
        "seed": int(seed),
        "source_pretrain": source_summary,
        "target_conditions": {
            "sf_scratch": scratch,
            "sf_transfer": transfer,
            "sf_transfer_warmup": transfer_warmup,
        },
    }


def _aggregate_seed_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        out[cond] = {
            "mean_eval_return": _mean(_collect(("target_conditions", cond, "eval_summary", "mean_return"))),
            "mean_eval_success_rate": _mean(_collect(("target_conditions", cond, "eval_summary", "success_rate"))),
            "mean_eval_hazard_per_1k": _mean(_collect(("target_conditions", cond, "eval_summary", "hazard_per_1k"))),
            "mean_early_return": _mean(_collect(("target_conditions", cond, "early_train_summary", "mean_return"))),
            "mean_early_hazard_per_1k": _mean(
                _collect(("target_conditions", cond, "early_train_summary", "hazard_per_1k"))
            ),
            "mean_eval_forward_mse": _mean(_collect(("target_conditions", cond, "eval_summary", "mean_forward_mse"))),
        }

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
    }
    return out


def _profile_params(*, profile: str, max_steps: int) -> Dict[str, Any]:
    p = str(profile).strip().lower()
    if p == "near_transfer":
        return {
            "profile": p,
            "description": "Navigation-isolated warehouse profile (near-transfer).",
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
    raise ValueError(f"Unknown profile: {profile}. Expected near_transfer or default_like.")


def _phase2_score(summary: Dict[str, Any]) -> float:
    cmp = summary.get("comparison", {}) if isinstance(summary.get("comparison"), dict) else {}
    ret_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_return", 0.0), 0.0)
    haz_gain = _safe_float(cmp.get("scratch_minus_transfer_warmup_eval_hazard_per_1k", 0.0), 0.0)
    sr_delta = _safe_float(cmp.get("transfer_warmup_minus_scratch_eval_success_rate", 0.0), 0.0)
    # Return remains primary; hazard and success are tie-break style bonuses.
    return float(ret_delta + 0.02 * haz_gain + 20.0 * sr_delta)


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
) -> Dict[str, Any]:
    adapter = EgoGridAdapter(size=int(ego_size))
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        row = _train_then_eval(
            seed=int(s),
            adapter=adapter,
            source_params=dict(profile_params["source_params"]),
            target_params=dict(profile_params["target_params"]),
            source_train_episodes=int(source_train_episodes),
            target_train_episodes=int(target_train_episodes),
            eval_episodes=int(eval_episodes),
            max_steps=int(max_steps),
            warmup_psi_episodes=int(warmup_psi_episodes),
        )
        per_seed.append(row)

    summary = _aggregate_seed_block(per_seed)
    out = {
        "config": {
            "ego_size": int(ego_size),
            "source_train_episodes": int(source_train_episodes),
            "target_train_episodes": int(target_train_episodes),
            "eval_episodes": int(eval_episodes),
            "warmup_psi_episodes": int(warmup_psi_episodes),
            "max_steps": int(max_steps),
        },
        "summary": summary,
        "score": _phase2_score(summary),
        "per_seed": per_seed,
    }
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
    ap.add_argument("--sweep_ego_sizes", type=str, default="3,5,7")
    ap.add_argument("--sweep_source_episodes", type=str, default="80,120")
    ap.add_argument("--sweep_warmup_episodes", type=str, default="0,8,16")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--strict_gate", action="store_true")
    ap.add_argument("--gate_min_return_delta", type=float, default=0.0)
    ap.add_argument("--gate_min_hazard_gain", type=float, default=0.0)
    ap.add_argument("--gate_min_success_delta", type=float, default=0.0)
    ap.add_argument("--out_json", type=str, default="models/validation/sf_transfer_validation_v2_phase2.json")
    args = ap.parse_args()

    register_builtin()
    seeds = _parse_seed_list(args.seeds)
    profiles = _parse_str_list(args.profiles, default=["near_transfer", "default_like"])

    if args.mode == "single":
        if len(profiles) != 1:
            raise ValueError("Single mode requires exactly one profile in --profiles.")
        prof = _profile_params(profile=profiles[0], max_steps=int(args.max_steps))
        run = _evaluate_config(
            seeds=seeds,
            profile_params=prof,
            ego_size=int(args.ego_size),
            source_train_episodes=int(args.source_train_episodes),
            target_train_episodes=int(args.target_train_episodes),
            eval_episodes=int(args.eval_episodes),
            max_steps=int(args.max_steps),
            warmup_psi_episodes=int(args.warmup_psi_episodes),
        )
        artifact = {
            "experiment": "sf_transfer_validation_v2_single",
            "notes": [
                "Ego-grid interface: grid_world uses global-map slice, warehouse_world uses lidar-ray approximation.",
                "Successor Features transfer: psi copied from source pretraining; target learns reward weights w.",
                "Auxiliary dynamics objective: one-step next-feature prediction (forward model) is always trained.",
            ],
            "config": {
                "mode": "single",
                "seeds": seeds,
                "profile": profiles[0],
                "profile_description": prof["description"],
                "run_config": run["config"],
                "source_params": prof["source_params"],
                "target_params": prof["target_params"],
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
                "Profiles include near-transfer and default-like warehouse dynamics.",
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
