"""
agents/q_agent.py

Tabular Q learning agent for discrete actions.

Works best for small environments where observations can be keyed reliably.
Uses epsilon greedy exploration and updates from ExperienceBatch transitions.

Requirements:
- action_space.type == "discrete"
- obs must be JSON serializable
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import AgentSpec, JSONValue, SpaceSpec
from core.agent_base import ActionResult, ExperienceBatch
from memory.sample_weighter import ReplayWeightConfig, compute_sample_weight


def obs_key(
    obs: JSONValue,
    *,
    warehouse_mode: str = "direction_only",
    grid_mode: str = "full",
) -> str:
    if isinstance(obs, dict):
        wk = _warehouse_obs_key(obs, mode=warehouse_mode)
        if wk is not None:
            return json.dumps(wk, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        gk = _grid_obs_key(obs, mode=grid_mode)
        if gk is not None:
            return json.dumps(gk, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    if isinstance(x, (int, float)):
        return bool(x)
    return default


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


def _sgn(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _warehouse_obs_key(obs: Dict[str, Any], *, mode: str = "direction_only") -> Optional[Dict[str, Any]]:
    # Stable warehouse keying mode used by default transfer runs.
    # Keep action-relevant direction while dropping brittle absolute/sensor fields.
    needed = {"x", "y", "goal_x", "goal_y", "battery"}
    if not needed.issubset(set(str(k) for k in obs.keys())):
        return None
    x = _safe_int(obs.get("x", 0), 0)
    y = _safe_int(obs.get("y", 0), 0)
    gx = _safe_int(obs.get("goal_x", 0), 0)
    gy = _safe_int(obs.get("goal_y", 0), 0)
    dx = gx - x
    dy = gy - y
    _ = str(mode or "direction_only").strip().lower()
    return {
        "_schema": "warehouse_key_direction_only_v1",
        "dx_dir": int(_sgn(dx)),
        "dy_dir": int(_sgn(dy)),
    }


def _grid_obs_key(obs: Dict[str, Any], *, mode: str = "full") -> Optional[Dict[str, Any]]:
    mode_n = str(mode or "full").strip().lower()
    if mode_n in {"", "full", "raw"}:
        return None
    needed = {"x", "y", "goal_x", "goal_y"}
    if not needed.issubset(set(str(k) for k in obs.keys())):
        return None
    x = _safe_int(obs.get("x", 0), 0)
    y = _safe_int(obs.get("y", 0), 0)
    gx = _safe_int(obs.get("goal_x", 0), 0)
    gy = _safe_int(obs.get("goal_y", 0), 0)
    if mode_n in {"xy_goal", "position_goal"}:
        return {
            "_schema": "grid_key_xy_goal_v1",
            "x": int(x),
            "y": int(y),
            "goal_x": int(gx),
            "goal_y": int(gy),
        }
    if mode_n in {"direction_only", "delta_dir"}:
        return {
            "_schema": "grid_key_direction_only_v1",
            "dx_dir": int(_sgn(gx - x)),
            "dy_dir": int(_sgn(gy - y)),
        }
    return None


@dataclass
class QStats:
    steps_seen: int = 0
    updates: int = 0
    epsilon: float = 1.0


class QLearningAgent:
    """
    Q(s,a) stored in a dict: state_key -> np.ndarray shape (n_actions,)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        if action_space.type != "discrete" or not action_space.n:
            raise ValueError("QLearningAgent requires discrete action space with n")

        cfg = spec.config if isinstance(spec.config, dict) else {}

        self.lr = float(cfg.get("lr", 0.1))
        self.gamma = float(cfg.get("gamma", 0.99))

        self.epsilon_start = float(cfg.get("epsilon_start", 1.0))
        self.epsilon_min = float(cfg.get("epsilon_min", 0.05))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 0.995))
        self.learn_success_bonus = float(cfg.get("learn_success_bonus", 0.0))
        self.learn_hazard_penalty = float(cfg.get("learn_hazard_penalty", 0.0))
        self._diag_temperature = max(1e-6, float(cfg.get("diag_temperature", 1.0)))
        self.dynamic_transfer_mix_enabled = _safe_bool(cfg.get("dynamic_transfer_mix_enabled", False), False)
        self.transfer_mix_start = max(0.0, min(1.0, float(cfg.get("transfer_mix_start", 1.0))))
        self.transfer_mix_end = max(0.0, min(1.0, float(cfg.get("transfer_mix_end", 0.0))))
        self.transfer_mix_decay_steps = max(1, int(cfg.get("transfer_mix_decay_steps", 10000)))
        self.transfer_mix_min_rows = max(1, int(cfg.get("transfer_mix_min_rows", 32)))
        self.transfer_replay_reward_scale = max(0.0, float(cfg.get("transfer_replay_reward_scale", 0.5)))
        self.warehouse_obs_key_mode = str(cfg.get("warehouse_obs_key_mode", "direction_only")).strip().lower()
        self.grid_obs_key_mode = str(cfg.get("grid_obs_key_mode", "full")).strip().lower()
        self.warmstart_use_transfer_score = _safe_bool(cfg.get("warmstart_use_transfer_score", False), False)
        self.warmstart_target = str(cfg.get("warmstart_target", "immediate")).strip().lower()
        self.warmstart_target_gamma = max(
            0.0,
            min(1.0, float(_safe_float(cfg.get("warmstart_target_gamma", self.gamma), self.gamma))),
        )
        self.warmstart_transfer_score_min = max(
            0.0, float(_safe_float(cfg.get("warmstart_transfer_score_min", 0.0), 0.0))
        )
        self.warmstart_transfer_score_max = max(
            self.warmstart_transfer_score_min,
            float(_safe_float(cfg.get("warmstart_transfer_score_max", 2.0), 2.0)),
        )
        self._transfer_rows: List[Tuple[str, int, float, str, bool, bool, float]] = []

        self.n_actions = int(action_space.n)

        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()

        self.q: Dict[str, np.ndarray] = {}
        self._warmstart_rows: int = 0
        replay_cfg = cfg.get("replay.weighting")
        if not isinstance(replay_cfg, dict):
            replay_cfg = cfg.get("replay_weighting")
        self._replay_weight_cfg = ReplayWeightConfig.from_dict(replay_cfg)

        self.stats = QStats(epsilon=self.epsilon_start)

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(None if seed is None else int(seed))

    def act(self, obs: JSONValue) -> ActionResult:
        k = self._obs_key(obs)
        qvals = self._get_q(k)
        greedy_action = int(np.argmax(qvals))

        if self._rng.random() < self.stats.epsilon:
            a = int(self._rng.integers(0, self.n_actions))
            return ActionResult(
                action=a,
                info={
                    "mode": "explore",
                    "epsilon": self.stats.epsilon,
                    "greedy_action": int(greedy_action),
                    "action_matches_greedy": bool(int(a) == int(greedy_action)),
                },
            )

        a = int(greedy_action)
        return ActionResult(
            action=a,
            info={
                "mode": "exploit",
                "epsilon": self.stats.epsilon,
                "greedy_action": int(greedy_action),
                "action_matches_greedy": True,
            },
        )

    def action_diagnostics(self, obs: JSONValue) -> Dict[str, JSONValue]:
        k = self._obs_key(obs)
        qvals = self._get_q(k)
        if qvals.size <= 0:
            return {"sample_probs": [], "danger_scores": [], "q_values": []}
        centered = qvals - float(np.max(qvals))
        scaled = centered / float(self._diag_temperature)
        ex = np.exp(scaled)
        denom = float(np.sum(ex))
        if not np.isfinite(denom) or denom <= 0.0:
            probs = np.full((self.n_actions,), 1.0 / float(max(1, self.n_actions)), dtype=np.float32)
        else:
            probs = ex / denom
        danger = np.clip(1.0 - probs, 0.0, 1.0)
        return {
            "sample_probs": [float(x) for x in probs.tolist()],
            "danger_scores": [float(x) for x in danger.tolist()],
            "q_values": [float(x) for x in qvals.tolist()],
            "epsilon": float(self.stats.epsilon),
        }

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}

        pre_online_steps = int(self.stats.steps_seen)
        updates = 0
        shaped_updates = 0
        native_td_abs: List[float] = []
        transfer_td_abs: List[float] = []
        transfer_td_score_pairs: List[Tuple[float, float]] = []

        for tr in batch.transitions:
            s = self._obs_key(tr.obs)
            a = int(tr.action)
            r = float(self._learning_reward(tr))
            sp = self._obs_key(tr.next_obs)
            done = bool(tr.done or tr.truncated)

            q_s = self._get_q(s)
            q_sp = self._get_q(sp)

            target = r if done else r + self.gamma * float(np.max(q_sp))
            td = target - float(q_s[a])
            q_s[a] = float(q_s[a]) + self.lr * td
            native_td_abs.append(abs(float(td)))

            updates += 1
            self.stats.steps_seen += 1
            if r != float(tr.reward):
                shaped_updates += 1

        transfer_mix_ratio = self._transfer_mix_ratio(steps_seen=pre_online_steps)
        transfer_updates = 0
        if (
            bool(self.dynamic_transfer_mix_enabled)
            and len(self._transfer_rows) >= int(self.transfer_mix_min_rows)
            and len(batch.transitions) > 0
            and float(transfer_mix_ratio) > 0.0
        ):
            target = int(round(float(transfer_mix_ratio) * float(len(batch.transitions))))
            n_offline = max(0, target)
            for _ in range(n_offline):
                idx = int(self._rng.integers(0, len(self._transfer_rows)))
                s, a, r, sp, done, truncated, tscore = self._transfer_rows[idx]
                q_s = self._get_q(s)
                q_sp = self._get_q(sp)
                rew = float(r) * float(self.transfer_replay_reward_scale)
                terminal = bool(done or truncated)
                tgt = rew if terminal else rew + self.gamma * float(np.max(q_sp))
                td = tgt - float(q_s[a])
                q_s[a] = float(q_s[a]) + self.lr * td
                td_abs = abs(float(td))
                transfer_td_abs.append(td_abs)
                transfer_td_score_pairs.append((float(tscore), float(td_abs)))
                transfer_updates += 1

        self.stats.updates += updates
        self.stats.epsilon = max(self.epsilon_min, self.stats.epsilon * self.epsilon_decay)

        return {
            "updates": updates,
            "steps_seen": self.stats.steps_seen,
            "epsilon": self.stats.epsilon,
            "q_states": len(self.q),
            "warmstart_rows": int(self._warmstart_rows),
            "shaped_updates": int(shaped_updates),
            "transfer_mix_ratio": float(transfer_mix_ratio),
            "transfer_replay_updates": int(transfer_updates),
            "transfer_replay_rows": int(len(self._transfer_rows)),
            "native_td_abs_mean": (
                float(sum(native_td_abs) / float(max(1, len(native_td_abs)))) if native_td_abs else None
            ),
            "native_td_abs_p90": self._percentile(native_td_abs, 0.90),
            "transfer_td_abs_mean": (
                float(sum(transfer_td_abs) / float(max(1, len(transfer_td_abs)))) if transfer_td_abs else None
            ),
            "transfer_td_abs_p90": self._percentile(transfer_td_abs, 0.90),
            "transfer_td_score_corr": self._corr(transfer_td_score_pairs),
        }

    def learn_from_dataset(self, dataset_path: str, *, reward_scale: Optional[float] = None) -> Dict[str, JSONValue]:
        """
        Warm-start Q-table from offline transitions.

        For each (obs, action), adjust Q(s,a) using:
          Q(s,a) += reward_scale * warmstart_target * sample_weight

        `warmstart_target` is configured by `warmstart_target`:
        - `immediate` (default): raw step reward (legacy behavior)
        - `return_to_go` / `rtg`: discounted episode return-to-go
        """
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        cfg = self.spec.config if isinstance(self.spec.config, dict) else {}
        scale = float(reward_scale if reward_scale is not None else cfg.get("warmstart_reward_scale", 0.5))
        scale = max(0.0, float(scale))
        target_mode = str(self.warmstart_target).strip().lower()
        target_gamma = float(self.warmstart_target_gamma)
        rows_raw = self._load_warmstart_rows(dataset_path)
        warmstart_target_rows = int(
            self._assign_warmstart_targets(rows_raw, mode=target_mode, gamma=target_gamma)
        )

        rows = 0
        transfer_score_rows = 0
        transfer_score_sum = 0.0
        for row in rows_raw:
            a = int(row["action"])
            obs = row["obs"]
            next_obs = row["next_obs"]
            done = bool(row["done"])
            truncated = bool(row["truncated"])
            k = self._obs_key(obs)
            kp = self._obs_key(next_obs)
            qvals = self._get_q(k)
            r = float(row["reward"])
            sample_w = float(compute_sample_weight(row["source"], cfg=self._replay_weight_cfg))
            target_val = float(row.get("warmstart_target", r))
            if np.isfinite(target_val):
                bonus = float(scale * target_val)
            else:
                bonus = 0.0
            score_w = 1.0
            if bool(self.warmstart_use_transfer_score):
                raw_score = _safe_float(row["source"].get("transfer_score", 1.0), 1.0)
                score_w = max(
                    float(self.warmstart_transfer_score_min),
                    min(float(self.warmstart_transfer_score_max), float(raw_score)),
                )
                transfer_score_rows += 1
                transfer_score_sum += float(score_w)
            qvals[a] = float(qvals[a]) + float(bonus) * float(sample_w) * float(score_w)
            self._transfer_rows.append(
                (k, int(a), float(r), kp, bool(done), bool(truncated), float(score_w))
            )
            rows += 1

        self._warmstart_rows += int(rows)
        return {
            "warmstart_rows_added": int(rows),
            "q_states": int(len(self.q)),
            "transfer_rows_loaded": int(len(self._transfer_rows)),
            "warmstart_target_mode": str(target_mode),
            "warmstart_target_gamma": float(target_gamma),
            "warmstart_target_rows": int(warmstart_target_rows),
            "warmstart_transfer_score_weighted": bool(self.warmstart_use_transfer_score),
            "warmstart_transfer_score_rows": int(transfer_score_rows),
            "warmstart_transfer_score_mean": (
                float(transfer_score_sum / float(max(1, transfer_score_rows))) if transfer_score_rows > 0 else None
            ),
        }

    def _load_warmstart_rows(self, dataset_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                try:
                    src = json.loads(s)
                except Exception:
                    continue
                if not isinstance(src, dict):
                    continue
                if "obs" not in src or "action" not in src:
                    continue
                try:
                    a = int(src.get("action"))
                except Exception:
                    continue
                if a < 0 or a >= int(self.n_actions):
                    continue
                obs = src.get("obs")
                next_obs = src.get("next_obs", obs)
                reward = float(_safe_float(src.get("reward", 0.1), 0.1))
                done = bool(src.get("done", False))
                truncated = bool(src.get("truncated", False))
                step_idx = _safe_int(src.get("step_idx", line_idx), line_idx)
                episode_id = src.get("episode_id")
                rows.append(
                    {
                        "source": src,
                        "line_idx": int(line_idx),
                        "episode_id": (None if episode_id is None else str(episode_id)),
                        "step_idx": int(step_idx),
                        "obs": obs,
                        "next_obs": next_obs,
                        "action": int(a),
                        "reward": float(reward),
                        "done": bool(done),
                        "truncated": bool(truncated),
                        "warmstart_target": float(reward),
                    }
                )
        return rows

    def _assign_warmstart_targets(self, rows: List[Dict[str, Any]], *, mode: str, gamma: float) -> int:
        mode_n = str(mode or "immediate").strip().lower()
        if mode_n not in {"return_to_go", "rtg", "mc_return", "monte_carlo"}:
            for row in rows:
                row["warmstart_target"] = float(_safe_float(row.get("reward", 0.0), 0.0))
            return int(len(rows))

        by_episode: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            ep = row.get("episode_id")
            if not ep:
                row["warmstart_target"] = float(_safe_float(row.get("reward", 0.0), 0.0))
                continue
            key = str(ep)
            if key not in by_episode:
                by_episode[key] = []
            by_episode[key].append(row)

        for ep_rows in by_episode.values():
            ep_rows.sort(key=lambda r: (int(_safe_int(r.get("step_idx", 0), 0)), int(_safe_int(r.get("line_idx", 0), 0))))
            running = 0.0
            for row in reversed(ep_rows):
                rew = float(_safe_float(row.get("reward", 0.0), 0.0))
                if bool(row.get("done", False) or row.get("truncated", False)):
                    running = rew
                else:
                    running = rew + float(gamma) * float(running)
                row["warmstart_target"] = float(running)
        return int(len(rows))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        payload = {
            "spec": self.spec.to_dict(),
            "n_actions": self.n_actions,
            "stats": {
                "steps_seen": self.stats.steps_seen,
                "updates": self.stats.updates,
                "epsilon": self.stats.epsilon,
                "warmstart_rows": int(self._warmstart_rows),
                "transfer_rows": int(len(self._transfer_rows)),
                "dynamic_transfer_mix_enabled": bool(self.dynamic_transfer_mix_enabled),
                "transfer_mix_start": float(self.transfer_mix_start),
                "transfer_mix_end": float(self.transfer_mix_end),
                "transfer_mix_decay_steps": int(self.transfer_mix_decay_steps),
                "transfer_mix_min_rows": int(self.transfer_mix_min_rows),
                "transfer_replay_reward_scale": float(self.transfer_replay_reward_scale),
            },
            "q": {k: v.tolist() for k, v in self.q.items()},
        }

        with open(os.path.join(path, "q_table.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        fp = os.path.join(path, "q_table.json")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"model file not found: {fp}")

        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.n_actions = int(payload.get("n_actions", self.n_actions))

        st = payload.get("stats", {}) or {}
        self.stats.steps_seen = int(st.get("steps_seen", 0))
        self.stats.updates = int(st.get("updates", 0))
        self.stats.epsilon = float(st.get("epsilon", self.epsilon_start))
        self._warmstart_rows = int(st.get("warmstart_rows", 0))

        q_raw = payload.get("q", {}) or {}
        self.q = {k: np.asarray(v, dtype=np.float32) for k, v in q_raw.items()}

    def close(self) -> None:
        return

    def _obs_key(self, obs: JSONValue) -> str:
        return obs_key(
            obs,
            warehouse_mode=self.warehouse_obs_key_mode,
            grid_mode=self.grid_obs_key_mode,
        )

    def _percentile(self, values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        try:
            arr = np.asarray(values, dtype=np.float32)
            return float(np.quantile(arr, float(q)))
        except Exception:
            return None

    def _corr(self, pairs: List[Tuple[float, float]]) -> Optional[float]:
        if len(pairs) < 2:
            return None
        xs = np.asarray([float(p[0]) for p in pairs], dtype=np.float32)
        ys = np.asarray([float(p[1]) for p in pairs], dtype=np.float32)
        xstd = float(np.std(xs))
        ystd = float(np.std(ys))
        if xstd <= 1e-12 or ystd <= 1e-12:
            return None
        try:
            corr = float(np.corrcoef(xs, ys)[0, 1])
        except Exception:
            return None
        if not np.isfinite(corr):
            return None
        return float(corr)

    def _transfer_mix_ratio(self, *, steps_seen: Optional[int] = None) -> float:
        if not bool(self.dynamic_transfer_mix_enabled):
            return 0.0
        start = float(self.transfer_mix_start)
        end = float(self.transfer_mix_end)
        if self.transfer_mix_decay_steps <= 0:
            return max(0.0, min(1.0, end))
        cur_steps = self.stats.steps_seen if steps_seen is None else int(steps_seen)
        progress = max(0.0, min(1.0, float(cur_steps) / float(self.transfer_mix_decay_steps)))
        cur = start + (end - start) * progress
        return max(0.0, min(1.0, cur))

    def _get_q(self, k: str) -> np.ndarray:
        if k not in self.q:
            self.q[k] = np.zeros((self.n_actions,), dtype=np.float32)
        return self.q[k]

    def _learning_reward(self, tr: Any) -> float:
        reward = float(tr.reward)
        if self.learn_success_bonus == 0.0 and self.learn_hazard_penalty == 0.0:
            return reward

        env_info = {}
        info = tr.info if isinstance(tr.info, dict) else {}
        if isinstance(info.get("env_info"), dict):
            env_info = info.get("env_info") or {}
        elif info:
            env_info = info

        if self.learn_success_bonus > 0.0 and bool(env_info.get("reached_goal", False)):
            reward += float(self.learn_success_bonus)

        if self.learn_hazard_penalty > 0.0:
            hazard_flags = (
                "fell_cliff",
                "fell_pit",
                "hit_laser",
                "battery_depleted",
                "hit_wall",
                "hit_obstacle",
                "battery_death",
            )
            if any(bool(env_info.get(k, False)) for k in hazard_flags):
                reward -= float(self.learn_hazard_penalty)

        return float(reward)
