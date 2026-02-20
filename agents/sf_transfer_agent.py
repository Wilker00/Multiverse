"""
agents/sf_transfer_agent.py

Feature-flagged tabular Successor Feature (SF) agent.

This agent expects (or synthesizes) an egocentric local occupancy grid `ego_grid`.
It learns:
  - psi(s,a): successor features (dynamics)
  - w: task preference vector (reward weights)
  - a lightweight forward model for next-feature prediction (auxiliary dynamics loss)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.agent_base import ActionResult, ExperienceBatch
from core.types import AgentSpec, JSONValue, SpaceSpec
from memory.semantic_bridge import task_embedding_weights


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
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


@dataclass
class SFStats:
    steps_seen: int = 0
    updates: int = 0
    epsilon: float = 0.30
    feature_dim: int = 0
    state_rows: int = 0


class SuccessorFeatureAgent:
    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space
        if action_space.type != "discrete" or not action_space.n:
            raise ValueError("SuccessorFeatureAgent requires discrete action space with n")
        self.n_actions = int(action_space.n)

        cfg = spec.config if isinstance(spec.config, dict) else {}
        self.gamma = float(cfg.get("gamma", 0.97))
        self.psi_lr = float(cfg.get("psi_lr", 0.20))
        self.w_lr = float(cfg.get("w_lr", 0.06))
        self.fwd_lr = float(cfg.get("fwd_lr", 0.02))
        self.epsilon_start = float(cfg.get("epsilon_start", 0.30))
        self.epsilon_min = float(cfg.get("epsilon_min", 0.03))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 0.997))
        self.ego_grid_size = max(3, int(cfg.get("ego_grid_size", 5)))
        if self.ego_grid_size % 2 == 0:
            self.ego_grid_size += 1

        self.verse_name = str(cfg.get("verse_name", "")).strip().lower()
        self.use_semantic_w_init = _safe_bool(cfg.get("use_semantic_w_init", True), True)
        self.semantic_profile = str(cfg.get("semantic_profile", "balanced")).strip().lower()

        allowed = cfg.get("allowed_actions")
        if isinstance(allowed, list) and allowed:
            parsed = [int(a) for a in allowed if 0 <= int(a) < self.n_actions]
            self.allowed_actions = parsed if parsed else list(range(self.n_actions))
        else:
            self.allowed_actions = list(range(self.n_actions))

        self._rng = np.random.default_rng(spec.seed)

        self._feature_dim: int = 0
        self._grid_cells: int = int(self.ego_grid_size * self.ego_grid_size)
        self._occ_slice: Tuple[int, int] = (1, 1 + self._grid_cells)
        self._goal_slice: Tuple[int, int] = (1 + self._grid_cells, 1 + 2 * self._grid_cells)
        self._extra_slice: Tuple[int, int] = (1 + 2 * self._grid_cells, 1 + 2 * self._grid_cells + 4)

        self.psi_table: Dict[str, np.ndarray] = {}
        self.w: Optional[np.ndarray] = None
        self.forward_model: Optional[np.ndarray] = None

        self.stats = SFStats(epsilon=float(self.epsilon_start))

    def seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(None if seed is None else int(seed))

    def _normalize_grid_shape(self, grid: np.ndarray) -> np.ndarray:
        k = int(self.ego_grid_size)
        out = np.zeros((k, k), dtype=np.int32)
        if grid.size <= 0:
            return out
        h, w = int(grid.shape[0]), int(grid.shape[1])
        src_y0 = max(0, (h - k) // 2)
        src_x0 = max(0, (w - k) // 2)
        dst_y0 = max(0, (k - h) // 2)
        dst_x0 = max(0, (k - w) // 2)
        hh = min(k, h)
        ww = min(k, w)
        out[dst_y0:dst_y0 + hh, dst_x0:dst_x0 + ww] = grid[src_y0:src_y0 + hh, src_x0:src_x0 + ww]
        return out

    def _extract_grid(self, obs: Dict[str, Any]) -> np.ndarray:
        raw = obs.get("ego_grid")
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], list):
                rows = []
                for row in raw:
                    if not isinstance(row, list):
                        continue
                    rows.append([_safe_int(v, 0) for v in row])
                if rows:
                    width = max(1, max(len(r) for r in rows))
                    padded = [r + ([0] * (width - len(r))) for r in rows]
                    arr = np.asarray(padded, dtype=np.int32)
                    arr = np.clip(arr, 0, 2)
                    return self._normalize_grid_shape(arr)
            vals = [_safe_int(v, 0) for v in raw]
            side = int(round(np.sqrt(len(vals))))
            if side * side == len(vals) and side > 0:
                arr = np.asarray(vals, dtype=np.int32).reshape(side, side)
                arr = np.clip(arr, 0, 2)
                return self._normalize_grid_shape(arr)

        # Fallback approximation from sparse fields.
        k = int(self.ego_grid_size)
        c = k // 2
        arr = np.zeros((k, k), dtype=np.int32)
        nearby = max(0, _safe_int(obs.get("nearby_obstacles", 0), 0))
        if nearby > 0:
            arr[max(0, c - 1), c] = 1
        if nearby > 1:
            arr[c, max(0, c - 1)] = 1
        if nearby > 2:
            arr[c, min(k - 1, c + 1)] = 1
        if nearby > 3:
            arr[min(k - 1, c + 1), c] = 1
        dx = _safe_int(obs.get("goal_x", 0), 0) - _safe_int(obs.get("x", 0), 0)
        dy = _safe_int(obs.get("goal_y", 0), 0) - _safe_int(obs.get("y", 0), 0)
        if abs(dx) <= c and abs(dy) <= c:
            gx = dx + c
            gy = dy + c
            if arr[gy, gx] == 0:
                arr[gy, gx] = 2
        return arr

    def _state_key_and_phi(self, obs: JSONValue) -> Tuple[str, np.ndarray]:
        od = obs if isinstance(obs, dict) else {}
        grid = self._extract_grid(od)
        occ = (grid == 1).astype(np.float32).reshape(-1)
        goal = (grid == 2).astype(np.float32).reshape(-1)

        battery = _safe_float(od.get("battery", 0.0), 0.0)
        battery_norm = float(max(0.0, min(1.0, battery / 20.0)))
        nearby_norm = float(max(0.0, min(1.0, _safe_float(od.get("nearby_obstacles", 0.0), 0.0) / 4.0)))
        patrol = _safe_float(od.get("patrol_dist", -1.0), -1.0)
        patrol_near = 0.0 if patrol < 0.0 else (1.0 if patrol <= 2.0 else 0.0)
        on_conveyor = float(max(0.0, min(1.0, _safe_float(od.get("on_conveyor", 0.0), 0.0))))

        phi = np.concatenate(
            [
                np.array([1.0], dtype=np.float32),
                occ,
                goal,
                np.array([battery_norm, nearby_norm, patrol_near, on_conveyor], dtype=np.float32),
            ],
            axis=0,
        )

        bkt = _safe_int(round(10.0 * battery_norm), 0)
        dx = _safe_int(od.get("goal_x", 0), 0) - _safe_int(od.get("x", 0), 0)
        dy = _safe_int(od.get("goal_y", 0), 0) - _safe_int(od.get("y", 0), 0)
        dx_dir = 1 if dx > 0 else (-1 if dx < 0 else 0)
        dy_dir = 1 if dy > 0 else (-1 if dy < 0 else 0)
        key = f"{''.join(str(int(v)) for v in grid.reshape(-1).tolist())}|b{bkt}|dx{dx_dir}|dy{dy_dir}"
        return key, phi

    def _ensure_initialized(self, phi: np.ndarray) -> None:
        if self._feature_dim > 0:
            return
        self._feature_dim = int(phi.shape[0])
        self.w = np.zeros((self._feature_dim,), dtype=np.float32)
        self.forward_model = np.zeros((self.n_actions, self._feature_dim, self._feature_dim), dtype=np.float32)
        if self.use_semantic_w_init:
            emb = task_embedding_weights(target_verse_name=self.verse_name, profile=self.semantic_profile)
            self.w[0] = float(_safe_float(emb.get("bias", -0.03), -0.03))
            occ_s, occ_e = self._occ_slice
            goal_s, goal_e = self._goal_slice
            if occ_e <= self._feature_dim:
                self.w[occ_s:occ_e] = float(_safe_float(emb.get("obstacle", -1.0), -1.0))
            if goal_e <= self._feature_dim:
                self.w[goal_s:goal_e] = float(_safe_float(emb.get("goal", 2.0), 2.0))
            ex_s, ex_e = self._extra_slice
            if ex_e <= self._feature_dim:
                # [battery_norm, nearby_norm, patrol_near, on_conveyor]
                self.w[ex_s + 0] = float(_safe_float(emb.get("battery", 0.1), 0.1))
                self.w[ex_s + 1] = float(_safe_float(emb.get("obstacle", -1.0), -1.0))
                self.w[ex_s + 2] = float(_safe_float(emb.get("patrol", -0.6), -0.6))
                self.w[ex_s + 3] = float(_safe_float(emb.get("conveyor", -0.1), -0.1))
        self.stats.feature_dim = int(self._feature_dim)

    def _psi_state(self, state_key: str) -> np.ndarray:
        arr = self.psi_table.get(state_key)
        if arr is None:
            if self._feature_dim <= 0:
                raise RuntimeError("SF feature dim not initialized")
            arr = np.zeros((self.n_actions, self._feature_dim), dtype=np.float32)
            self.psi_table[state_key] = arr
        return arr

    def _q_values(self, state_key: str) -> np.ndarray:
        psi = self._psi_state(state_key)
        assert self.w is not None
        return psi @ self.w

    def act(self, obs: JSONValue) -> ActionResult:
        s_key, phi = self._state_key_and_phi(obs)
        self._ensure_initialized(phi)
        if self._rng.random() < float(self.stats.epsilon):
            a = int(self._rng.choice(self.allowed_actions))
            return ActionResult(action=a, info={"mode": "explore", "epsilon": float(self.stats.epsilon)})
        q = self._q_values(s_key)
        best = max(self.allowed_actions, key=lambda idx: float(q[idx]))
        return ActionResult(action=int(best), info={"mode": "exploit", "epsilon": float(self.stats.epsilon)})

    def action_diagnostics(self, obs: JSONValue) -> Dict[str, JSONValue]:
        s_key, phi = self._state_key_and_phi(obs)
        self._ensure_initialized(phi)
        q = self._q_values(s_key)
        centered = q - float(np.max(q))
        ex = np.exp(centered)
        probs = ex / float(max(1e-8, np.sum(ex)))
        out = {
            "q_values": [float(v) for v in q.tolist()],
            "sample_probs": [float(v) for v in probs.tolist()],
            "epsilon": float(self.stats.epsilon),
        }
        if self.w is not None:
            goal_s, goal_e = self._goal_slice
            occ_s, occ_e = self._occ_slice
            if goal_e <= self._feature_dim and occ_e <= self._feature_dim:
                out["w_goal_mean"] = float(np.mean(self.w[goal_s:goal_e]))
                out["w_obstacle_mean"] = float(np.mean(self.w[occ_s:occ_e]))
        return out

    def learn(self, batch: ExperienceBatch) -> Dict[str, JSONValue]:
        if not batch.transitions:
            return {}

        td_abs: List[float] = []
        fwd_mse: List[float] = []
        rew_err: List[float] = []
        updates = 0

        for tr in batch.transitions:
            s_key, phi_s = self._state_key_and_phi(tr.obs)
            sp_key, phi_sp = self._state_key_and_phi(tr.next_obs)
            self._ensure_initialized(phi_s)
            assert self.w is not None
            assert self.forward_model is not None

            psi_s = self._psi_state(s_key)
            psi_sp = self._psi_state(sp_key)

            a = int(tr.action)
            if a < 0 or a >= self.n_actions:
                continue
            done = bool(tr.done or tr.truncated)

            if done:
                target_vec = phi_s
            else:
                q_sp = psi_sp @ self.w
                a_next = max(self.allowed_actions, key=lambda idx: float(q_sp[idx]))
                target_vec = phi_s + self.gamma * psi_sp[a_next]

            td_vec = target_vec - psi_s[a]
            psi_s[a] = np.clip(psi_s[a] + self.psi_lr * td_vec, -200.0, 200.0)

            pred_r = float(np.dot(self.w, phi_s))
            err_r = float(tr.reward) - pred_r
            self.w = np.clip(self.w + self.w_lr * err_r * phi_s, -20.0, 20.0)

            pred_phi_sp = self.forward_model[a] @ phi_s
            err_f = phi_sp - pred_phi_sp
            self.forward_model[a] = np.clip(
                self.forward_model[a] + self.fwd_lr * np.outer(err_f, phi_s),
                -20.0,
                20.0,
            )

            td_abs.append(float(np.mean(np.abs(td_vec))))
            rew_err.append(abs(float(err_r)))
            fwd_mse.append(float(np.mean(err_f ** 2)))
            updates += 1
            self.stats.steps_seen += 1

        self.stats.updates += int(updates)
        self.stats.epsilon = float(max(self.epsilon_min, self.stats.epsilon * self.epsilon_decay))
        self.stats.state_rows = int(len(self.psi_table))
        return {
            "updates": int(updates),
            "steps_seen": int(self.stats.steps_seen),
            "epsilon": float(self.stats.epsilon),
            "sf_state_rows": int(self.stats.state_rows),
            "feature_dim": int(self._feature_dim),
            "td_abs_mean": (float(np.mean(td_abs)) if td_abs else None),
            "reward_pred_err_abs_mean": (float(np.mean(rew_err)) if rew_err else None),
            "forward_mse_mean": (float(np.mean(fwd_mse)) if fwd_mse else None),
        }

    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "n_actions": int(self.n_actions),
            "feature_dim": int(self._feature_dim),
            "w": ([] if self.w is None else [float(v) for v in self.w.tolist()]),
            "psi_table": {k: arr.tolist() for k, arr in self.psi_table.items()},
            "epsilon": float(self.stats.epsilon),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fd = int(_safe_int(payload.get("feature_dim", 0), 0))
        if fd <= 0:
            raise ValueError("Invalid SF checkpoint feature_dim")
        self._feature_dim = int(fd)
        self.w = np.asarray(payload.get("w", []), dtype=np.float32)
        if self.w.shape[0] != self._feature_dim:
            raise ValueError("Invalid SF checkpoint w length")
        self.forward_model = np.zeros((self.n_actions, self._feature_dim, self._feature_dim), dtype=np.float32)
        table_raw = payload.get("psi_table", {})
        self.psi_table = {}
        if isinstance(table_raw, dict):
            for k, arr in table_raw.items():
                v = np.asarray(arr, dtype=np.float32)
                if v.shape != (self.n_actions, self._feature_dim):
                    continue
                self.psi_table[str(k)] = v
        self.stats.epsilon = float(_safe_float(payload.get("epsilon", self.epsilon_start), self.epsilon_start))
        self.stats.feature_dim = int(self._feature_dim)
        self.stats.state_rows = int(len(self.psi_table))

    def close(self) -> None:
        return
