from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


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
    occupancy: np.ndarray
    goal: np.ndarray


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

        self._mark_goal_direction(goal=ego.goal, dgx=int(gx - x), dgy=int(gy - y))
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
        self._mark_goal_direction(goal=ego.goal, dgx=int(gx - x), dgy=int(gy - y))
        return ego

    def from_maze_world(self, obs: Dict[str, Any]) -> EgoObservation:
        ego = self._blank()
        c = self.radius
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
        self._mark_goal_direction(goal=ego.goal, dgx=int(gx - x), dgy=int(gy - y))
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
        occ = ego.occupancy.astype(np.float32).reshape(-1)
        goal = ego.goal.astype(np.float32).reshape(-1)
        return np.concatenate([np.array([1.0], dtype=np.float32), occ, goal], axis=0)

    def state_key(self, ego: EgoObservation) -> str:
        occ = "".join("1" if int(v) else "0" for v in ego.occupancy.reshape(-1).tolist())
        goal = "".join("1" if int(v) else "0" for v in ego.goal.reshape(-1).tolist())
        return f"{occ}|{goal}"


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
