from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.types import VerseSpec
from verses.registry import create_verse

from tools.validate_sf_transfer_support import EgoGridAdapter, EgoObservation, _safe_float, _safe_int


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
                if arr.ndim != 2 or int(arr.shape[1]) != int(self.feature_dim) or int(arr.shape[0]) != int(self.n_actions):
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
        return self._psi_state(key) @ self.w

    def q_values(self, key: str) -> np.ndarray:
        q = self._q_values_self(key)
        if not self._gpi_psi_banks:
            return q
        best = q.copy()
        for bank in self._gpi_psi_banks:
            arr = bank.get(key)
            if arr is None:
                continue
            best = np.maximum(best, (arr @ self.w).astype(np.float32))
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
        if done:
            target_vec = phi_sp
        else:
            q_sp = self.q_values(sp_key)
            a_next = max(self.allowed_actions, key=lambda idx: float(q_sp[idx]))
            target_vec = phi_sp + self.gamma * psi_sp[a_next]
        td_vec = target_vec - psi_s[a]
        if learn_psi:
            psi_s[a] = np.clip(psi_s[a] + self.psi_lr * td_vec, -200.0, 200.0)
        err_r = float(reward) - float(np.dot(self.w, phi_sp))
        if learn_w:
            self.w = np.clip(self.w + self.w_lr * err_r * phi_sp, -20.0, 20.0)
        fwd_err_vec = phi_sp - (self.forward_model[a] @ phi_s)
        self.forward_model[a] = np.clip(self.forward_model[a] + self.fwd_lr * np.outer(fwd_err_vec, phi_s), -20.0, 20.0)
        return {
            "td_abs_mean": float(np.mean(np.abs(td_vec))),
            "reward_pred_err_abs": abs(err_r),
            "forward_mse": float(np.mean((fwd_err_vec) ** 2)),
        }

    def predict_forward_mse(self, *, a: int, phi_s: np.ndarray, phi_sp: np.ndarray) -> float:
        return float(np.mean((phi_sp - (self.forward_model[a] @ phi_s)) ** 2))


@dataclass
class EpisodeStats:
    return_sum: float
    success: bool
    steps: int
    hazards: int
    mean_fwd_mse: float


def _semantic_reward_weights(*, feature_dim: int, grid_size: int) -> np.ndarray:
    w = np.zeros((feature_dim,), dtype=np.float32)
    occ_start = 1
    occ_end = 1 + grid_size * grid_size
    goal_start = occ_end
    goal_end = goal_start + grid_size * grid_size
    w[0] = -0.03
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
        if arr.ndim != 2 or int(arr.shape[1]) != int(feature_dim):
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
        return np.zeros((feature_dim,), dtype=np.float32) if fallback is None else np.asarray(fallback, dtype=np.float32).copy()
    x = np.asarray([np.asarray(f, dtype=np.float32) for f in features], dtype=np.float32)
    y = np.asarray([float(r) for r in rewards], dtype=np.float32)
    d = int(feature_dim)
    try:
        xtx = (x.T @ x).astype(np.float32)
        reg = np.eye(d, dtype=np.float32) * float(max(1e-6, l2))
        xty = (x.T @ y).astype(np.float32)
        return np.clip(np.linalg.solve(xtx + reg, xty).astype(np.float32), -20.0, 20.0)
    except Exception:
        return np.zeros((feature_dim,), dtype=np.float32) if fallback is None else np.asarray(fallback, dtype=np.float32).copy()


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
            phi_s = adapter.phi(ego)
            a = agent.select_action(adapter.state_key(ego), epsilon=float(epsilon), rng=rng)
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


def _build_verse(*, verse_name: str, seed: int, params: Dict[str, Any]) -> Any:
    spec = VerseSpec(spec_version="v1", verse_name=verse_name, verse_version="0.1", seed=int(seed), params=dict(params))
    verse = create_verse(spec)
    verse.seed(int(seed))
    return verse


def _hazard_count(info: Dict[str, Any]) -> int:
    keys = ["hit_wall", "bumped_wall", "hit_obstacle", "hit_patrol", "battery_death", "battery_depleted", "hit_hazard", "fell_pit", "hit_laser"]
    return int(sum(1 for k in keys if bool(info.get(k, False))))


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
        success = bool(success or info.get("reached_goal", False))
        ego_sp = adapter.extract(verse_name=verse_name, verse=verse, obs=next_obs)
        sp_key = adapter.state_key(ego_sp)
        phi_sp = adapter.phi(ego_sp)
        done = bool(sr.done or sr.truncated)
        if train:
            upd = agent.update(s_key=s_key, a=int(a), phi_s=phi_s, reward=float(sr.reward), sp_key=sp_key, phi_sp=phi_sp, done=done, learn_psi=bool(learn_psi), learn_w=bool(learn_w))
            fwd_mses.append(float(upd["forward_mse"]))
        else:
            fwd_mses.append(float(agent.predict_forward_mse(a=int(a), phi_s=phi_s, phi_sp=phi_sp)))
        obs = next_obs
        if done:
            return EpisodeStats(return_sum=float(total), success=bool(success), steps=int(step + 1), hazards=int(hazards), mean_fwd_mse=float(sum(fwd_mses) / float(max(1, len(fwd_mses)))))
    return EpisodeStats(return_sum=float(total), success=bool(success), steps=int(max_steps), hazards=int(hazards), mean_fwd_mse=float(sum(fwd_mses) / float(max(1, len(fwd_mses)))))


def _summarize(stats: List[EpisodeStats]) -> Dict[str, Any]:
    if not stats:
        return {"episodes": 0, "mean_return": 0.0, "success_rate": 0.0, "hazard_per_1k": 0.0, "mean_steps": 0.0, "mean_forward_mse": 0.0}
    returns = [s.return_sum for s in stats]
    wins = [1.0 if s.success else 0.0 for s in stats]
    steps = [float(s.steps) for s in stats]
    hazards = [float(s.hazards) for s in stats]
    fwd = [float(s.mean_fwd_mse) for s in stats]
    total_steps = max(1.0, float(sum(steps)))
    return {
        "episodes": int(len(stats)),
        "mean_return": float(sum(returns) / float(len(returns))),
        "median_return": float(statistics.median(returns)),
        "success_rate": float(sum(wins) / float(len(wins))),
        "hazard_per_1k": float(1000.0 * float(sum(hazards)) / total_steps),
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
        return {"episodes": 0, "return": [], "success": [], "hazard_per_1k": [], "forward_mse": [], "steps": []}
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
    return {"episodes": int(k), "return": out_ret, "success": out_succ, "hazard_per_1k": out_haz, "forward_mse": out_fwd, "steps": out_steps}


def _trace_diagnostics(trace: Dict[str, Any]) -> Dict[str, Any]:
    def _arr(name: str) -> List[float]:
        raw = trace.get(name, [])
        return [] if not isinstance(raw, list) else [float(x) for x in raw if isinstance(x, (int, float))]

    ret = _arr("return")
    suc = _arr("success")
    haz = _arr("hazard_per_1k")
    fwd = _arr("forward_mse")
    n = int(min(len(ret), len(suc), len(haz), len(fwd))) if ret and suc and haz and fwd else int(max(len(ret), len(suc), len(haz), len(fwd)))

    def _first_last_delta(vals: List[float]) -> float:
        return 0.0 if len(vals) <= 1 else float(vals[-1] - vals[0])

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
    n = min(int(a.get("episodes", 0) or 0), int(b.get("episodes", 0) or 0), int(max_points))
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
            sr = verse.step(int(rng.randrange(n_actions)))
            obs = sr.obs if isinstance(sr.obs, dict) else {}
            if bool(sr.done or sr.truncated):
                break
    return probes


def _policy_bank_agreement_diag(*, snapshots: List[TabularSFAgent], probes: List[EgoObservation], adapter: EgoGridAdapter) -> Dict[str, Any]:
    if not snapshots or not probes:
        return {"num_snapshots": int(len(snapshots)), "num_probes": int(len(probes)), "evaluated_probes": 0, "mean_majority_fraction": 0.0, "mean_unique_actions": 0.0, "mean_vote_margin": 0.0}
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
        return {"num_snapshots": int(len(snapshots)), "num_probes": int(len(probes)), "evaluated_probes": 0, "mean_majority_fraction": 0.0, "mean_unique_actions": 0.0, "mean_vote_margin": 0.0}
    return {"num_snapshots": int(len(snapshots)), "num_probes": int(len(probes)), "evaluated_probes": int(len(probe_majority)), "mean_majority_fraction": float(sum(probe_majority) / float(len(probe_majority))), "mean_unique_actions": float(sum(probe_unique) / float(len(probe_unique))), "mean_vote_margin": float(sum(probe_margin) / float(len(probe_margin)))}


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
