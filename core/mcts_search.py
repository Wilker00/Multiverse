"""
core/mcts_search.py

Neural-guided Monte Carlo Tree Search (MCTS) with PUCT.
Designed to be generic for any Verse that supports:
- discrete action space
- export_state/import_state checkpoints
- step(action)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.types import JSONValue


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_float_list(x: Any) -> List[float]:
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out
    if isinstance(x, tuple):
        return _to_float_list(list(x))
    return []


def _normalize_probs(values: List[float], legal_actions: List[int]) -> Dict[int, float]:
    if not legal_actions:
        return {}
    raw = [max(0.0, _safe_float(values[a] if 0 <= a < len(values) else 0.0, 0.0)) for a in legal_actions]
    s = sum(raw)
    if s <= 0.0:
        u = 1.0 / float(len(legal_actions))
        return {int(a): u for a in legal_actions}
    return {int(a): float(raw[i] / s) for i, a in enumerate(legal_actions)}


def _extract_value_scalar(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return 0.0
    if isinstance(x, list) and x:
        return _extract_value_scalar(x[0])
    return 0.0


@dataclass
class MCTSConfig:
    num_simulations: int = 96
    max_depth: int = 12
    c_puct: float = 1.4
    discount: float = 0.99
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25
    min_prior: float = 1e-6
    reward_scale: float = 10.0
    terminal_win_value: float = 1.0
    terminal_loss_value: float = -1.0
    forced_loss_threshold: float = -0.95
    forced_loss_min_visits: int = 4
    value_confidence_threshold: float = 0.0
    transposition_cache: bool = True
    transposition_max_entries: int = 20000
    seed: Optional[int] = None


@dataclass
class MCTSSearchResult:
    action_probs: List[float]
    visit_counts: List[int]
    action_values: List[float]
    best_action: int
    forced_loss_actions: List[int] = field(default_factory=list)
    forced_loss_detected: bool = False
    principal_variation: List[int] = field(default_factory=list)
    root_value: float = 0.0
    simulations: int = 0
    avg_leaf_value: float = 0.0
    runtime_errors: Dict[str, int] = field(default_factory=dict)


@dataclass
class _MCTSNode:
    state: Dict[str, JSONValue]
    obs: JSONValue
    done: bool = False
    truncated: bool = False
    info: Dict[str, JSONValue] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    legal_actions: List[int] = field(default_factory=list)
    priors: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "_MCTSNode"] = field(default_factory=dict)

    def q(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return float(self.value_sum) / float(self.visit_count)


class MCTSSearch:
    def __init__(self, *, verse: Any, config: Optional[MCTSConfig] = None):
        self.verse = verse
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)
        self._action_count = max(1, _safe_int(getattr(getattr(verse, "action_space", None), "n", 0), 0))
        self._prior_cache: Dict[str, Dict[int, float]] = {}
        self._value_cache: Dict[str, float] = {}
        self._last_leaf_values: List[float] = []
        self._runtime_error_counts: Dict[str, int] = {}

    @property
    def action_count(self) -> int:
        return int(self._action_count)

    def seed(self, seed: Optional[int]) -> None:
        self.config.seed = seed
        self._rng.seed(seed)

    def search(
        self,
        *,
        root_obs: JSONValue,
        policy_net: Optional[Any] = None,
        value_net: Optional[Any] = None,
        num_simulations: Optional[int] = None,
    ) -> MCTSSearchResult:
        if not hasattr(self.verse, "export_state") or not hasattr(self.verse, "import_state"):
            raise RuntimeError("MCTS requires verse.export_state() and verse.import_state().")

        root_state = self._export_state()
        original_state = dict(root_state)
        root = _MCTSNode(state=dict(root_state), obs=root_obs, done=False, truncated=False, info={})
        self._runtime_error_counts = {}

        try:
            self._expand_node(root, policy_net=policy_net, history=None, add_root_noise=True)
            sims = max(1, _safe_int(num_simulations if num_simulations is not None else self.config.num_simulations, 1))
            self._last_leaf_values = []
            for _ in range(sims):
                lv = self._simulate(root=root, policy_net=policy_net, value_net=value_net)
                self._last_leaf_values.append(float(lv))
        finally:
            self._import_state(original_state)

        visits = [0 for _ in range(self.action_count)]
        values = [0.0 for _ in range(self.action_count)]
        for a in root.legal_actions:
            child = root.children.get(int(a))
            if child is None:
                continue
            visits[int(a)] = int(child.visit_count)
            values[int(a)] = float(child.q())

        total_visits = sum(visits)
        if total_visits <= 0:
            pri = [0.0 for _ in range(self.action_count)]
            for a, p in root.priors.items():
                if 0 <= int(a) < self.action_count:
                    pri[int(a)] = float(max(0.0, p))
            probs = self._safe_prob_vector(pri, root.legal_actions)
        else:
            probs = [float(v) / float(total_visits) for v in visits]

        best_action = int(max(range(self.action_count), key=lambda i: probs[i]))
        forced_loss_actions = [
            int(a)
            for a in root.legal_actions
            if visits[int(a)] >= int(self.config.forced_loss_min_visits)
            and values[int(a)] <= float(self.config.forced_loss_threshold)
        ]
        pv = self._principal_variation(root, max_depth=max(1, int(self.config.max_depth)))
        return MCTSSearchResult(
            action_probs=probs,
            visit_counts=visits,
            action_values=values,
            best_action=best_action,
            forced_loss_actions=forced_loss_actions,
            forced_loss_detected=bool(len(forced_loss_actions) > 0),
            principal_variation=pv,
            root_value=float(root.q()),
            simulations=int(sims),
            avg_leaf_value=(
                float(sum(self._last_leaf_values) / float(len(self._last_leaf_values)))
                if self._last_leaf_values
                else 0.0
            ),
            runtime_errors=dict(self._runtime_error_counts),
        )

    def _record_runtime_error(self, code: str) -> None:
        key = str(code or "").strip() or "unknown_error"
        self._runtime_error_counts[key] = int(self._runtime_error_counts.get(key, 0)) + 1

    def _simulate(self, *, root: _MCTSNode, policy_net: Optional[Any], value_net: Optional[Any]) -> float:
        self._import_state(root.state)
        node = root
        nodes: List[_MCTSNode] = [node]
        rewards: List[float] = []
        history_rows: List[Dict[str, JSONValue]] = []
        leaf_value = 0.0

        for depth in range(max(1, int(self.config.max_depth))):
            if node.done or node.truncated:
                leaf_value = self._terminal_value(
                    reward=(rewards[-1] if rewards else 0.0),
                    info=node.info,
                )
                break

            if not node.expanded:
                self._expand_node(node, policy_net=policy_net, history=history_rows, add_root_noise=False)
                leaf_value = self._evaluate_value(value_net=value_net, obs=node.obs, history=history_rows, state=node.state)
                break

            action = self._select_action(node)
            step = self.verse.step(int(action))
            reward_norm = self._normalize_reward(step.reward)
            rewards.append(float(reward_norm))
            history_rows.append({"obs": node.obs, "action": int(action), "reward": float(step.reward)})

            next_state = self._export_state()
            child = node.children.get(int(action))
            is_new = child is None
            if child is None:
                child = _MCTSNode(
                    state=next_state,
                    obs=step.obs,
                    done=bool(step.done),
                    truncated=bool(step.truncated),
                    info=dict(step.info or {}),
                )
                node.children[int(action)] = child
            else:
                # Keep tree useful in stochastic verses by refreshing latest sampled successor.
                child.state = next_state
                child.obs = step.obs
                child.done = bool(step.done)
                child.truncated = bool(step.truncated)
                child.info = dict(step.info or {})

            node = child
            nodes.append(node)

            if node.done or node.truncated:
                leaf_value = self._terminal_value(reward=reward_norm, info=node.info)
                break
            if is_new:
                self._expand_node(node, policy_net=policy_net, history=history_rows, add_root_noise=False)
                leaf_value = self._evaluate_value(value_net=value_net, obs=node.obs, history=history_rows, state=node.state)
                break
        else:
            leaf_value = self._evaluate_value(value_net=value_net, obs=node.obs, history=history_rows, state=node.state)

        g = float(max(-1.0, min(1.0, leaf_value)))
        for i in range(len(nodes) - 1, -1, -1):
            cur = nodes[i]
            cur.visit_count += 1
            cur.value_sum += float(g)
            if i > 0 and i - 1 < len(rewards):
                g = float(rewards[i - 1]) + float(self.config.discount) * float(g)
                g = float(max(-1.0, min(1.0, g)))
        return float(leaf_value)

    def _expand_node(
        self,
        node: _MCTSNode,
        *,
        policy_net: Optional[Any],
        history: Optional[List[Dict[str, JSONValue]]],
        add_root_noise: bool,
    ) -> None:
        legal_actions = self._legal_actions(node.obs)
        state_key = self._state_key(node.state)
        pri = None
        if bool(self.config.transposition_cache):
            pri = self._prior_cache.get(state_key)
        if not isinstance(pri, dict):
            pri = self._policy_priors(
                policy_net=policy_net,
                obs=node.obs,
                legal_actions=legal_actions,
                history=history,
            )
            if bool(self.config.transposition_cache):
                self._cache_put(self._prior_cache, state_key, dict(pri))
        if add_root_noise and len(legal_actions) > 1 and self.config.dirichlet_epsilon > 0.0:
            noise = self._dirichlet_noise(len(legal_actions), alpha=float(self.config.dirichlet_alpha))
            eps = max(0.0, min(1.0, float(self.config.dirichlet_epsilon)))
            mixed: Dict[int, float] = {}
            for i, a in enumerate(legal_actions):
                base = float(pri.get(int(a), 0.0))
                mixed[int(a)] = ((1.0 - eps) * base) + (eps * float(noise[i]))
            s = sum(mixed.values())
            if s > 0.0:
                pri = {a: float(v / s) for a, v in mixed.items()}

        node.legal_actions = [int(a) for a in legal_actions]
        node.priors = {int(a): max(float(self.config.min_prior), float(pri.get(int(a), 0.0))) for a in legal_actions}
        s = sum(node.priors.values())
        if s > 0.0:
            node.priors = {a: float(v / s) for a, v in node.priors.items()}
        node.expanded = True

    def _select_action(self, node: _MCTSNode) -> int:
        legal = node.legal_actions if node.legal_actions else list(range(self.action_count))
        if not legal:
            return 0
        sqrt_parent = math.sqrt(float(max(1, node.visit_count)))
        best_score = -10**9
        best_actions: List[int] = []
        for a in legal:
            child = node.children.get(int(a))
            q = 0.0 if child is None else child.q()
            n = 0 if child is None else int(child.visit_count)
            prior = float(node.priors.get(int(a), max(float(self.config.min_prior), 1e-6)))
            u = float(self.config.c_puct) * prior * sqrt_parent / float(1 + n)
            score = q + u
            if score > best_score + 1e-12:
                best_score = score
                best_actions = [int(a)]
            elif abs(score - best_score) <= 1e-12:
                best_actions.append(int(a))
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(best_actions[self._rng.randrange(len(best_actions))])

    def _policy_priors(
        self,
        *,
        policy_net: Optional[Any],
        obs: JSONValue,
        legal_actions: List[int],
        history: Optional[List[Dict[str, JSONValue]]],
    ) -> Dict[int, float]:
        if not legal_actions:
            return {}
        raw: Any = None
        if policy_net is None:
            raw = None
        elif hasattr(policy_net, "policy_distribution") and callable(getattr(policy_net, "policy_distribution")):
            try:
                raw = policy_net.policy_distribution(obs)
            except Exception:
                self._record_runtime_error("policy_distribution_error")
                raw = None
        elif hasattr(policy_net, "action_diagnostics") and callable(getattr(policy_net, "action_diagnostics")):
            try:
                diag = policy_net.action_diagnostics(obs)
                if isinstance(diag, dict):
                    raw = diag.get("sample_probs")
            except Exception:
                self._record_runtime_error("policy_action_diagnostics_error")
                raw = None
        elif callable(policy_net):
            try:
                raw = self._call_with_optional_history(policy_net, obs, history)
            except Exception:
                self._record_runtime_error("policy_callable_error")
                raw = None

        if isinstance(raw, dict):
            probs = [0.0 for _ in range(self.action_count)]
            for k, v in raw.items():
                idx = _safe_int(k, -1)
                if 0 <= idx < self.action_count:
                    probs[idx] = max(0.0, _safe_float(v, 0.0))
            return _normalize_probs(probs, legal_actions)

        probs = _to_float_list(raw)
        if len(probs) < self.action_count:
            probs = probs + [0.0] * (self.action_count - len(probs))
        return _normalize_probs(probs, legal_actions)

    def _evaluate_value(
        self,
        *,
        value_net: Optional[Any],
        obs: JSONValue,
        history: Optional[List[Dict[str, JSONValue]]],
        state: Optional[Dict[str, JSONValue]] = None,
    ) -> float:
        if value_net is None:
            return 0.0

        state_key = self._state_key(state) if isinstance(state, dict) else ""
        if bool(self.config.transposition_cache) and state_key:
            cached = self._value_cache.get(state_key)
            if cached is not None:
                return float(max(-1.0, min(1.0, cached)))

        raw: Any = None
        if hasattr(value_net, "value_with_confidence") and callable(getattr(value_net, "value_with_confidence")):
            try:
                raw = self._call_with_optional_history(value_net.value_with_confidence, obs, history)
            except Exception:
                self._record_runtime_error("value_with_confidence_error")
                raw = None
        if raw is None and hasattr(value_net, "value_estimate") and callable(getattr(value_net, "value_estimate")):
            try:
                raw = self._call_with_optional_history(value_net.value_estimate, obs, history)
            except Exception:
                self._record_runtime_error("value_estimate_error")
                raw = None
        elif raw is None and hasattr(value_net, "predict_value") and callable(getattr(value_net, "predict_value")):
            try:
                raw = self._call_with_optional_history(value_net.predict_value, obs, history)
            except Exception:
                self._record_runtime_error("predict_value_error")
                raw = None
        elif raw is None and hasattr(value_net, "predict_policy_value") and callable(getattr(value_net, "predict_policy_value")):
            try:
                out = self._call_with_optional_history(value_net.predict_policy_value, obs, history)
                if isinstance(out, dict):
                    raw = out.get("value")
                else:
                    raw = out
            except Exception:
                self._record_runtime_error("predict_policy_value_error")
                raw = None
        elif raw is None and callable(value_net):
            try:
                raw = self._call_with_optional_history(value_net, obs, history)
            except Exception:
                self._record_runtime_error("value_callable_error")
                raw = None

        conf = 1.0
        if isinstance(raw, dict):
            conf = _safe_float(raw.get("confidence", 1.0), 1.0)
            raw = raw.get("value", 0.0)
        if conf < float(max(0.0, min(1.0, self.config.value_confidence_threshold))):
            v = 0.0
        else:
            v = _extract_value_scalar(raw)
        if bool(self.config.transposition_cache) and state_key:
            self._cache_put(self._value_cache, state_key, float(v))
        return float(max(-1.0, min(1.0, v)))

    def _call_with_optional_history(self, fn: Any, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]]) -> Any:
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 2:
                return fn(obs, history)
        except Exception:
            self._record_runtime_error("call_history_signature_error")
        try:
            return fn(obs)
        except TypeError:
            try:
                return fn(obs, history)
            except Exception:
                self._record_runtime_error("call_history_fallback_error")
                raise

    def _terminal_value(self, *, reward: float, info: Optional[Dict[str, JSONValue]]) -> float:
        meta = info if isinstance(info, dict) else {}
        if bool(meta.get("reached_goal", False)):
            return float(self.config.terminal_win_value)
        if bool(meta.get("lost_game", False)):
            return float(self.config.terminal_loss_value)
        if (
            bool(meta.get("fell_cliff", False))
            or bool(meta.get("fell_pit", False))
            or bool(meta.get("hit_laser", False))
            or bool(meta.get("battery_depleted", False))
            or bool(meta.get("battery_death", False))
            or bool(meta.get("hit_wall", False))
            or bool(meta.get("hit_obstacle", False))
        ):
            return float(self.config.terminal_loss_value)
        return float(max(-1.0, min(1.0, reward)))

    def _normalize_reward(self, reward: float) -> float:
        scale = max(1e-6, float(self.config.reward_scale))
        return float(math.tanh(float(reward) / scale))

    def _legal_actions(self, obs: JSONValue) -> List[int]:
        methods = ("legal_actions", "valid_actions", "available_actions", "get_valid_actions")
        for name in methods:
            fn = getattr(self.verse, name, None)
            if not callable(fn):
                continue
            raw = None
            try:
                raw = fn(obs)
            except TypeError:
                try:
                    raw = fn()
                except Exception:
                    self._record_runtime_error(f"legal_actions_{str(name)}_error")
                    raw = None
            except Exception:
                self._record_runtime_error(f"legal_actions_{str(name)}_error")
                raw = None
            if not isinstance(raw, list):
                continue
            out: List[int] = []
            seen = set()
            for x in raw:
                a = _safe_int(x, -1)
                if 0 <= a < self.action_count and a not in seen:
                    seen.add(a)
                    out.append(a)
            if out:
                return out
        return list(range(self.action_count))

    def _dirichlet_noise(self, n: int, *, alpha: float) -> List[float]:
        n = max(1, int(n))
        a = max(1e-6, float(alpha))
        xs = [self._rng.gammavariate(a, 1.0) for _ in range(n)]
        s = sum(xs)
        if s <= 0.0:
            return [1.0 / float(n) for _ in range(n)]
        return [float(v / s) for v in xs]

    def _principal_variation(self, root: _MCTSNode, *, max_depth: int) -> List[int]:
        pv: List[int] = []
        node = root
        for _ in range(max(1, int(max_depth))):
            if not node.children:
                break
            best = max(node.children.items(), key=lambda kv: int(kv[1].visit_count))
            action = int(best[0])
            child = best[1]
            if int(child.visit_count) <= 0:
                break
            pv.append(action)
            node = child
        return pv

    def _safe_prob_vector(self, probs: List[float], legal_actions: List[int]) -> List[float]:
        out = [0.0 for _ in range(self.action_count)]
        if not legal_actions:
            return out
        total = 0.0
        for a in legal_actions:
            if 0 <= int(a) < len(probs):
                out[int(a)] = max(0.0, float(probs[int(a)]))
                total += out[int(a)]
        if total <= 0.0:
            u = 1.0 / float(len(legal_actions))
            for a in legal_actions:
                out[int(a)] = u
            return out
        return [float(v / total) for v in out]

    def _state_key(self, state: Optional[Dict[str, JSONValue]]) -> str:
        if not isinstance(state, dict):
            return ""
        try:
            return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            self._record_runtime_error("state_key_serialize_error")
            return str(state)

    def _cache_put(self, cache: Dict[str, Any], key: str, value: Any) -> None:
        if not key:
            return
        if key in cache:
            cache[key] = value
            return
        if len(cache) >= max(128, int(self.config.transposition_max_entries)):
            try:
                cache.pop(next(iter(cache)))
            except Exception:
                self._record_runtime_error("cache_eviction_error")
                cache.clear()
        cache[key] = value

    def _export_state(self) -> Dict[str, JSONValue]:
        st = self.verse.export_state()
        if not isinstance(st, dict):
            raise RuntimeError("verse.export_state() must return dict[str, JSONValue].")
        return dict(st)

    def _import_state(self, state: Dict[str, JSONValue]) -> None:
        self.verse.import_state(dict(state))


class AgentPolicyPrior:
    """
    Adapter: extract policy priors P(a|s) from an agent-like object.
    """

    def __init__(self, *, agent: Any, action_count: int):
        self.agent = agent
        self.action_count = max(1, int(action_count))

    def __call__(self, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]] = None) -> List[float]:
        if hasattr(self.agent, "policy_distribution") and callable(getattr(self.agent, "policy_distribution")):
            try:
                probs = self.agent.policy_distribution(obs)
                vec = _to_float_list(probs)
                if len(vec) >= self.action_count:
                    return vec[: self.action_count]
            except Exception:
                pass
        if hasattr(self.agent, "action_diagnostics") and callable(getattr(self.agent, "action_diagnostics")):
            try:
                diag = self.agent.action_diagnostics(obs)
                if isinstance(diag, dict):
                    vec = _to_float_list(diag.get("sample_probs"))
                    if len(vec) >= self.action_count:
                        return vec[: self.action_count]
            except Exception:
                pass
        try:
            action = int(self.agent.act(obs).action)
        except Exception:
            action = 0
        out = [0.0 for _ in range(self.action_count)]
        if 0 <= action < self.action_count:
            out[action] = 1.0
        else:
            u = 1.0 / float(self.action_count)
            out = [u for _ in range(self.action_count)]
        return out


class MetaTransformerValue:
    """
    Adapter: use MetaTransformer checkpoint to estimate V(s).
    """

    def __init__(self, *, checkpoint_path: str, history_len: int = 6):
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"MetaTransformer checkpoint not found: {checkpoint_path}")
        try:
            import torch
        except Exception as e:
            raise RuntimeError("torch is required for MetaTransformer value adapter") from e
        from memory.embeddings import obs_to_vector
        from models.meta_transformer import MetaTransformer

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(cfg, dict):
            cfg = {}
        state_dim = max(1, int(cfg.get("state_dim", 1) or 1))
        action_dim = max(1, int(cfg.get("action_dim", 1) or 1))
        n_embd = max(16, int(cfg.get("n_embd", 256) or 256))
        model = MetaTransformer(state_dim=state_dim, action_dim=action_dim, n_embd=n_embd)
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        model.eval()

        self._torch = torch
        self._model = model
        self._obs_to_vector = obs_to_vector
        self._state_dim = state_dim
        self._context_dim = int(getattr(model, "context_input_dim", state_dim + 2))
        self._history_len = max(1, int(ckpt.get("history_len", history_len) or history_len))

    def _pad(self, vec: List[float], size: int) -> List[float]:
        if len(vec) >= size:
            return vec[:size]
        return vec + [0.0] * (size - len(vec))

    def value_estimate(self, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]] = None) -> float:
        out = self.value_with_confidence(obs=obs, history=history)
        return float(max(-1.0, min(1.0, _extract_value_scalar(out.get("value", 0.0)))))

    def value_with_confidence(self, obs: JSONValue, history: Optional[List[Dict[str, JSONValue]]] = None) -> Dict[str, float]:
        try:
            vec = self._pad(self._obs_to_vector(obs), self._state_dim)
        except Exception:
            return {"value": 0.0, "confidence": 0.0}

        rows: List[List[float]] = []
        if isinstance(history, list):
            tail = history[-self._history_len :]
            for row in tail:
                if not isinstance(row, dict):
                    continue
                h_obs = row.get("obs")
                h_action = _safe_float(row.get("action", 0.0), 0.0)
                h_reward = _safe_float(row.get("reward", 0.0), 0.0)
                try:
                    h_vec = self._pad(self._obs_to_vector(h_obs), self._state_dim)
                except Exception:
                    continue
                rows.append(h_vec + [float(h_action), float(h_reward)])

        if len(rows) < self._history_len:
            rows = ([[0.0] * self._context_dim] * (self._history_len - len(rows))) + rows

        t_state = self._torch.tensor([vec], dtype=self._torch.float32)
        t_hist = self._torch.tensor([rows], dtype=self._torch.float32)
        with self._torch.no_grad():
            logits = self._model(t_state, t_hist)
            probs = self._torch.softmax(logits, dim=-1).squeeze(0)
            entropy = float(-(probs * self._torch.log(probs + 1e-8)).sum().item())
            max_entropy = math.log(float(max(2, int(probs.shape[0]))))
            confidence = float(max(0.0, min(1.0, 1.0 - (entropy / max(1e-8, max_entropy)))))
            if hasattr(self._model, "forward_value"):
                value = self._model.forward_value(t_state, t_hist)
            else:
                pred = self._model.predict(state=t_state, recent_history=t_hist)
                value = pred.get("value", 0.0) if isinstance(pred, dict) else 0.0
        v = _extract_value_scalar(value)
        return {
            "value": float(max(-1.0, min(1.0, v))),
            "confidence": float(confidence),
        }
