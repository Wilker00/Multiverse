"""
agents/special_moe_agent.py

Special MoE agent:
- Uses multiple imitation experts (one dataset per expert skill)
- Optionally routes with a MicroSelector checkpoint (softmax + top-k)
- Preserves boundary-avoid behavior via bad DNA state filtering
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.distilled_agent import DistilledAgent
from agents.imitation_agent import ImitationLookupAgent, obs_key
from core.agent_base import ActionResult
from core.types import AgentSpec, JSONValue, SpaceSpec
from memory.boundary import load_bad_obs


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


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    if isinstance(x, (int, float)):
        return bool(x)
    return default


def _flatten_obs(obs: JSONValue, keys: List[str]) -> List[float]:
    if not isinstance(obs, dict):
        return []
    flat: List[float] = []
    for key in keys:
        if key not in obs:
            continue
        val = obs.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            flat.append(float(val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    flat.append(float(item))
    return flat


def _pad_or_truncate(values: List[float], target_len: int) -> List[float]:
    if target_len <= 0:
        return []
    if len(values) == target_len:
        return values
    if len(values) > target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))


def _is_navigation_skill(skill_id: str) -> bool:
    s = str(skill_id or "").strip().lower()
    if not s:
        return False
    nav_tokens = ("grid", "park", "warehouse", "cliff", "labyrinth", "line", "navigation")
    return any(tok in s for tok in nav_tokens)


def _is_strategy_observation(obs: JSONValue) -> bool:
    if not isinstance(obs, dict):
        return False
    strategy_keys = {
        "material_delta",
        "king_safety",
        "development",
        "center_control",
        "territory_delta",
        "liberties_delta",
        "capture_threat",
        "ko_risk",
        "my_cards",
        "opp_cards",
        "color_control",
        "uno_ready",
    }
    return any(k in obs for k in strategy_keys)


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _validate_allowed_keys(*, payload: Dict[str, Any], allowed: List[str], label: str) -> None:
    allow = {str(k) for k in allowed}
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in allow)
    if not unknown:
        return
    raise ValueError(f"{label} has unknown keys: {unknown}. Allowed keys: {sorted(allow)}")


def _as_str_list(value: Any, *, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out if out else list(default)
    raise ValueError(f"Expected list[str], got {type(value)}")


@dataclass(frozen=True)
class SpecialMoEConfig:
    top_k: int = 2
    selector_temperature: float = 1.0
    selector_unknown_mix: float = 0.35
    state_keys: List[str] = field(default_factory=lambda: ["pos", "t", "x", "y", "agent"])
    goal_keys: List[str] = field(default_factory=lambda: ["goal", "goal_x", "goal_y", "target"])
    skill_bias: Dict[str, Any] = field(default_factory=dict)
    verse_skill_bias: Dict[str, Any] = field(default_factory=dict)
    verse_name: str = ""
    expert_lookup_config: Dict[str, Any] = field(default_factory=dict)
    expert_performance: Dict[str, Any] = field(default_factory=dict)
    expert_performance_path: str = ""
    bad_dna_path: str = ""
    selector_model_path: str = ""
    expert_dataset_map: Dict[str, Any] = field(default_factory=dict)
    expert_model_map: Dict[str, Any] = field(default_factory=dict)
    dataset_dir: str = ""
    expert_dataset_dir: str = ""
    expert_model_dir: str = ""
    dataset_path: str = ""
    dataset_paths: List[str] = field(default_factory=list)
    strict_config_validation: bool = True

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "SpecialMoEConfig":
        c = cfg if isinstance(cfg, dict) else {}
        strict = bool(c.get("strict_config_validation", True))
        if strict:
            _validate_allowed_keys(
                payload=c,
                allowed=[
                    "top_k",
                    "selector_temperature",
                    "selector_unknown_mix",
                    "state_keys",
                    "goal_keys",
                    "skill_bias",
                    "verse_skill_bias",
                    "verse_name",
                    "bad_dna_path",
                    "selector_model_path",
                    "expert_lookup_config",
                    "expert_dataset_map",
                    "expert_model_map",
                    "dataset_dir",
                    "expert_dataset_dir",
                    "expert_model_dir",
                    "dataset_path",
                    "dataset_paths",
                    "expert_performance",
                    "expert_performance_path",
                    "strict_config_validation",
                    # Compatibility: often set globally by trainer CLI.
                    "train",
                ],
                label="special_moe config",
            )
        expert_lookup_cfg = _coerce_dict(c.get("expert_lookup_config", {}))
        if strict and expert_lookup_cfg:
            _validate_allowed_keys(
                payload=expert_lookup_cfg,
                allowed=[
                    "enable_mlp_generalizer",
                    "enable_nn_fallback",
                    "nn_fallback_k",
                    "mlp_epochs",
                ],
                label="special_moe.expert_lookup_config",
            )
        dataset_paths_raw = c.get("dataset_paths")
        dataset_paths: List[str] = []
        if isinstance(dataset_paths_raw, list):
            dataset_paths = [str(p).strip() for p in dataset_paths_raw if str(p).strip()]
        elif isinstance(dataset_paths_raw, str) and dataset_paths_raw.strip():
            dataset_paths = [dataset_paths_raw.strip()]
        elif dataset_paths_raw not in (None,):
            raise ValueError("special_moe.dataset_paths must be list[str] or string")

        return SpecialMoEConfig(
            top_k=max(1, _safe_int(c.get("top_k", 2), default=2)),
            selector_temperature=max(1e-6, _safe_float(c.get("selector_temperature", 1.0), default=1.0)),
            selector_unknown_mix=max(0.0, min(1.0, _safe_float(c.get("selector_unknown_mix", 0.35), default=0.35))),
            state_keys=_as_str_list(c.get("state_keys"), default=["pos", "t", "x", "y", "agent"]),
            goal_keys=_as_str_list(c.get("goal_keys"), default=["goal", "goal_x", "goal_y", "target"]),
            skill_bias=_coerce_dict(c.get("skill_bias", {})),
            verse_skill_bias=_coerce_dict(c.get("verse_skill_bias", {})),
            verse_name=str(c.get("verse_name", "") or ""),
            expert_lookup_config=expert_lookup_cfg,
            expert_performance=_coerce_dict(c.get("expert_performance", {})),
            expert_performance_path=str(c.get("expert_performance_path", "") or "").strip(),
            bad_dna_path=str(c.get("bad_dna_path", "") or "").strip(),
            selector_model_path=str(c.get("selector_model_path", "") or "").strip(),
            expert_dataset_map=_coerce_dict(c.get("expert_dataset_map", {})),
            expert_model_map=_coerce_dict(c.get("expert_model_map", {})),
            dataset_dir=str(c.get("dataset_dir", "") or "").strip(),
            expert_dataset_dir=str(c.get("expert_dataset_dir", "") or "").strip(),
            expert_model_dir=str(c.get("expert_model_dir", "") or "").strip(),
            dataset_path=str(c.get("dataset_path", "") or "").strip(),
            dataset_paths=list(dataset_paths),
            strict_config_validation=bool(strict),
        )

    def to_runtime_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "top_k": int(self.top_k),
            "selector_temperature": float(self.selector_temperature),
            "selector_unknown_mix": float(self.selector_unknown_mix),
            "state_keys": list(self.state_keys),
            "goal_keys": list(self.goal_keys),
            "skill_bias": dict(self.skill_bias),
            "verse_skill_bias": dict(self.verse_skill_bias),
            "verse_name": str(self.verse_name),
            "expert_lookup_config": dict(self.expert_lookup_config),
            "expert_performance": dict(self.expert_performance),
            "expert_performance_path": str(self.expert_performance_path),
            "bad_dna_path": str(self.bad_dna_path),
            "selector_model_path": str(self.selector_model_path),
            "expert_dataset_map": dict(self.expert_dataset_map),
            "expert_model_map": dict(self.expert_model_map),
            "dataset_dir": str(self.dataset_dir),
            "expert_dataset_dir": str(self.expert_dataset_dir),
            "expert_model_dir": str(self.expert_model_dir),
            "dataset_path": str(self.dataset_path),
            "dataset_paths": list(self.dataset_paths),
            "strict_config_validation": bool(self.strict_config_validation),
        }
        return out


@dataclass
class _Expert:
    skill_id: str
    source_type: str
    source_path: str
    agent: Any
    selector_idx: Optional[int] = None
    performance_weight: float = 1.0


class SpecialMoEAgent:
    """
    Mixture-of-Experts variant of SpecialAgent.

    Config keys in AgentSpec.config:
    - expert_dataset_map: dict[str, str]  skill_id -> dataset path
    - expert_dataset_dir: str             directory containing <skill_id>.jsonl files
    - expert_model_map: dict[str, str]    skill_id -> distilled model dir/file
    - expert_model_dir: str               dir with distilled expert subdirs/files
    - dataset_path / dataset_paths / dataset_dir are also supported (fallback experts)
    - selector_model_path: str            torch checkpoint from tools/train_selector.py
    - top_k: int                          number of experts to activate (default 2)
    - selector_temperature: float         softmax temperature (default 1.0)
    - state_keys: list[str]               default ['pos','t','x','y','agent']
    - goal_keys: list[str]                default ['goal','goal_x','goal_y','target']
    - skill_bias: dict[str,float]         additive bias to selector logits per skill id
    - verse_skill_bias: dict[str,dict] or dict[str,list]
    - verse_name: str                     used to apply verse_skill_bias
    - bad_dna_path: str                   bad-DNA JSONL for boundary filtering
    - expert_lookup_config: dict          forwarded to ImitationLookupAgent experts
      (e.g. enable_mlp_generalizer, enable_nn_fallback, nn_fallback_k)
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        self._seed: Optional[int] = None
        self._rng = None
        import random
        self._random = random

        self._bad_obs: Set[str] = set()
        self._experts: List[_Expert] = []
        self._expert_by_skill: Dict[str, _Expert] = {}

        self._selector = None
        self._torch = None
        self._selector_state_dim = 0
        self._selector_vocab: Dict[str, int] = {}
        self._selector_idx_to_skill: Dict[int, str] = {}
        self._last_selector_stats: Dict[str, float] = {}

        cfg_obj = SpecialMoEConfig.from_dict(spec.config if isinstance(spec.config, dict) else {})
        cfg = cfg_obj.to_runtime_dict()
        self.top_k = int(cfg_obj.top_k)
        self.temperature = float(cfg_obj.selector_temperature)
        self.selector_unknown_mix = float(cfg_obj.selector_unknown_mix)
        self.state_keys = list(cfg_obj.state_keys)
        self.goal_keys = list(cfg_obj.goal_keys)

        self._skill_bias: Dict[str, float] = {
            str(k): float(v) for k, v in dict(cfg_obj.skill_bias).items()
        }
        self._verse_skill_bias = dict(cfg_obj.verse_skill_bias)
        self._verse_name = str(cfg_obj.verse_name or "")
        self._expert_lookup_cfg = dict(cfg_obj.expert_lookup_config)
        self._expert_performance = self._load_expert_performance(cfg)

        bad_path = cfg_obj.bad_dna_path
        if bad_path:
            self._bad_obs = load_bad_obs(str(bad_path))

        self._load_selector_if_available(str(cfg_obj.selector_model_path or ""))
        self._load_experts_from_config(cfg)

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = self._random.Random(seed)
        for exp in self._experts:
            exp.agent.seed(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        if self._rng is None:
            self.seed(self._seed)

        if obs_key(obs) in self._bad_obs:
            return ActionResult(action=self._sample_random_action(), info={"mode": "boundary_avoid"})

        if not self._experts:
            return ActionResult(action=self._sample_random_action(), info={"mode": "no_experts"})

        ranked = self._rank_experts(obs)
        selected = ranked[: min(self.top_k, len(ranked))]
        if not selected:
            return ActionResult(action=self._sample_random_action(), info={"mode": "no_ranked_experts"})

        actions: List[Tuple[float, JSONValue, str]] = []
        for weight, exp in selected:
            action = exp.agent.act(obs).action
            actions.append((float(weight), action, exp.skill_id))

        blended = self._blend_actions(actions)
        selected_expert = str(actions[0][2]) if actions else ""
        selector_conf = float(actions[0][0]) if actions else 0.0
        info = {
            "mode": "special_moe",
            "experts": [sid for _, _, sid in actions],
            "weights": [float(w) for w, _, _ in actions],
            "selector_active": bool(self._selector is not None),
            "selected_expert": selected_expert,
            "selector_confidence": float(selector_conf),
        }
        if self._last_selector_stats:
            info.update(dict(self._last_selector_stats))
        return ActionResult(action=blended, info=info)

    def policy_distribution(self, obs: JSONValue, *, top_k: Optional[int] = None) -> List[float]:
        if self.action_space.type != "discrete":
            return []
        n = _safe_int(self.action_space.n, default=0)
        if n <= 0:
            return []
        if not self._experts:
            return [1.0 / float(n) for _ in range(n)]

        ranked = self._rank_experts(obs)
        k = min(len(ranked), max(1, _safe_int(top_k if top_k is not None else self.top_k, self.top_k)))
        selected = ranked[:k]
        if not selected:
            return [1.0 / float(n) for _ in range(n)]

        probs = [0.0 for _ in range(n)]
        for weight, exp in selected:
            action = exp.agent.act(obs).action
            ai = _safe_int(action, -1)
            if 0 <= ai < n:
                probs[ai] += max(0.0, float(weight))

        total = sum(probs)
        if total <= 0.0:
            return [1.0 / float(n) for _ in range(n)]
        return [float(p) / float(total) for p in probs]

    def action_diagnostics(self, obs: JSONValue) -> Dict[str, JSONValue]:
        probs = self.policy_distribution(obs)
        if not probs:
            return {"sample_probs": [], "danger_scores": []}
        # MoE does not have explicit danger models; use inverse confidence as a conservative proxy.
        danger = [max(0.0, min(1.0, 1.0 - float(p))) for p in probs]
        return {
            "sample_probs": [float(p) for p in probs],
            "danger_scores": [float(d) for d in danger],
        }

    def learn(self, batch) -> Dict[str, JSONValue]:
        raise NotImplementedError("SpecialMoEAgent is inference-only. Train experts and selector offline.")

    def learn_from_dataset(self, dataset_path: str) -> Dict[str, JSONValue]:
        """
        Backward-compatible hook for Trainer dataset loading.
        Each dataset is treated as one expert keyed by file basename.
        """
        skill_id = os.path.splitext(os.path.basename(str(dataset_path)))[0]
        loaded = self._add_expert(skill_id=skill_id, dataset_path=str(dataset_path))
        return {"loaded": 1 if loaded else 0, "experts_total": len(self._experts)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "top_k": self.top_k,
            "selector_temperature": self.temperature,
            "state_keys": self.state_keys,
            "goal_keys": self.goal_keys,
            "experts": [
                {
                    "skill_id": exp.skill_id,
                    "source_type": exp.source_type,
                    "source_path": exp.source_path,
                    "selector_idx": exp.selector_idx,
                    "performance_weight": float(exp.performance_weight),
                }
                for exp in self._experts
            ],
        }
        with open(os.path.join(path, "special_moe.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        file_path = os.path.join(path, "special_moe.json")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"model file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.top_k = max(1, _safe_int(payload.get("top_k", self.top_k), self.top_k))
        self.temperature = max(1e-6, _safe_float(payload.get("selector_temperature", self.temperature), self.temperature))
        self.state_keys = list(payload.get("state_keys", self.state_keys))
        self.goal_keys = list(payload.get("goal_keys", self.goal_keys))

        experts = payload.get("experts", []) or []
        for e in experts:
            skill_id = str(e.get("skill_id", ""))
            src_type = str(e.get("source_type", "dataset"))
            src_path = str(e.get("source_path", ""))
            if src_type == "model":
                self._add_model_expert(skill_id=skill_id, model_path=src_path)
            else:
                self._add_expert(skill_id=skill_id, dataset_path=src_path)

    def close(self) -> None:
        for exp in self._experts:
            exp.agent.close()

    def _load_selector_if_available(self, model_path: str) -> None:
        if not model_path:
            return
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"selector_model_path not found: {model_path}")

        try:
            import torch
        except Exception as e:
            raise RuntimeError("selector_model_path provided but torch is unavailable") from e

        from models.micro_selector import MicroSelector

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        vocab = ckpt.get("vocab") or ckpt.get("lesson_vocab") or {}
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("Selector checkpoint missing non-empty vocab/lesson_vocab")

        state_dim = _safe_int(ckpt.get("state_dim", 0), default=0)
        if state_dim <= 0:
            raise ValueError("Selector checkpoint missing valid state_dim")

        model_cfg = _coerce_dict(ckpt.get("model_config", {}))
        n_embd = _safe_int(model_cfg.get("n_embd", 0), default=0)
        n_head = _safe_int(model_cfg.get("n_head", 0), default=0)
        n_layer = _safe_int(model_cfg.get("n_layer", 0), default=0)
        dropout = _safe_float(model_cfg.get("dropout", 0.0), default=0.0)
        ff_mult = _safe_int(model_cfg.get("ff_mult", 4), default=4)
        use_interaction_tokens = _safe_bool(model_cfg.get("use_interaction_tokens", False), default=False)
        use_cls_token = _safe_bool(model_cfg.get("use_cls_token", False), default=False)
        use_deep_stem = _safe_bool(model_cfg.get("use_deep_stem", False), default=False)
        pooling = str(model_cfg.get("pooling", "last") or "last").strip().lower()
        if pooling not in {"last", "mean", "cls"}:
            pooling = "last"

        # Backward-compatible inference for older selector checkpoints.
        state_dict = ckpt.get("model_state_dict", {}) or {}
        state_w = state_dict.get("state_encoder.weight")
        pos_emb = state_dict.get("pos_emb")
        if n_embd <= 0 and hasattr(state_w, "shape") and len(state_w.shape) == 2:
            n_embd = int(state_w.shape[0])
        if n_embd <= 0:
            n_embd = 256

        if n_layer <= 0:
            n_layer = 1
            for key in ckpt.get("model_state_dict", {}).keys():
                if key.startswith("transformer.layers."):
                    parts = key.split(".")
                    if len(parts) > 2:
                        idx = _safe_int(parts[2], default=0)
                        n_layer = max(n_layer, idx + 1)

        if n_head <= 0:
            for cand in [8, 4, 2, 1]:
                if n_embd % cand == 0:
                    n_head = cand
                    break
            if n_head <= 0:
                n_head = 1

        if (not isinstance(model_cfg.get("use_interaction_tokens"), bool)) or (
            not isinstance(model_cfg.get("use_cls_token"), bool)
        ):
            token_count = 0
            if hasattr(pos_emb, "shape") and len(pos_emb.shape) == 3:
                token_count = _safe_int(pos_emb.shape[1], default=0)
            if token_count == 2:
                use_interaction_tokens = False
                use_cls_token = False
            elif token_count == 3:
                use_interaction_tokens = False
                use_cls_token = True
            elif token_count == 4:
                use_interaction_tokens = True
                use_cls_token = False
            elif token_count == 5:
                use_interaction_tokens = True
                use_cls_token = True

        model = MicroSelector(
            state_dim=state_dim,
            skill_vocab_size=len(vocab),
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            use_interaction_tokens=use_interaction_tokens,
            use_cls_token=use_cls_token,
            ff_mult=ff_mult,
            use_deep_stem=use_deep_stem,
            pooling=pooling,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        self._torch = torch
        self._selector = model
        self._selector_state_dim = state_dim
        self._selector_vocab = {str(k): _safe_int(v) for k, v in vocab.items()}
        self._selector_idx_to_skill = {idx: skill for skill, idx in self._selector_vocab.items()}

    def _load_experts_from_config(self, cfg: Dict[str, Any]) -> None:
        skill_to_dataset: Dict[str, str] = {}
        skill_to_model: Dict[str, str] = {}

        explicit = _coerce_dict(cfg.get("expert_dataset_map", {}))
        if explicit:
            for skill, path in explicit.items():
                skill_to_dataset[str(skill)] = str(path)

        explicit_models = _coerce_dict(cfg.get("expert_model_map", {}))
        if explicit_models:
            for skill, path in explicit_models.items():
                skill_to_model[str(skill)] = str(path)

        dataset_dir = cfg.get("dataset_dir")
        expert_dataset_dir = cfg.get("expert_dataset_dir")
        for root in [expert_dataset_dir, dataset_dir]:
            if root and os.path.isdir(str(root)):
                for name in sorted(os.listdir(str(root))):
                    if not name.endswith(".jsonl"):
                        continue
                    skill_id = os.path.splitext(name)[0]
                    skill_to_dataset.setdefault(skill_id, os.path.join(str(root), name))

        expert_model_dir = cfg.get("expert_model_dir")
        if expert_model_dir and os.path.isdir(str(expert_model_dir)):
            for name in sorted(os.listdir(str(expert_model_dir))):
                full = os.path.join(str(expert_model_dir), name)
                skill_id = os.path.splitext(name)[0]
                if os.path.isdir(full):
                    if os.path.isfile(os.path.join(full, "distilled_policy.json")):
                        skill_to_model.setdefault(skill_id, full)
                elif name.endswith(".json"):
                    skill_to_model.setdefault(skill_id, full)

        dataset_path = cfg.get("dataset_path")
        if dataset_path:
            skill_to_dataset.setdefault(
                os.path.splitext(os.path.basename(str(dataset_path)))[0],
                str(dataset_path),
            )
        dataset_paths = cfg.get("dataset_paths") or []
        if isinstance(dataset_paths, list):
            for p in dataset_paths:
                skill_to_dataset.setdefault(
                    os.path.splitext(os.path.basename(str(p)))[0],
                    str(p),
                )

        for skill_id, ds_path in sorted(skill_to_dataset.items()):
            self._add_expert(skill_id=skill_id, dataset_path=ds_path)
        for skill_id, m_path in sorted(skill_to_model.items()):
            self._add_model_expert(skill_id=skill_id, model_path=m_path)

    def _selector_index_for_skill(self, skill: str) -> Optional[int]:
        if not self._selector_vocab:
            return None
        key = str(skill)
        candidates = [key]
        if key.endswith(".txt"):
            candidates.append(key[:-4])
        else:
            candidates.append(f"{key}.txt")
        for cand in candidates:
            idx = self._selector_vocab.get(cand)
            if idx is not None:
                return idx
        return None

    def _load_expert_performance(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        inline = _coerce_dict(cfg.get("expert_performance", {}))
        for k, v in inline.items():
            out[str(k)] = max(0.05, min(5.0, _safe_float(v, 1.0)))

        perf_path = str(cfg.get("expert_performance_path", "") or "").strip()
        if perf_path and os.path.isfile(perf_path):
            try:
                with open(perf_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        out[str(k)] = max(0.05, min(5.0, _safe_float(v, 1.0)))
            except Exception:
                pass
        return out

    def _add_expert(self, skill_id: str, dataset_path: str) -> bool:
        skill = str(skill_id).strip()
        path = str(dataset_path).strip()
        if not skill or not path:
            return False
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expert dataset not found for skill '{skill}': {path}")
        if skill in self._expert_by_skill:
            return False

        expert_spec = AgentSpec(
            spec_version=self.spec.spec_version,
            policy_id=f"{self.spec.policy_id}:{skill}",
            policy_version=self.spec.policy_version,
            algo="imitation_lookup",
            framework=self.spec.framework,
            seed=self.spec.seed,
            tags=list(self.spec.tags),
            config=(dict(self._expert_lookup_cfg) if self._expert_lookup_cfg else {}),
        )
        agent = ImitationLookupAgent(
            spec=expert_spec,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        agent.learn_from_dataset(path)
        if self._seed is not None:
            agent.seed(self._seed)

        selector_idx = self._selector_index_for_skill(skill)
        exp = _Expert(
            skill_id=skill,
            source_type="dataset",
            source_path=path,
            agent=agent,
            selector_idx=selector_idx,
            performance_weight=max(0.05, min(5.0, _safe_float(self._expert_performance.get(skill, 1.0), 1.0))),
        )
        self._experts.append(exp)
        self._expert_by_skill[skill] = exp
        return True

    def _add_model_expert(self, skill_id: str, model_path: str) -> bool:
        skill = str(skill_id).strip()
        path = str(model_path).strip()
        if not skill or not path:
            return False
        if skill in self._expert_by_skill:
            return False
        resolved = path
        if os.path.isdir(path):
            resolved = path
        elif os.path.isfile(path):
            # If a direct file path was provided, assume parent dir has distilled_policy.json
            resolved = os.path.dirname(path)
        if not os.path.isfile(os.path.join(resolved, "distilled_policy.json")):
            raise FileNotFoundError(
                f"Distilled expert model not found for skill '{skill}': {resolved}/distilled_policy.json"
            )

        expert_spec = AgentSpec(
            spec_version=self.spec.spec_version,
            policy_id=f"{self.spec.policy_id}:{skill}",
            policy_version=self.spec.policy_version,
            algo="distilled",
            framework=self.spec.framework,
            seed=self.spec.seed,
            tags=list(self.spec.tags),
            config={"model_path": resolved},
        )
        agent = DistilledAgent(
            spec=expert_spec,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        if self._seed is not None:
            agent.seed(self._seed)

        selector_idx = self._selector_index_for_skill(skill)
        exp = _Expert(
            skill_id=skill,
            source_type="model",
            source_path=resolved,
            agent=agent,
            selector_idx=selector_idx,
            performance_weight=max(0.05, min(5.0, _safe_float(self._expert_performance.get(skill, 1.0), 1.0))),
        )
        self._experts.append(exp)
        self._expert_by_skill[skill] = exp
        return True

    def _rank_experts(self, obs: JSONValue) -> List[Tuple[float, _Expert]]:
        # Fallback: uniform weighting if no selector.
        if self._selector is None or self._torch is None:
            w = 1.0 / float(len(self._experts))
            self._last_selector_stats = {
                "selector_margin": 0.0,
                "selector_entropy": 0.0,
                "selector_raw_confidence": float(w),
            }
            ranked = [(w, e) for e in self._experts]
            if _is_strategy_observation(obs):
                return self._cap_navigation_mass_for_strategy(ranked, nav_cap_total=0.095)
            return ranked

        state_raw = _flatten_obs(obs, self.state_keys)
        goal_raw = _flatten_obs(obs, self.goal_keys)
        state = _pad_or_truncate(state_raw, self._selector_state_dim)
        goal = _pad_or_truncate(goal_raw, self._selector_state_dim)

        t_state = self._torch.tensor(state, dtype=self._torch.float32).unsqueeze(0)
        t_goal = self._torch.tensor(goal, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._selector(t_state, t_goal).squeeze(0)

        # Global skill bias.
        if self._skill_bias:
            for skill, bias in self._skill_bias.items():
                idx = self._selector_index_for_skill(str(skill))
                if idx is not None and 0 <= idx < int(logits.shape[0]):
                    logits[idx] += float(bias)

        # Optional verse-wise bias if verse_name is set in config.
        verse_bias = self._verse_skill_bias.get(self._verse_name, {})
        if isinstance(verse_bias, dict):
            for skill, bias in verse_bias.items():
                idx = self._selector_index_for_skill(str(skill))
                if idx is not None and 0 <= idx < int(logits.shape[0]):
                    logits[idx] += float(bias)
        elif isinstance(verse_bias, list):
            for skill in verse_bias:
                idx = self._selector_index_for_skill(str(skill))
                if idx is not None and 0 <= idx < int(logits.shape[0]):
                    logits[idx] += 1.0

        # Guardrail: suppress navigation experts for clearly strategy-shaped observations.
        if _is_strategy_observation(obs):
            for exp in self._experts:
                if not _is_navigation_skill(exp.skill_id):
                    continue
                idx = exp.selector_idx
                if idx is None or idx < 0 or idx >= int(logits.shape[0]):
                    continue
                logits[idx] -= 4.5

        probs = self._torch.softmax(logits / float(self.temperature), dim=0)
        try:
            p_sorted, _ = self._torch.sort(probs, descending=True)
            top1 = float(p_sorted[0].item()) if int(p_sorted.shape[0]) >= 1 else 0.0
            top2 = float(p_sorted[1].item()) if int(p_sorted.shape[0]) >= 2 else 0.0
            margin = max(0.0, top1 - top2)
            entropy = float((-(probs * self._torch.log(probs + 1e-9))).sum().item())
        except Exception:
            top1 = 0.0
            margin = 0.0
            entropy = 0.0
        self._last_selector_stats = {
            "selector_margin": float(margin),
            "selector_entropy": float(entropy),
            "selector_raw_confidence": float(top1),
        }

        scored: List[Tuple[float, _Expert]] = []
        for exp in self._experts:
            if exp.selector_idx is not None and 0 <= exp.selector_idx < int(probs.shape[0]):
                weight = float(probs[exp.selector_idx].item()) * float(exp.performance_weight)
                scored.append((float(weight), exp))

        if not scored:
            # If selector vocab does not overlap with loaded expert skill IDs,
            # rank experts by their own confidence on this observation.
            conf_scored: List[Tuple[float, _Expert]] = []
            for e in self._experts:
                conf = 0.0
                try:
                    if hasattr(e.agent, "estimate_confidence"):
                        conf = float(e.agent.estimate_confidence(obs))
                except Exception:
                    conf = 0.0
                # Verse-aware boost for same-domain experts on unseen selector vocab.
                if self._verse_name:
                    skill_norm = str(e.skill_id).strip().lower()
                    if skill_norm == self._verse_name or skill_norm.startswith(f"{self._verse_name}."):
                        conf += 1.0
                conf_scored.append((max(0.0, conf) * float(e.performance_weight), e))
            total_conf = sum(c for c, _ in conf_scored)
            if total_conf <= 0.0:
                w = 1.0 / float(len(self._experts))
                ranked = [(w, e) for e in self._experts]
                if _is_strategy_observation(obs):
                    return self._cap_navigation_mass_for_strategy(ranked, nav_cap_total=0.095)
                return ranked
            conf_scored.sort(key=lambda x: x[0], reverse=True)
            top = conf_scored[: min(self.top_k, len(conf_scored))]
            top_sum = sum(c for c, _ in top)
            if top_sum <= 0.0:
                w = 1.0 / float(len(top))
                ranked = [(w, e) for _, e in top]
                if _is_strategy_observation(obs):
                    return self._cap_navigation_mass_for_strategy(ranked, nav_cap_total=0.095)
                return ranked
            ranked = [(c / top_sum, e) for c, e in top]
            if _is_strategy_observation(obs):
                return self._cap_navigation_mass_for_strategy(ranked, nav_cap_total=0.095)
            return ranked

        scored.sort(key=lambda x: x[0], reverse=True)
        # Blend selector-scored experts with selector-unknown experts using confidence.
        if self.selector_unknown_mix > 0.0:
            unknown_conf: List[Tuple[float, _Expert]] = []
            for exp in self._experts:
                if exp.selector_idx is not None:
                    continue
                conf = 0.0
                try:
                    if hasattr(exp.agent, "estimate_confidence"):
                        conf = float(exp.agent.estimate_confidence(obs))
                except Exception:
                    conf = 0.0
                if self._verse_name:
                    skill_norm = str(exp.skill_id).strip().lower()
                    if skill_norm == self._verse_name or skill_norm.startswith(f"{self._verse_name}."):
                        conf += 1.0
                unknown_conf.append((max(0.0, conf) * float(exp.performance_weight), exp))
            if unknown_conf:
                known_norm = sum(w for w, _ in scored)
                known_norm = known_norm if known_norm > 0.0 else 1.0
                known = [((w / known_norm) * (1.0 - self.selector_unknown_mix), e) for w, e in scored]
                unk_sum = sum(c for c, _ in unknown_conf)
                if unk_sum > 0.0:
                    unknown = [((c / unk_sum) * self.selector_unknown_mix, e) for c, e in unknown_conf]
                    scored = known + unknown
                else:
                    scored = known

        total = sum(w for w, _ in scored[: max(1, len(scored))])
        if total <= 0:
            w = 1.0 / float(len(scored))
            ranked = [(w, e) for _, e in scored]
        else:
            ranked = [(w / total, e) for w, e in scored]
        ranked.sort(key=lambda x: x[0], reverse=True)
        if _is_strategy_observation(obs):
            ranked = self._cap_navigation_mass_for_strategy(ranked, nav_cap_total=0.095)
        return ranked

    def _cap_navigation_mass_for_strategy(
        self,
        ranked: List[Tuple[float, _Expert]],
        *,
        nav_cap_total: float = 0.095,
    ) -> List[Tuple[float, _Expert]]:
        if not ranked:
            return ranked

        cap = max(0.0, min(1.0, float(nav_cap_total)))
        nav_total = 0.0
        non_nav_total = 0.0
        for w, exp in ranked:
            if _is_navigation_skill(exp.skill_id):
                nav_total += float(w)
            else:
                non_nav_total += float(w)

        if nav_total <= cap or nav_total <= 0.0 or non_nav_total <= 0.0:
            return ranked

        nav_scale = cap / nav_total
        non_nav_scale = (1.0 - cap) / non_nav_total
        adjusted: List[Tuple[float, _Expert]] = []
        for w, exp in ranked:
            if _is_navigation_skill(exp.skill_id):
                nw = float(w) * nav_scale
            else:
                nw = float(w) * non_nav_scale
            adjusted.append((max(0.0, nw), exp))

        total = sum(w for w, _ in adjusted)
        if total > 0.0:
            adjusted = [(w / total, exp) for w, exp in adjusted]
        adjusted.sort(key=lambda x: x[0], reverse=True)
        return adjusted

    def _blend_actions(self, actions: List[Tuple[float, JSONValue, str]]) -> JSONValue:
        if self.action_space.type == "discrete":
            scores: Dict[int, float] = {}
            for weight, action, _ in actions:
                a = _safe_int(action, default=0)
                scores[a] = scores.get(a, 0.0) + float(weight)
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            return int(best)

        if self.action_space.type == "continuous":
            vec_actions: List[Tuple[float, List[float]]] = []
            for weight, action, _ in actions:
                if isinstance(action, list):
                    numeric = [float(v) for v in action if isinstance(v, (int, float))]
                    if numeric:
                        vec_actions.append((float(weight), numeric))

            if not vec_actions:
                return self._sample_random_action()

            dim = min(len(v) for _, v in vec_actions)
            out = [0.0] * dim
            for weight, vec in vec_actions:
                for i in range(dim):
                    out[i] += float(weight) * float(vec[i])
            return out

        # For unsupported action spaces, pick highest-weight expert action.
        return actions[0][1]

    def _sample_random_action(self) -> JSONValue:
        if self._rng is None:
            self.seed(self._seed)

        if self.action_space.type == "discrete":
            n = _safe_int(self.action_space.n, default=0)
            if n <= 0:
                raise ValueError("discrete action space requires n > 0")
            return int(self._rng.randrange(n))

        if self.action_space.type == "continuous":
            low = self.action_space.low or []
            high = self.action_space.high or []
            if len(low) != len(high):
                raise ValueError("continuous action space requires equal low/high lengths")
            return [float(self._rng.uniform(float(low[i]), float(high[i]))) for i in range(len(low))]

        raise ValueError(f"Unsupported action space type: {self.action_space.type}")
