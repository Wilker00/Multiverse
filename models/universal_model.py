"""
models/universal_model.py

Deployable wrapper around centralized memory + scenario matcher.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.types import JSONValue
from memory.central_repository import CentralMemoryConfig
from memory.embeddings import obs_to_vector
from orchestrator.scenario_matcher import ScenarioAdvice, ScenarioRequest, recommend_action


@dataclass
class UniversalModelConfig:
    memory_dir: str = "central_memory"
    default_top_k: int = 5
    default_min_score: float = 0.0
    default_verse_name: Optional[str] = None
    meta_model_path: Optional[str] = None
    meta_confidence_threshold: float = 0.35
    prefer_meta_policy: bool = False
    meta_history_len: int = 6
    learned_bridge_enabled: bool = False
    learned_bridge_model_path: Optional[str] = None
    learned_bridge_score_weight: float = 0.35


class UniversalModel:
    """
    Thin model interface backed by shared memory retrieval.
    """

    def __init__(self, config: UniversalModelConfig):
        self.config = config
        self.memory_cfg = CentralMemoryConfig(root_dir=config.memory_dir)
        self._meta_torch = None
        self._meta_model = None
        self._meta_state_dim = 0
        self._meta_action_dim = 0
        self._meta_context_dim = 0
        self._meta_n_embd = 0
        self._meta_use_generalized_input = False
        self._meta_history_len = max(1, int(config.meta_history_len))
        self._load_meta_if_available()

    def _load_meta_if_available(self) -> None:
        path = self.config.meta_model_path
        if not path:
            return
        if not os.path.isfile(path):
            return
        try:
            import torch
        except Exception:
            return
        from models.meta_transformer import MetaTransformer

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_config", {})
        if not isinstance(model_cfg, dict):
            return
        state_dim = int(model_cfg.get("state_dim", 0) or 0)
        action_dim = int(model_cfg.get("action_dim", 0) or 0)
        n_embd = int(model_cfg.get("n_embd", 256) or 256)
        use_generalized_input = bool(int(model_cfg.get("use_generalized_input", 0) or 0))
        if action_dim <= 0:
            return
        if not use_generalized_input and state_dim <= 0:
            return
        model = MetaTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            n_embd=n_embd,
            use_generalized_input=use_generalized_input,
        )
        load_state = model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        missing = list(getattr(load_state, "missing_keys", []))
        unexpected = list(getattr(load_state, "unexpected_keys", []))
        if missing or unexpected:
            return
        model.eval()

        self._meta_torch = torch
        self._meta_model = model
        self._meta_state_dim = state_dim
        self._meta_action_dim = action_dim
        self._meta_n_embd = n_embd
        self._meta_use_generalized_input = use_generalized_input
        self._meta_context_dim = (n_embd + 2) if use_generalized_input else (state_dim + 2)
        self._meta_history_len = int(ckpt.get("history_len", self._meta_history_len) or self._meta_history_len)

    def _pad(self, values: List[float], size: int) -> List[float]:
        if len(values) >= size:
            return values[:size]
        return values + [0.0] * (size - len(values))

    def _build_meta_inputs(
        self,
        *,
        obs: JSONValue,
        recent_history: Optional[List[Dict[str, JSONValue]]],
    ):
        if self._meta_model is None or self._meta_torch is None:
            return None, None

        if self._meta_use_generalized_input:
            history_rows: List[List[float]] = []
            if isinstance(recent_history, list):
                for item in recent_history[-self._meta_history_len :]:
                    if not isinstance(item, dict):
                        continue
                    h_obs = item.get("obs")
                    h_action = item.get("action")
                    h_reward = item.get("reward")
                    try:
                        with self._meta_torch.no_grad():
                            hv_tensor = self._meta_model.input_encoder([h_obs])  # type: ignore[attr-defined]
                        hv = [float(x) for x in hv_tensor.squeeze(0).tolist()]
                    except Exception:
                        continue
                    try:
                        a = float(h_action)
                    except Exception:
                        a = 0.0
                    try:
                        r = float(h_reward)
                    except Exception:
                        r = 0.0
                    history_rows.append(hv + [a, r])
            if len(history_rows) < self._meta_history_len:
                pad_rows = self._meta_history_len - len(history_rows)
                history_rows = ([[0.0] * self._meta_context_dim] * pad_rows) + history_rows
            t_hist = self._meta_torch.tensor([history_rows], dtype=self._meta_torch.float32)
            return None, t_hist

        try:
            obs_vec = obs_to_vector(obs)
        except Exception:
            return None, None
        state_vec = self._pad(obs_vec, self._meta_state_dim)

        history_rows: List[List[float]] = []
        if isinstance(recent_history, list):
            for item in recent_history[-self._meta_history_len :]:
                if not isinstance(item, dict):
                    continue
                h_obs = item.get("obs")
                h_action = item.get("action")
                h_reward = item.get("reward")
                try:
                    hv = self._pad(obs_to_vector(h_obs), self._meta_state_dim)
                except Exception:
                    continue
                try:
                    a = float(h_action)  # discrete scalar action
                except Exception:
                    a = 0.0
                try:
                    r = float(h_reward)
                except Exception:
                    r = 0.0
                history_rows.append(hv + [a, r])

        if len(history_rows) < self._meta_history_len:
            pad_rows = self._meta_history_len - len(history_rows)
            history_rows = ([[0.0] * self._meta_context_dim] * pad_rows) + history_rows

        t_state = self._meta_torch.tensor([state_vec], dtype=self._meta_torch.float32)
        t_hist = self._meta_torch.tensor([history_rows], dtype=self._meta_torch.float32)
        return t_state, t_hist

    def _predict_meta(
        self,
        *,
        obs: JSONValue,
        recent_history: Optional[List[Dict[str, JSONValue]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self._meta_model is None or self._meta_torch is None:
            return None
        t_state, t_hist = self._build_meta_inputs(obs=obs, recent_history=recent_history)
        if t_hist is None:
            return None
        with self._meta_torch.no_grad():
            if self._meta_use_generalized_input:
                logits = self._meta_model(state=None, recent_history=t_hist, raw_obs=[obs])
            else:
                if t_state is None:
                    return None
                logits = self._meta_model(t_state, t_hist)
            probs = self._meta_torch.softmax(logits, dim=-1).squeeze(0)
            conf, action = self._meta_torch.max(probs, dim=-1)
        weights = {str(i): float(probs[i].item()) for i in range(int(probs.shape[0]))}
        return {
            "action": int(action.item()),
            "confidence": float(conf.item()),
            "weights": weights,
            "matched": 0,
            "strategy": "meta_transformer",
            "bridge_source_verse": None,
            "meta_used": True,
        }

    def recommend(
        self,
        *,
        obs: JSONValue,
        verse_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_run_ids: Optional[List[str]] = None,
    ) -> Optional[ScenarioAdvice]:
        req = ScenarioRequest(
            obs=obs,
            verse_name=verse_name if verse_name is not None else self.config.default_verse_name,
            top_k=int(top_k if top_k is not None else self.config.default_top_k),
            min_score=float(min_score if min_score is not None else self.config.default_min_score),
            exclude_run_ids=exclude_run_ids or [],
            learned_bridge_enabled=bool(self.config.learned_bridge_enabled),
            learned_bridge_model_path=self.config.learned_bridge_model_path,
            learned_bridge_score_weight=float(self.config.learned_bridge_score_weight),
        )
        return recommend_action(request=req, cfg=self.memory_cfg)

    def predict(
        self,
        *,
        obs: JSONValue,
        verse_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_run_ids: Optional[List[str]] = None,
        recent_history: Optional[List[Dict[str, JSONValue]]] = None,
    ) -> Dict[str, Any]:
        advice = self.recommend(
            obs=obs,
            verse_name=verse_name,
            top_k=top_k,
            min_score=min_score,
            exclude_run_ids=exclude_run_ids,
        )
        retrieval_pred: Optional[Dict[str, Any]] = None
        if advice is None:
            retrieval_pred = None
        else:
            retrieval_pred = {
                "action": advice.action,
                "confidence": float(advice.confidence),
                "matched": len(advice.matches),
                "weights": advice.weights,
                "strategy": advice.strategy,
                "bridge_source_verse": advice.bridge_source_verse,
                "meta_used": False,
            }

        meta_pred = None
        should_try_meta = self._meta_model is not None and (
            bool(self.config.prefer_meta_policy)
            or retrieval_pred is None
            or float(retrieval_pred.get("confidence", 0.0)) < float(self.config.meta_confidence_threshold)
        )
        if should_try_meta:
            meta_pred = self._predict_meta(obs=obs, recent_history=recent_history)

        if retrieval_pred is None and meta_pred is not None:
            return meta_pred
        if retrieval_pred is not None and meta_pred is None:
            return retrieval_pred
        if retrieval_pred is not None and meta_pred is not None:
            if float(meta_pred["confidence"]) > float(retrieval_pred["confidence"]):
                return meta_pred
            return retrieval_pred

        return {
                "action": None,
                "confidence": 0.0,
                "matched": 0,
                "weights": {},
                "strategy": "none",
                "bridge_source_verse": None,
                "meta_used": False,
            }

    def save(self, out_dir: str, *, snapshot_memory: bool = False) -> str:
        os.makedirs(out_dir, exist_ok=True)
        cfg = {
            "memory_dir": self.config.memory_dir,
            "default_top_k": int(self.config.default_top_k),
            "default_min_score": float(self.config.default_min_score),
            "default_verse_name": self.config.default_verse_name,
            "meta_model_path": self.config.meta_model_path,
            "meta_confidence_threshold": float(self.config.meta_confidence_threshold),
            "prefer_meta_policy": bool(self.config.prefer_meta_policy),
            "meta_history_len": int(self.config.meta_history_len),
            "learned_bridge_enabled": bool(self.config.learned_bridge_enabled),
            "learned_bridge_model_path": self.config.learned_bridge_model_path,
            "learned_bridge_score_weight": float(self.config.learned_bridge_score_weight),
            "created_at_ms": int(time.time() * 1000),
        }

        if snapshot_memory:
            snap_dir = os.path.join(out_dir, "central_memory")
            os.makedirs(snap_dir, exist_ok=True)
            for name in ("memories.jsonl", "dedupe_index.json"):
                src = os.path.join(self.config.memory_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(snap_dir, name))
            cfg["memory_dir"] = "central_memory"
        if self.config.meta_model_path and os.path.isfile(self.config.meta_model_path):
            meta_dst = os.path.join(out_dir, "meta_transformer.pt")
            shutil.copy2(self.config.meta_model_path, meta_dst)
            cfg["meta_model_path"] = "meta_transformer.pt"
        if self.config.learned_bridge_model_path and os.path.isfile(self.config.learned_bridge_model_path):
            bridge_dst = os.path.join(out_dir, "contrastive_bridge.pt")
            shutil.copy2(self.config.learned_bridge_model_path, bridge_dst)
            cfg["learned_bridge_model_path"] = "contrastive_bridge.pt"

        cfg_path = os.path.join(out_dir, "model_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return cfg_path

    @staticmethod
    def load(model_dir: str) -> "UniversalModel":
        cfg_path = os.path.join(model_dir, "model_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Model config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory_dir = str(data.get("memory_dir", "central_memory"))
        # Resolve relative memory path against model directory.
        if not os.path.isabs(memory_dir):
            memory_dir = os.path.join(model_dir, memory_dir)

        cfg = UniversalModelConfig(
            memory_dir=memory_dir,
            default_top_k=int(data.get("default_top_k", 5)),
            default_min_score=float(data.get("default_min_score", 0.0)),
            default_verse_name=data.get("default_verse_name"),
            meta_model_path=(
                os.path.join(model_dir, str(data.get("meta_model_path")))
                if data.get("meta_model_path") and not os.path.isabs(str(data.get("meta_model_path")))
                else data.get("meta_model_path")
            ),
            meta_confidence_threshold=float(data.get("meta_confidence_threshold", 0.35)),
            prefer_meta_policy=bool(data.get("prefer_meta_policy", False)),
            meta_history_len=int(data.get("meta_history_len", 6)),
            learned_bridge_enabled=bool(data.get("learned_bridge_enabled", False)),
            learned_bridge_model_path=(
                os.path.join(model_dir, str(data.get("learned_bridge_model_path")))
                if data.get("learned_bridge_model_path")
                and not os.path.isabs(str(data.get("learned_bridge_model_path")))
                else data.get("learned_bridge_model_path")
            ),
            learned_bridge_score_weight=float(data.get("learned_bridge_score_weight", 0.35)),
        )
        return UniversalModel(cfg)
