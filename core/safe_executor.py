"""
core/safe_executor.py

Runtime safety wrapper with three controls:
1) Competence Shield: veto high-risk/low-confidence actions.
2) Recursive Fallback: temporarily route control to a safer fallback policy.
3) Checkpoint Recovery: rewind to last safe checkpoint after dangerous outcomes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

from core.agent_base import ActionResult
from core.mcts_search import AgentPolicyPrior, MCTSConfig, MCTSSearch, MetaTransformerValue
from core.planning_budget import PlanningBudget, PlanningBudgetConfig
from core.runtime_confidence import RuntimeConfidenceMonitor
from core.safe_executor_runtime_support import (
    can_checkpoint,
    can_rewind,
    check_shield,
    choose_safe_alternative,
    mcts_takeover,
    planner_takeover,
    rewind_to_checkpoint,
    save_checkpoint,
)
from core.safe_executor_support import (
    SafeExecutorConfig,
    _danger_label_from_info,
    _default_mcts_overrides,
    _infer_failure_mode,
    _is_nan_action,
    _is_safety_violation,
    _obs_key,
    _project_vector,
    _safe_float,
    _safe_int,
)
from core.types import JSONValue
from core.verse_base import StepResult
from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector


class SafeExecutor:
    def __init__(
        self,
        *,
        config: SafeExecutorConfig,
        verse: Any,
        fallback_agent: Optional[Any] = None,
    ):
        self.config = config
        self.verse = verse
        self.fallback_agent = fallback_agent

        self._fallback_remaining = 0
        self._rewinds_used = 0
        self._confidence_torch = None
        self._confidence_model = None
        self._monitor = RuntimeConfidenceMonitor(
            window_size=max(1, int(config.competence_window)),
            min_competence_rate=float(config.min_competence_rate),
            min_action_confidence=float(config.min_action_confidence),
            danger_threshold=float(config.danger_threshold),
        )
        self._planning_budget = PlanningBudget(
            PlanningBudgetConfig(
                enabled=bool(config.planner_enabled),
                base_threshold=float(config.planner_confidence_threshold),
                regret_adaptation=float(config.planning_regret_adaptation),
                budget_per_episode=int(config.planning_budget_per_episode),
                budget_per_minute=int(config.planning_budget_per_minute),
                history_window=max(8, int(config.competence_window) * 4),
            )
        )

        self._checkpoint_state: Optional[Dict[str, JSONValue]] = None
        self._checkpoint_obs: Optional[JSONValue] = None
        self._checkpoint_agent_state: Optional[Dict[str, Any]] = None
        self._blocked_actions: Dict[str, Set[int]] = {}
        self._planner_buffer: List[int] = []
        self._planner_takeover_remaining: int = 0
        self._mcts: Optional[MCTSSearch] = None
        self._mcts_value_net: Optional[Any] = None
        self._last_mcts_result: Dict[str, JSONValue] = {}
        self._shield: Optional[Any] = None # Will lazy init
        self._runtime_error_counts: Dict[str, int] = {}
        self._runtime_error_recent: List[Dict[str, Any]] = []
        self._danger_clusters: List[Dict[str, Any]] = []
        self._danger_map_embedding_dim: int = 0
        if self.config.danger_map_path and os.path.isfile(self.config.danger_map_path):
            try:
                with open(self.config.danger_map_path, "r", encoding="utf-8") as f:
                    danger_map = json.load(f)
                    self._danger_clusters = danger_map.get("clusters", [])
                    self._danger_map_embedding_dim = int(danger_map.get("embedding_dim", 0))
            except Exception as exc:
                self._danger_clusters = []
                self._danger_map_embedding_dim = 0
                self._record_runtime_error(
                    code="danger_map_load_error",
                    exc=exc,
                    context="safe_executor.danger_map",
                )
        self._failure_signatures: List[Dict[str, Any]] = []
        if self.config.failure_signature_path and os.path.isfile(self.config.failure_signature_path):
            self._failure_signatures = self._load_failure_signatures(self.config.failure_signature_path)
        self._load_confidence_model_if_available(self.config.confidence_model_path)

        vname = ""
        try:
            vname = str(getattr(getattr(self.verse, "spec", None), "verse_name", "")).strip().lower()
        except Exception as exc:
            vname = ""
            self._record_runtime_error(
                code="verse_name_resolution_error",
                exc=exc,
                context="safe_executor.verse_name",
            )
        self._apply_mcts_overrides_for_verse(vname)
        allow = set(str(x).strip().lower() for x in (self.config.planner_verse_allowlist or []))
        self._planner_active = bool(self.config.planner_enabled) and (not allow or vname in allow)
        self._mcts_active = bool(self.config.mcts_enabled)
        if self._mcts_active:
            try:
                self._mcts = MCTSSearch(
                    verse=self.verse,
                    config=MCTSConfig(
                        num_simulations=int(self.config.mcts_num_simulations),
                        max_depth=int(self.config.mcts_max_depth),
                        c_puct=float(self.config.mcts_c_puct),
                        discount=float(self.config.mcts_discount),
                        forced_loss_threshold=float(self.config.mcts_loss_threshold),
                        forced_loss_min_visits=int(self.config.mcts_min_visits),
                        value_confidence_threshold=float(self.config.mcts_value_confidence_threshold),
                    ),
                )
            except Exception as exc:
                self._mcts = None
                self._mcts_active = False
                self._record_runtime_error(
                    code="mcts_init_error",
                    exc=exc,
                    context="safe_executor.mcts.init",
                )
        if self._mcts_active and str(self.config.mcts_meta_model_path).strip():
            try:
                self._mcts_value_net = MetaTransformerValue(
                    checkpoint_path=str(self.config.mcts_meta_model_path).strip(),
                    history_len=int(self.config.mcts_meta_history_len),
                )
            except Exception as exc:
                self._mcts_value_net = None
                self._record_runtime_error(
                    code="mcts_value_model_load_error",
                    exc=exc,
                    context="safe_executor.mcts.value_model",
                )

        self._episode_counters: Dict[str, int] = {
            "shield_vetoes": 0,
            "fallback_actions": 0,
            "rewinds": 0,
            "dangerous_outcomes": 0,
            "planner_actions": 0,
            "planner_queries": 0,
            "mcts_queries": 0,
            "mcts_vetoes": 0,
            "runtime_errors": 0,
        }
        self._steps_observed: int = 0

    def _record_runtime_error(self, *, code: str, exc: Exception, context: str = "") -> None:
        key = str(code or "").strip() or "unknown_error"
        self._runtime_error_counts[key] = int(self._runtime_error_counts.get(key, 0)) + 1
        if isinstance(getattr(self, "_episode_counters", None), dict):
            self._episode_counters["runtime_errors"] = int(self._episode_counters.get("runtime_errors", 0)) + 1
        event = {
            "code": key,
            "context": str(context),
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }
        self._runtime_error_recent.append(event)
        if len(self._runtime_error_recent) > 20:
            self._runtime_error_recent = self._runtime_error_recent[-20:]

    def _load_confidence_model_if_available(self, model_path: str) -> None:
        path = str(model_path or "").strip()
        if not path or not os.path.isfile(path):
            return
        try:
            import torch  # type: ignore
            from models.confidence_monitor import load_confidence_monitor
        except Exception as exc:
            self._confidence_torch = None
            self._confidence_model = None
            self._record_runtime_error(
                code="confidence_model_import_error",
                exc=exc,
                context="safe_executor.confidence_model.import",
            )
            return
        try:
            model = load_confidence_monitor(path, map_location="cpu")
            model.eval()
        except Exception as exc:
            self._confidence_torch = None
            self._confidence_model = None
            self._record_runtime_error(
                code="confidence_model_load_error",
                exc=exc,
                context="safe_executor.confidence_model.load",
            )
            return
        self._confidence_torch = torch
        self._confidence_model = model

    def _predict_confidence_model_danger(self, *, obs: JSONValue, action: int) -> Optional[float]:
        if self._confidence_model is None or self._confidence_torch is None:
            return None
        try:
            obs_vec = obs_to_universal_vector(obs, dim=int(self.config.confidence_model_obs_dim))
            if not obs_vec:
                return None
            a = max(-10.0, min(10.0, float(action)))
            # Keep inference features aligned with the trained monitor input dim.
            # Newer checkpoints are trained on:
            #   [obs_u..., action_norm, se_danger, se_confidence, bias]
            # while older checkpoints may only use:
            #   [obs_u..., action_norm, bias]
            obs_list = list(obs_vec)
            action_norm = float(a) / 10.0
            target_dim = 0
            try:
                model_cfg = getattr(self._confidence_model, "cfg", None)
                target_dim = int(getattr(model_cfg, "input_dim", 0) or 0)
            except Exception:
                target_dim = 0
            legacy_dim = len(obs_list) + 2
            extended_dim = len(obs_list) + 4
            if target_dim == extended_dim:
                features = obs_list + [action_norm, 0.0, 1.0, 1.0]
            elif target_dim <= 0 or target_dim == legacy_dim:
                features = obs_list + [action_norm, 1.0]
            else:
                # Generic fallback for unexpected model dims: pad/truncate to target.
                features = obs_list + [action_norm, 0.0, 1.0, 1.0]
                if len(features) < target_dim:
                    features = features + [0.0] * int(target_dim - len(features))
                    features[-1] = 1.0
                elif len(features) > target_dim:
                    features = features[:target_dim]
                    if features:
                        features[-1] = 1.0
            t = self._confidence_torch.tensor(features, dtype=self._confidence_torch.float32).unsqueeze(0)
            with self._confidence_torch.no_grad():
                prob = self._confidence_model.predict_danger_prob(t).squeeze(0).item()
            return max(0.0, min(1.0, float(prob)))
        except Exception as exc:
            self._record_runtime_error(
                code="confidence_model_predict_error",
                exc=exc,
                context="safe_executor.confidence_model.predict",
            )
            return None

    def _get_danger_map_match(self, obs: JSONValue) -> Optional[Dict[str, Any]]:
        if not self._danger_clusters or self._danger_map_embedding_dim <= 0:
            return None
        try:
            raw_vec = obs_to_vector(obs)
            if not raw_vec:
                return None
            vec = _project_vector(raw_vec, dim=self._danger_map_embedding_dim)
            best_match: Optional[Dict[str, Any]] = None
            max_sim = -1.0
            for cluster in self._danger_clusters:
                centroid = cluster.get("centroid")
                if not isinstance(centroid, list) or len(centroid) != len(vec):
                    continue
                dot = sum(v * c for v, c in zip(vec, centroid))
                if dot > max_sim:
                    max_sim = dot
                    best_match = cluster
            if best_match and max_sim >= self.config.danger_map_similarity_threshold:
                return {"cluster": best_match, "similarity": max_sim}
        except Exception as exc:
            self._record_runtime_error(
                code="danger_map_match_error",
                exc=exc,
                context="safe_executor.danger_map.match",
            )
            return None
        return None

    def _load_failure_signatures(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        dim = max(8, int(self.config.failure_signature_embedding_dim))
        try:
            if str(path).lower().endswith(".jsonl"):
                rows = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        rows.append(obj)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and isinstance(obj.get("signatures"), list):
                    rows = obj.get("signatures", [])
                elif isinstance(obj, list):
                    rows = obj
                else:
                    rows = []
        except Exception as exc:
            rows = []
            self._record_runtime_error(
                code="failure_signature_load_error",
                exc=exc,
                context="safe_executor.failure_signature.load",
            )
        for row in rows:
            if not isinstance(row, dict):
                continue
            failure_family = str(row.get("failure_family", "")).strip().lower()
            if failure_family and failure_family not in {"declarative", "declarative_fail", "declarative_failure"}:
                continue
            raw_vec = row.get("obs_vector")
            vec: List[float] = []
            if isinstance(raw_vec, list):
                try:
                    vec = [float(v) for v in raw_vec]
                except Exception:
                    vec = []
            if not vec:
                try:
                    vec = obs_to_vector(row.get("obs"))
                except Exception:
                    vec = []
            if not vec:
                continue
            proj = _project_vector(vec, dim=dim)
            avoid_action = _safe_int(row.get("avoid_action", -1), -1)
            out.append(
                {
                    "vector": proj,
                    "avoid_action": avoid_action,
                    "source_verse": str(row.get("source_verse", "")),
                    "failure_type": str(row.get("failure_type", row.get("failure_mode", "declarative_signature"))),
                    "cluster_id": _safe_int(row.get("cluster_id", -1), -1),
                    "risk_score": _safe_float(row.get("risk_score", 0.0), 0.0),
                }
            )
        return out

    def _get_failure_signature_match(self, obs: JSONValue, action: int) -> Optional[Dict[str, Any]]:
        if not self._failure_signatures:
            return None
        try:
            raw_vec = obs_to_vector(obs)
            if not raw_vec:
                return None
            qvec = _project_vector(raw_vec, dim=max(8, int(self.config.failure_signature_embedding_dim)))
            best: Optional[Dict[str, Any]] = None
            best_sim = -1.0
            for sig in self._failure_signatures:
                vec = sig.get("vector")
                if not isinstance(vec, list) or len(vec) != len(qvec):
                    continue
                sim = cosine_similarity(qvec, [float(v) for v in vec])
                if sim > best_sim:
                    best_sim = float(sim)
                    best = sig
            if best is None:
                return None
            if best_sim < float(self.config.failure_signature_similarity_threshold):
                return None
            avoid_action = _safe_int(best.get("avoid_action", -1), -1)
            if avoid_action >= 0 and action >= 0 and avoid_action != int(action):
                return None
            return {
                "similarity": float(best_sim),
                "avoid_action": int(avoid_action),
                "cluster_id": _safe_int(best.get("cluster_id", -1), -1),
                "source_verse": str(best.get("source_verse", "")),
                "failure_type": str(best.get("failure_type", "declarative_signature")),
                "risk_score": _safe_float(best.get("risk_score", 0.0), 0.0),
            }
        except Exception as exc:
            self._record_runtime_error(
                code="failure_signature_match_error",
                exc=exc,
                context="safe_executor.failure_signature.match",
            )
            return None

    def _generate_veto_explanation(
        self,
        *,
        low_conf: bool,
        high_danger: bool,
        blocked: bool,
        danger_map_match: Optional[Dict[str, Any]],
        failure_signature_match: Optional[Dict[str, Any]] = None,
    ) -> str:
        reasons = []
        if blocked:
            reasons.append("action is blocked due to a previous failure in a similar state")
        if low_conf:
            reasons.append("action has low confidence")
        if high_danger:
            reasons.append("action has a high predicted danger score")

        if danger_map_match:
            similarity = danger_map_match.get("similarity", 0.0)
            cluster = danger_map_match.get("cluster", {})
            cluster_id = cluster.get("cluster_id", -1)
            verse_counts = cluster.get("verse_counts", {})
            top_verse = max(verse_counts, key=verse_counts.get) if verse_counts else "unknown verse"
            reasons.append(
                f"state is {similarity:.1%} similar to known danger pattern #{cluster_id} "
                f"(most common in '{top_verse}')"
            )
        if failure_signature_match:
            similarity = _safe_float(failure_signature_match.get("similarity", 0.0), 0.0)
            source_verse = str(failure_signature_match.get("source_verse", "unknown verse"))
            failure_type = str(failure_signature_match.get("failure_type", "declarative_signature"))
            reasons.append(
                f"state matches declarative failure signature ({failure_type}) at {similarity:.1%} similarity from '{source_verse}'"
            )

        if not reasons:
            return "Vetoing action for general safety reasons."

        return f"Vetoing action because {', and '.join(reasons)}."

    def reset_episode(self, seed: Optional[int]) -> None:
        self._fallback_remaining = 0
        self._rewinds_used = 0
        self._monitor.reset()
        self._checkpoint_state = None
        self._checkpoint_obs = None
        self._checkpoint_agent_state = None
        self._blocked_actions.clear()
        self._planner_buffer = []
        self._planner_takeover_remaining = 0
        self._last_mcts_result = {}
        self._runtime_error_counts = {}
        self._runtime_error_recent = []
        self._episode_counters = {
            "shield_vetoes": 0,
            "fallback_actions": 0,
            "rewinds": 0,
            "dangerous_outcomes": 0,
            "planner_actions": 0,
            "planner_queries": 0,
            "mcts_queries": 0,
            "mcts_vetoes": 0,
            "runtime_errors": 0,
        }
        self._steps_observed = 0
        self._planning_budget.reset_episode()
        if self._mcts is not None:
            try:
                self._mcts.seed(seed)
            except Exception as exc:
                self._record_runtime_error(
                    code="mcts_seed_error",
                    exc=exc,
                    context="safe_executor.mcts.seed",
                )
        if self.fallback_agent is not None:
            try:
                self.fallback_agent.seed(seed)
            except Exception as exc:
                self._record_runtime_error(
                    code="fallback_seed_error",
                    exc=exc,
                    context="safe_executor.fallback.seed",
                )

    def select_action(self, primary_agent: Any, obs: JSONValue) -> ActionResult:
        if not self.config.enabled:
            return primary_agent.act(obs)

        if self._planner_active and self._planner_takeover_remaining > 0 and self._planner_buffer:
            chosen = int(self._planner_buffer.pop(0))
            self._planner_takeover_remaining = max(0, int(self._planner_takeover_remaining - 1))
            self._episode_counters["planner_actions"] = int(self._episode_counters["planner_actions"]) + 1
            return ActionResult(
                action=chosen,
                info={
                    "safe_executor": {
                        "mode": "planner_takeover",
                        "planner_remaining": int(self._planner_takeover_remaining),
                    }
                },
            )

        force_fallback = bool(self._fallback_remaining > 0 and self.fallback_agent is not None)
        if force_fallback:
            self._fallback_remaining = max(0, int(self._fallback_remaining - 1))
            self._episode_counters["fallback_actions"] = int(self._episode_counters["fallback_actions"]) + 1
            fres = self.fallback_agent.act(obs)
            finfo = dict(fres.info or {})
            finfo["safe_executor"] = {
                "mode": "fallback",
                "fallback_remaining": int(self._fallback_remaining),
                "shield_veto": False,
            }
            return ActionResult(action=fres.action, info=finfo)

        pres = primary_agent.act(obs)
        p_action = _safe_int(pres.action, -1)
        blocked = self._is_action_blocked(obs, p_action)
        risk = self._estimate_action_risk(primary_agent, obs, p_action)
        eff_conf, eff_danger, veto_adaptation = self._effective_veto_thresholds()
        low_conf = bool(risk["confidence"] < float(eff_conf))
        high_danger = bool(risk["danger"] >= float(eff_danger))
        failure_signature_match = self._get_failure_signature_match(obs, p_action)
        memory_forced_veto = bool(failure_signature_match is not None)
        if memory_forced_veto:
            blocked = True
            high_danger = True
        planner_low_conf = bool(risk["confidence"] < float(self.config.planner_confidence_threshold))
        mcts_trigger = bool(
            self._mcts_active
            and self._mcts is not None
            and (
                (blocked and bool(self.config.mcts_trigger_on_block))
                or (low_conf and bool(self.config.mcts_trigger_on_low_confidence))
                or (high_danger and bool(self.config.mcts_trigger_on_high_danger))
                or (memory_forced_veto and bool(self.config.force_mcts_on_failure_signature))
            )
        )
        if mcts_trigger:
            mcts_alt = self._mcts_takeover(
                primary_agent=primary_agent,
                obs=obs,
                denied_action=p_action,
                risk=risk,
                blocked=blocked,
                low_conf=low_conf,
                high_danger=high_danger,
                effective_min_confidence=float(eff_conf),
                effective_danger_threshold=float(eff_danger),
                veto_adaptation=float(veto_adaptation),
                force_alternative=bool(memory_forced_veto),
                failure_signature_match=failure_signature_match,
            )
            if mcts_alt is not None:
                return mcts_alt

        # Neural Shield check
        if self.config.shield_enabled and self.config.shield_model_path:
            shield_danger = self._check_shield(obs, p_action)
            if shield_danger >= self.config.shield_threshold:
                # Veto based on neural shield
                alt = self._choose_safe_alternative(primary_agent=primary_agent, obs=obs, denied_action=p_action)
                if alt:
                    info = dict(alt.info or {})
                    info["safe_executor"] = {
                        "mode": "shield_reflex",
                        "denied_action": int(p_action),
                        "shield_danger": float(shield_danger),
                    }
                    return ActionResult(action=alt.action, info=info)

        if blocked or low_conf or high_danger:
            planner_failed = False
            danger_map_match = self._get_danger_map_match(obs)
            planner_allowed = bool(
                self._planner_active
                and (
                    planner_low_conf
                    or (blocked and self.config.planner_trigger_on_block)
                    or (high_danger and self.config.planner_trigger_on_high_danger)
                )
            )
            if planner_allowed:
                dyn_threshold = self._monitor.adaptive_planner_threshold(
                    base_threshold=float(self.config.planner_confidence_threshold),
                    regret_adaptation=float(self.config.planning_regret_adaptation),
                )
                planner_allowed = bool(
                    self._planning_budget.can_invoke(
                        verse_name=str(getattr(getattr(self.verse, "spec", None), "verse_name", "unknown")),
                        confidence=float(risk["confidence"]),
                        base_threshold=float(dyn_threshold),
                    )
                )
            if planner_allowed:
                planned = self._planner_takeover(obs=obs, denied_action=p_action)
                if planned is not None:
                    self._planning_budget.consume()
                    return planned
                planner_failed = True
            alt = self._choose_safe_alternative(primary_agent=primary_agent, obs=obs, denied_action=p_action)
            if alt is None and memory_forced_veto and getattr(getattr(self.verse, "action_space", None), "type", "") == "discrete":
                n = _safe_int(getattr(getattr(self.verse, "action_space", None), "n", 0), 0)
                for cand in range(max(0, n)):
                    if cand == int(p_action):
                        continue
                    if self._is_action_blocked(obs, int(cand)):
                        continue
                    alt = ActionResult(action=int(cand), info={"safe_executor_alt": "memory_forced_fallback"})
                    break
            if alt is not None:
                self._episode_counters["shield_vetoes"] = int(self._episode_counters["shield_vetoes"]) + 1
                explanation = self._generate_veto_explanation(
                    low_conf=low_conf,
                    high_danger=high_danger,
                    blocked=blocked,
                    danger_map_match=danger_map_match,
                    failure_signature_match=failure_signature_match,
                )
                meta = dict(alt.info or {})
                meta["safe_executor"] = {
                    "mode": "shield_veto",
                    "denied_action": int(p_action),
                    "danger": float(risk["danger"]),
                    "confidence": float(risk["confidence"]),
                    "confidence_model_danger": _safe_float(risk.get("confidence_model_danger", -1.0), -1.0),
                    "confidence_model_blend_weight": _safe_float(risk.get("confidence_model_blend_weight", 0.0), 0.0),
                    "effective_min_confidence": float(eff_conf),
                    "effective_danger_threshold": float(eff_danger),
                    "veto_adaptation": float(veto_adaptation),
                    "blocked": bool(blocked),
                    "low_confidence": bool(low_conf),
                    "high_danger": bool(high_danger),
                    "planner_attempted": bool(planner_allowed),
                    "planner_failed": bool(planner_failed),
                    "explanation": explanation,
                    "danger_map_match": danger_map_match,
                    "declarative_failure_signature": bool(memory_forced_veto),
                    "failure_signature_match": failure_signature_match,
                }
                return ActionResult(action=alt.action, info=meta)

        info = dict(pres.info or {})
        info["safe_executor"] = {
            "mode": "primary",
            "shield_veto": False,
            "danger": float(risk["danger"]),
            "confidence": float(risk["confidence"]),
            "confidence_model_danger": _safe_float(risk.get("confidence_model_danger", -1.0), -1.0),
            "confidence_model_blend_weight": _safe_float(risk.get("confidence_model_blend_weight", 0.0), 0.0),
            "effective_min_confidence": float(eff_conf),
            "effective_danger_threshold": float(eff_danger),
            "veto_adaptation": float(veto_adaptation),
            "declarative_failure_signature": bool(memory_forced_veto),
            "failure_signature_match": failure_signature_match,
        }
        return ActionResult(action=pres.action, info=info)

    def post_step(
        self,
        *,
        obs: JSONValue,
        action_result: ActionResult,
        step_result: StepResult,
        step_idx: int,
        primary_agent: Any = None,
    ) -> StepResult:
        if not self.config.enabled:
            return step_result

        self._steps_observed = int(self._steps_observed) + 1
        info = dict(step_result.info or {})
        se_meta = dict(info.get("safe_executor") or {})
        se_meta.update(dict(action_result.info or {}).get("safe_executor") or {})

        safety_violation = _is_safety_violation(info)
        severe_penalty = float(step_result.reward) <= float(self.config.severe_reward_threshold)
        danger_label, danger_label_source = _danger_label_from_info(info)
        dangerous_outcome = (
            bool(danger_label)
            if danger_label is not None
            else bool(safety_violation or severe_penalty)
        )
        action_meta = dict(action_result.info or {}).get("safe_executor") or {}
        action_conf = _safe_float(action_meta.get("confidence", 1.0), 1.0)
        action_danger = _safe_float(action_meta.get("danger", 0.0), 0.0)
        eff_conf = _safe_float(
            action_meta.get("effective_min_confidence", self.config.min_action_confidence),
            float(self.config.min_action_confidence),
        )
        eff_danger = _safe_float(
            action_meta.get("effective_danger_threshold", self.config.danger_threshold),
            float(self.config.danger_threshold),
        )
        low_confidence = bool(action_conf < float(eff_conf))
        high_danger = bool(action_danger >= float(eff_danger))
        planner_failed = bool(action_meta.get("planner_failed", False))
        policy_nan = _is_nan_action(action_result.action)
        self._monitor.observe(
            confidence=float(action_conf),
            danger=float(action_danger),
            dangerous_outcome=bool(dangerous_outcome),
            reward=float(step_result.reward),
        )
        monitor_status = self._monitor.status()
        self._planning_budget.record_outcome(
            verse_name=str(getattr(getattr(self.verse, "spec", None), "verse_name", "unknown")),
            reward=float(step_result.reward),
            failed=bool(dangerous_outcome),
        )

        if dangerous_outcome:
            self._episode_counters["dangerous_outcomes"] = int(self._episode_counters["dangerous_outcomes"]) + 1
            if self.config.block_repeated_fail_action:
                self._block_action(obs, _safe_int(action_result.action, -1))

        if monitor_status.in_incompetence_zone:
            self._fallback_remaining = max(int(self._fallback_remaining), int(self.config.fallback_horizon_steps))

        rewound = False
        if dangerous_outcome and self._can_rewind():
            rewound = self._rewind_to_checkpoint(agent=primary_agent)
            if rewound:
                self._rewinds_used += 1
                self._episode_counters["rewinds"] = int(self._episode_counters["rewinds"]) + 1

        if not dangerous_outcome and self._can_checkpoint() and (int(step_idx) % int(self.config.checkpoint_interval) == 0):
            self._save_checkpoint(obs=step_result.obs, agent=primary_agent)

        se_meta["dangerous_outcome"] = bool(dangerous_outcome)
        se_meta["danger_label"] = (
            None if danger_label is None else bool(danger_label)
        )
        se_meta["danger_label_source"] = str(danger_label_source)
        se_meta["safety_violation"] = bool(safety_violation)
        se_meta["severe_penalty"] = bool(severe_penalty)
        se_meta["low_confidence"] = bool(low_confidence)
        se_meta["high_danger"] = bool(high_danger)
        se_meta["effective_min_confidence"] = float(eff_conf)
        se_meta["effective_danger_threshold"] = float(eff_danger)
        se_meta["planner_failed"] = bool(planner_failed)
        se_meta["policy_nan"] = bool(policy_nan)
        se_meta["rewound"] = bool(rewound)
        se_meta["rewinds_used"] = int(self._rewinds_used)
        se_meta["fallback_remaining"] = int(self._fallback_remaining)
        se_meta["confidence_status"] = {
            "window_size": int(monitor_status.window_size),
            "competence_rate": float(monitor_status.competence_rate),
            "mean_confidence": float(monitor_status.mean_confidence),
            "mean_danger": float(monitor_status.mean_danger),
            "mean_regret": float(monitor_status.mean_regret),
            "failure_rate": float(monitor_status.failure_rate),
            "in_incompetence_zone": bool(monitor_status.in_incompetence_zone),
        }
        se_meta["planning_budget"] = self._planning_budget.snapshot(
            verse_name=str(getattr(getattr(self.verse, "spec", None), "verse_name", "unknown"))
        )
        mcts_queries = int(self._episode_counters.get("mcts_queries", 0))
        mcts_vetoes = int(self._episode_counters.get("mcts_vetoes", 0))
        se_meta["mcts_stats"] = {
            "enabled": bool(self._mcts_active and self._mcts is not None),
            "queries": mcts_queries,
            "vetoes": mcts_vetoes,
            "veto_rate": float(mcts_vetoes / float(max(1, mcts_queries))),
            "last_query": dict(self._last_mcts_result or {}),
        }
        se_meta["runtime_errors"] = {
            "total": int(sum(int(v) for v in self._runtime_error_counts.values())),
            "by_code": dict(self._runtime_error_counts),
            "recent": list(self._runtime_error_recent[-5:]),
        }
        se_meta["blocked_action_count"] = int(sum(len(v) for v in self._blocked_actions.values()))
        se_meta["counters"] = dict(self._episode_counters)
        failure_mode = _infer_failure_mode(
            info=info,
            dangerous_outcome=bool(dangerous_outcome),
            severe_penalty=bool(severe_penalty),
            safety_violation=bool(safety_violation),
            low_confidence=bool(low_confidence),
            high_danger=bool(high_danger),
            planner_failed=bool(planner_failed),
            policy_nan=bool(policy_nan),
            done=bool(step_result.done),
            truncated=bool(step_result.truncated),
        )
        failure_signals: List[str] = []
        if bool(severe_penalty):
            failure_signals.append("severe_penalty")
        if bool(safety_violation):
            failure_signals.append("safety_violation")
        if bool(low_confidence):
            failure_signals.append("low_confidence")
        if bool(high_danger):
            failure_signals.append("high_danger")
        if bool(action_meta.get("declarative_failure_signature", False)):
            failure_signals.append("declarative_failure_signature")
        if bool(planner_failed):
            failure_signals.append("planner_failed")
        if bool(policy_nan):
            failure_signals.append("policy_nan")
        if bool(rewound):
            failure_signals.append("rewound")
        if "warning" in info or "error" in info:
            failure_signals.append("env_warning")
        if int(sum(int(v) for v in self._runtime_error_counts.values())) > 0:
            failure_signals.append("runtime_error")
        se_meta["failure_mode"] = str(failure_mode)
        se_meta["failure_signals"] = failure_signals
        info["safe_executor"] = se_meta

        if rewound and self._checkpoint_obs is not None:
            return StepResult(
                obs=self._checkpoint_obs,
                reward=float(step_result.reward),
                done=False,
                truncated=False,
                info=info,
            )

        return StepResult(
            obs=step_result.obs,
            reward=float(step_result.reward),
            done=bool(step_result.done),
            truncated=bool(step_result.truncated),
            info=info,
        )

    def close(self) -> None:
        if self.fallback_agent is not None:
            try:
                self.fallback_agent.close()
            except Exception as exc:
                self._record_runtime_error(
                    code="fallback_close_error",
                    exc=exc,
                    context="safe_executor.fallback.close",
                )

    def _apply_mcts_overrides_for_verse(self, verse_name: str) -> None:
        merged: Dict[str, Any] = {}
        merged.update(_default_mcts_overrides(verse_name))
        if isinstance(self.config.mcts_verse_overrides, dict):
            merged.update(dict(self.config.mcts_verse_overrides.get(str(verse_name).strip().lower(), {}) or {}))
        if not merged:
            return
        # Apply only known scalar MCTS keys.
        if "mcts_enabled" in merged:
            self.config.mcts_enabled = bool(merged.get("mcts_enabled"))
        if "mcts_num_simulations" in merged:
            self.config.mcts_num_simulations = max(8, _safe_int(merged.get("mcts_num_simulations"), self.config.mcts_num_simulations))
        if "mcts_max_depth" in merged:
            self.config.mcts_max_depth = max(2, _safe_int(merged.get("mcts_max_depth"), self.config.mcts_max_depth))
        if "mcts_c_puct" in merged:
            self.config.mcts_c_puct = max(0.1, _safe_float(merged.get("mcts_c_puct"), self.config.mcts_c_puct))
        if "mcts_discount" in merged:
            self.config.mcts_discount = max(0.0, min(1.0, _safe_float(merged.get("mcts_discount"), self.config.mcts_discount)))
        if "mcts_loss_threshold" in merged:
            self.config.mcts_loss_threshold = max(-1.0, min(0.0, _safe_float(merged.get("mcts_loss_threshold"), self.config.mcts_loss_threshold)))
        if "mcts_min_visits" in merged:
            self.config.mcts_min_visits = max(1, _safe_int(merged.get("mcts_min_visits"), self.config.mcts_min_visits))
        if "mcts_value_confidence_threshold" in merged:
            self.config.mcts_value_confidence_threshold = max(
                0.0,
                min(1.0, _safe_float(merged.get("mcts_value_confidence_threshold"), self.config.mcts_value_confidence_threshold)),
            )

    def _effective_veto_thresholds(self) -> Tuple[float, float, float]:
        min_conf = float(self.config.min_action_confidence)
        danger_thr = float(self.config.danger_threshold)
        adaptation = 0.0
        if not bool(self.config.adaptive_veto_enabled):
            return min_conf, danger_thr, adaptation
        if int(self._steps_observed) < int(self.config.adaptive_veto_warmup_steps):
            return min_conf, danger_thr, adaptation
        st = self._monitor.status()
        if st.window_size <= 0:
            return min_conf, danger_thr, adaptation
        relax_strength = float(self.config.adaptive_veto_relaxation)
        if bool(self.config.adaptive_veto_schedule_enabled):
            sched_steps = max(1, int(self.config.adaptive_veto_schedule_steps))
            sched_power = max(0.10, float(self.config.adaptive_veto_schedule_power))
            progress_steps = max(0, int(self._steps_observed) - int(self.config.adaptive_veto_warmup_steps))
            progress = max(0.0, min(1.0, float(progress_steps) / float(sched_steps)))
            shaped = float(progress) ** float(sched_power)
            start = max(0.0, min(1.0, float(self.config.adaptive_veto_relaxation_start)))
            end = max(0.0, min(1.0, float(self.config.adaptive_veto_relaxation_end)))
            relax_strength = float(start + (end - start) * shaped)
        denom = max(1e-6, 1.0 - float(self.config.min_competence_rate))
        competence_gain = max(0.0, min(1.0, (float(st.competence_rate) - float(self.config.min_competence_rate)) / denom))
        failure_guard = 1.0 - max(
            0.0,
            min(1.0, float(st.failure_rate) / float(max(1e-6, self.config.adaptive_veto_failure_guard))),
        )
        adaptation = max(
            0.0,
            min(
                1.0,
                float(relax_strength) * float(competence_gain) * float(failure_guard),
            ),
        )
        eff_min_conf = max(0.01, min(1.0, float(min_conf) * (1.0 - float(adaptation))))
        eff_danger = max(
            0.0,
            min(0.999, float(danger_thr) + (1.0 - float(danger_thr)) * float(adaptation)),
        )
        return eff_min_conf, eff_danger, adaptation

    def _is_action_blocked(self, obs: JSONValue, action: int) -> bool:
        if action < 0:
            return False
        return action in self._blocked_actions.get(_obs_key(obs), set())

    def _block_action(self, obs: JSONValue, action: int) -> None:
        if action < 0:
            return
        k = _obs_key(obs)
        bucket = self._blocked_actions.setdefault(k, set())
        bucket.add(int(action))

    def _estimate_action_risk(self, agent: Any, obs: JSONValue, action: int) -> Dict[str, float]:
        out = {"danger": 0.0, "confidence": 1.0}
        if action < 0:
            return out
        try:
            if hasattr(agent, "action_diagnostics"):
                diag = agent.action_diagnostics(obs)  # type: ignore[attr-defined]
                dangers = diag.get("danger_scores") if isinstance(diag, dict) else None
                probs = diag.get("sample_probs") if isinstance(diag, dict) else None
                if isinstance(dangers, list) and 0 <= action < len(dangers):
                    out["danger"] = max(0.0, min(1.0, _safe_float(dangers[action], 0.0)))
                if isinstance(probs, list) and 0 <= action < len(probs):
                    out["confidence"] = max(0.0, min(1.0, _safe_float(probs[action], 1.0)))
        except Exception as exc:
            self._record_runtime_error(
                code="risk_diag_error",
                exc=exc,
                context="safe_executor.risk.action_diagnostics",
            )
        model_danger = self._predict_confidence_model_danger(obs=obs, action=int(action))
        if model_danger is not None:
            w = max(0.0, min(1.0, float(self.config.confidence_model_weight)))
            diag_danger = max(0.0, min(1.0, _safe_float(out.get("danger", 0.0), 0.0)))
            diag_conf = max(0.0, min(1.0, _safe_float(out.get("confidence", 1.0), 1.0)))
            fused_danger = ((1.0 - w) * float(diag_danger)) + (w * float(model_danger))
            fused_conf = ((1.0 - w) * float(diag_conf)) + (w * (1.0 - float(model_danger)))
            out["danger"] = max(0.0, min(1.0, float(fused_danger)))
            out["confidence"] = max(0.0, min(1.0, float(fused_conf)))
            out["confidence_model_danger"] = max(0.0, min(1.0, float(model_danger)))
            out["confidence_model_blend_weight"] = float(w)
        return out

    def _choose_safe_alternative(self, *, primary_agent: Any, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
        return choose_safe_alternative(self, primary_agent=primary_agent, obs=obs, denied_action=denied_action)

    def _can_checkpoint(self) -> bool:
        return can_checkpoint(self)

    def _can_rewind(self) -> bool:
        return can_rewind(self)

    def _save_checkpoint(self, obs: JSONValue, agent: Optional[Any] = None) -> None:
        save_checkpoint(self, obs, agent)

    def _rewind_to_checkpoint(self, agent: Optional[Any] = None) -> bool:
        return rewind_to_checkpoint(self, agent)

    def _check_shield(self, obs: JSONValue, action: int) -> float:
        return check_shield(self, obs, action)

    def _mcts_takeover(
        self,
        *,
        primary_agent: Any,
        obs: JSONValue,
        denied_action: int,
        risk: Dict[str, float],
        blocked: bool,
        low_conf: bool,
        high_danger: bool,
        effective_min_confidence: float,
        effective_danger_threshold: float,
        veto_adaptation: float,
        force_alternative: bool = False,
        failure_signature_match: Optional[Dict[str, Any]] = None,
    ) -> Optional[ActionResult]:
        return mcts_takeover(
            self,
            primary_agent=primary_agent,
            obs=obs,
            denied_action=denied_action,
            risk=risk,
            blocked=blocked,
            low_conf=low_conf,
            high_danger=high_danger,
            effective_min_confidence=effective_min_confidence,
            effective_danger_threshold=effective_danger_threshold,
            veto_adaptation=veto_adaptation,
            force_alternative=force_alternative,
            failure_signature_match=failure_signature_match,
        )

    def _planner_takeover(self, *, obs: JSONValue, denied_action: int) -> Optional[ActionResult]:
        return planner_takeover(self, obs=obs, denied_action=denied_action)
