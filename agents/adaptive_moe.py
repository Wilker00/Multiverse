"""
agents/adaptive_moe.py

Adaptive MoE routing with uncertainty-aware centroid fallback.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from agents.distilled_agent import DistilledAgent
from agents.imitation_agent import ImitationLookupAgent, obs_key
from agents.special_moe_agent import SpecialMoEAgent, _safe_float, _safe_int
from core.agent_base import ActionResult
from core.types import AgentSpec


class AdaptiveMoEAgent(SpecialMoEAgent):
    """
    Extends SpecialMoEAgent with disagreement-aware routing.

    Config keys:
    - uncertainty_threshold: float in [0,1], default 0.35
    - centroid_model_path: optional distilled model dir for fallback
    - centroid_dataset_path: optional imitation dataset fallback
    """

    def __init__(self, spec: AgentSpec, observation_space, action_space):
        super().__init__(spec=spec, observation_space=observation_space, action_space=action_space)
        cfg = spec.config if isinstance(spec.config, dict) else {}
        self.uncertainty_threshold = max(0.0, min(1.0, _safe_float(cfg.get("uncertainty_threshold", 0.35), 0.35)))
        self._centroid_agent = None
        self._centroid_source = ""
        self._load_centroid(cfg)

    def _load_centroid(self, cfg: Dict[str, Any]) -> None:
        model_path = str(cfg.get("centroid_model_path", "") or "").strip()
        dataset_path = str(cfg.get("centroid_dataset_path", "") or "").strip()

        if model_path:
            if not os.path.isfile(os.path.join(model_path, "distilled_policy.json")):
                raise FileNotFoundError(f"centroid_model_path missing distilled_policy.json: {model_path}")
            centroid_spec = AgentSpec(
                spec_version=self.spec.spec_version,
                policy_id=f"{self.spec.policy_id}:centroid",
                policy_version=self.spec.policy_version,
                algo="distilled",
                framework=self.spec.framework,
                seed=self.spec.seed,
                tags=list(self.spec.tags),
                config={"model_path": model_path},
            )
            self._centroid_agent = DistilledAgent(
                spec=centroid_spec,
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
            self._centroid_source = "distilled"
            return

        if dataset_path:
            if not os.path.isfile(dataset_path):
                raise FileNotFoundError(f"centroid_dataset_path not found: {dataset_path}")
            centroid_spec = AgentSpec(
                spec_version=self.spec.spec_version,
                policy_id=f"{self.spec.policy_id}:centroid",
                policy_version=self.spec.policy_version,
                algo="imitation_lookup",
                framework=self.spec.framework,
                seed=self.spec.seed,
                tags=list(self.spec.tags),
                config={},
            )
            ag = ImitationLookupAgent(
                spec=centroid_spec,
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
            ag.learn_from_dataset(dataset_path)
            self._centroid_agent = ag
            self._centroid_source = "dataset"

    def seed(self, seed: Optional[int]) -> None:
        super().seed(seed)
        if self._centroid_agent is not None:
            self._centroid_agent.seed(seed)

    def close(self) -> None:
        super().close()
        if self._centroid_agent is not None:
            self._centroid_agent.close()

    def act(self, obs):
        if obs_key(obs) in self._bad_obs:
            return ActionResult(action=self._sample_random_action(), info={"mode": "boundary_avoid"})

        if not self._experts:
            return ActionResult(action=self._sample_random_action(), info={"mode": "no_experts"})

        ranked = self._rank_experts(obs)
        selected = ranked[: min(self.top_k, len(ranked))]
        if not selected:
            return ActionResult(action=self._sample_random_action(), info={"mode": "no_ranked_experts"})

        actions: List[Tuple[float, Any, str]] = []
        for weight, exp in selected:
            action = exp.agent.act(obs).action
            actions.append((float(weight), action, exp.skill_id))
        selected_expert = str(actions[0][2]) if actions else ""
        selector_conf = float(actions[0][0]) if actions else 0.0

        uncertainty = self._estimate_disagreement(actions)
        if self._centroid_agent is not None and uncertainty > float(self.uncertainty_threshold):
            c_action = self._centroid_agent.act(obs).action
            return ActionResult(
                action=c_action,
                info={
                    "mode": "adaptive_moe_centroid",
                    "experts": [sid for _, _, sid in actions],
                    "weights": [float(w) for w, _, _ in actions],
                    "uncertainty": float(uncertainty),
                    "threshold": float(self.uncertainty_threshold),
                    "centroid_source": self._centroid_source,
                    "selected_expert": selected_expert,
                    "selector_confidence": float(selector_conf),
                },
            )

        blended = self._blend_actions(actions)
        return ActionResult(
            action=blended,
            info={
                "mode": "adaptive_moe",
                "experts": [sid for _, _, sid in actions],
                "weights": [float(w) for w, _, _ in actions],
                "uncertainty": float(uncertainty),
                "threshold": float(self.uncertainty_threshold),
                "selector_active": bool(self._selector is not None),
                "selected_expert": selected_expert,
                "selector_confidence": float(selector_conf),
            },
        )

    def _estimate_disagreement(self, actions: List[Tuple[float, Any, str]]) -> float:
        if not actions:
            return 0.0
        if self.action_space.type == "discrete":
            counts: Dict[int, float] = {}
            total = 0.0
            for w, action, _ in actions:
                a = _safe_int(action, 0)
                ww = max(0.0, float(w))
                counts[a] = counts.get(a, 0.0) + ww
                total += ww
            if total <= 0.0:
                return 1.0
            max_mass = max(counts.values()) if counts else 0.0
            return float(max(0.0, 1.0 - (max_mass / total)))

        if self.action_space.type == "continuous":
            vecs: List[List[float]] = []
            weights: List[float] = []
            for w, action, _ in actions:
                if not isinstance(action, list):
                    continue
                vec = [float(v) for v in action if isinstance(v, (int, float))]
                if not vec:
                    continue
                vecs.append(vec)
                weights.append(max(0.0, float(w)))
            if not vecs:
                return 1.0
            dim = min(len(v) for v in vecs)
            if dim <= 0:
                return 1.0
            # Weighted mean variance across dimensions, squashed to [0,1).
            total_w = sum(weights) if sum(weights) > 0 else float(len(weights))
            var_sum = 0.0
            for d in range(dim):
                mean_d = sum(weights[i] * vecs[i][d] for i in range(len(vecs))) / total_w
                var_d = sum(weights[i] * ((vecs[i][d] - mean_d) ** 2) for i in range(len(vecs))) / total_w
                var_sum += var_d
            mean_var = var_sum / float(dim)
            return float(mean_var / (1.0 + mean_var))

        # Unsupported spaces: treat differing predictions as high disagreement.
        first = str(actions[0][1])
        same = all(str(a) == first for _, a, _ in actions[1:])
        return 0.0 if same else 1.0
