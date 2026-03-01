import re

with open('agents/transformer_agent.py', 'r') as f:
    content = f.read()

header = '''
def _as_set(raw: Any) -> Optional[set]:
    if raw is None:
        return None
    if isinstance(raw, (set, list, tuple)):
        out = set(str(x).strip().lower() for x in raw if str(x).strip())
        return out if out else None
    s = str(raw).strip().lower()
    if not s:
        return None
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return set(parts) if parts else None
'''

content = content.replace('def _load_torch_payload', header + '\n\ndef _load_torch_payload')

init_addition = '''
        self._verse_name = str(cfg.get("verse_name", "")).strip().lower()
        self._recall_enabled = bool(cfg.get("recall_enabled", False))
        self._recall_top_k = max(1, _safe_int(cfg.get("recall_top_k", 5), 5))
        self._recall_min_score = _safe_float(cfg.get("recall_min_score", -0.2), -0.2)
        self._recall_same_verse_only = bool(cfg.get("recall_same_verse_only", True))
        self._recall_memory_types = _as_set(cfg.get("recall_memory_types"))
        self._recall_vote_weight = max(0.0, min(3.0, _safe_float(cfg.get("recall_vote_weight", 0.75), 0.75)))
        self._recall_use_source_greedy_action = bool(cfg.get("recall_use_source_greedy_action", False))
        self._recall_risk_key = str(cfg.get("recall_risk_key", "risk")).strip() or "risk"
        self._recall_risk_threshold = _safe_float(cfg.get("recall_risk_threshold", 6.0), 6.0)
        self._recall_uncertainty_margin = max(0.0, _safe_float(cfg.get("recall_uncertainty_margin", 0.10), 0.10))
        self._recall_cooldown_steps = max(1, _safe_int(cfg.get("recall_cooldown_steps", 2), 2))

        self._last_query_step = -10**9
        self._last_bundle: Optional[Dict[str, Any]] = None
        self._recall_uses = 0
'''

content = content.replace('        self._online_updates = 0', '        self._online_updates = 0' + init_addition)

methods_addition = '''
    def memory_query_request(self, *, obs: JSONValue, step_idx: int) -> Optional[Dict[str, Any]]:
        if not bool(self._recall_enabled):
            return None
        step = int(step_idx)
        if step < int(self._last_query_step):
            # New episode: rollout step_idx resets to 0.
            self._last_query_step = -10**9
        if (step - int(self._last_query_step)) < int(self._recall_cooldown_steps):
            return None

        risk_value = None
        if isinstance(obs, dict) and self._recall_risk_key in obs:
            risk_value = _safe_float(obs.get(self._recall_risk_key), None)  # type: ignore[arg-type]
        trigger_risk = bool(risk_value is not None and float(risk_value) >= float(self._recall_risk_threshold))

        # Skip uncertainty trigger to prevent unnecessary forward passes in memory query.
        trigger_uncertain = False
        
        if not (trigger_risk or trigger_uncertain):
            return None

        reason = "high_risk" if trigger_risk else "uncertain_state"
        req = {
            "query_obs": obs,
            "top_k": int(self._recall_top_k),
            "min_score": float(self._recall_min_score),
            "verse_name": (self._verse_name if bool(self._recall_same_verse_only and self._verse_name) else None),
            "memory_types": (sorted(list(self._recall_memory_types)) if self._recall_memory_types else None),
            "reason": str(reason),
        }
        self._last_query_step = int(step)
        return req

    def on_memory_response(self, payload: Dict[str, Any]) -> None:
        self._last_bundle = payload if isinstance(payload, dict) else None

    def _extract_recall_bundle(self, *, hint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if isinstance(hint, dict):
            raw = hint.get("memory_recall")
            if isinstance(raw, dict):
                return raw
        if isinstance(self._last_bundle, dict):
            return self._last_bundle
        return None

    def _memory_action_prior(self, recall: Optional[Dict[str, Any]]) -> Optional[List[float]]:
        if not isinstance(recall, dict):
            return None
        matches = recall.get("matches")
        if not isinstance(matches, list) or not matches:
            return None
        prior = [0.0] * self._n_actions
        for row in matches:
            if not isinstance(row, dict):
                continue
            a = -1
            if bool(self._recall_use_source_greedy_action):
                a = _safe_int(row.get("source_greedy_action"), -1)
            if a < 0:
                a = _safe_int(row.get("action"), -1)
            if a < 0 or a >= int(self._n_actions):
                continue
            score = max(0.0, _safe_float(row.get("score"), 0.0))
            prior[a] += float(score)
        mx = max(prior)
        if mx <= 0.0:
            return None
        return [p / mx for p in prior]

    def act_with_hint(self, obs: JSONValue, hint: Optional[Dict[str, Any]]) -> ActionResult:
        self._maybe_reset_on_obs(obs)
        state_vec = obs_to_universal_vector(obs, dim=self._state_dim)
        model_in = self._build_inputs(state_vec)
        with torch.no_grad():
            action_t, conf_t, probs_t = self.model.predict_next_action(
                states=model_in["states"],
                returns_to_go=model_in["returns_to_go"],
                prev_actions=model_in["prev_actions"],
                timesteps=model_in["timesteps"],
                attention_mask=model_in["attention_mask"],
                temperature=float(self._temperature),
                top_k=int(self._top_k),
                sample=bool(self._sample),
            )
        
        probs_base = probs_t.squeeze(0).cpu().numpy().astype(float)
        probs_recall = probs_base.copy()
        
        recall = self._extract_recall_bundle(hint=hint)
        recall_prior = self._memory_action_prior(recall)
        recall_eligible = bool(recall_prior is not None and max(recall_prior) > 0.0)
        
        if recall_eligible and recall_prior is not None:
            for i in range(self._n_actions):
                probs_recall[i] += float(self._recall_vote_weight) * float(recall_prior[i])
                
        import numpy as np
        if self._sample:
            p_sum = float(np.sum(probs_recall))
            if p_sum > 0:
                probs_recall = probs_recall / p_sum
            else:
                probs_recall = probs_base
            action = int(np.random.choice(self._n_actions, p=probs_recall))
        else:
            action = int(np.argmax(probs_recall))

        action = max(0, min(self._n_actions - 1, action))

        self._state_history.append(list(state_vec))
        self._action_history.append(int(action))
        self._step_count += 1
        
        info = {
            "mode": "adt",
            "confidence": float(conf_t.item()),
            "target_return": float(self._target_return),
            "context_len": int(self._context_len),
            "memory_recall_eligible": bool(recall_eligible),
            "memory_recall_used": bool(recall_eligible),  # Simplification
        }
        if recall_eligible:
            self._recall_uses += 1
            info["memory_recall_uses"] = int(self._recall_uses)

        return ActionResult(
            action=int(action),
            info=info,
        )
'''

content = content.replace('    def act(self, obs: JSONValue) -> ActionResult:', methods_addition + '\n    def act(self, obs: JSONValue) -> ActionResult:\n        return self.act_with_hint(obs, None)\n\n    def _old_act(self, obs: JSONValue) -> ActionResult:')

with open('agents/transformer_agent.py', 'w') as f:
    f.write(content)
