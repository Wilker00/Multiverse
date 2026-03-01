import os

with open("agents/transformer_agent.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update memory_query_request
from_str = """        req = {
            "query_obs": obs,
            "top_k": int(self._recall_top_k),
            "min_score": float(self._recall_min_score),
            "verse_name": (self._verse_name if bool(self._recall_same_verse_only and self._verse_name) else None),
            "memory_types": (sorted(list(self._recall_memory_types)) if self._recall_memory_types else None),
            "reason": str(reason),
        }"""

to_str = """        req = {
            "query_obs": obs,
            "top_k": int(self._recall_top_k),
            "min_score": float(self._recall_min_score),
            "verse_name": (self._verse_name if bool(self._recall_same_verse_only and self._verse_name) else None),
            "memory_types": (sorted(list(self._recall_memory_types)) if self._recall_memory_types else None),
            "reason": str(reason),
            "trajectory_window": int(self._context_len // 2) if self._context_len > 4 else 0,
        }"""
content = content.replace(from_str, to_str)

# 2. Update _build_inputs signature & logic
from_build = """    def _build_inputs(self, current_state: List[float]) -> Dict[str, torch.Tensor]:
        hist_states = list(self._state_history)
        hist_actions = list(self._action_history)
        states_seq = hist_states + [list(current_state)]

        if len(states_seq) > self._context_len:
            overflow = len(states_seq) - self._context_len
            states_seq = states_seq[overflow:]
            hist_actions = hist_actions[overflow:]

        seq_len = len(states_seq)
        if self._action_space_type == "continuous":
            prev_actions_seq = [[0.0] * self._n_actions for _ in range(seq_len)]
            for i in range(1, seq_len):
                hist_idx = i - 1
                if hist_idx < len(hist_actions):
                    prev_actions_seq[i] = list(hist_actions[hist_idx])
        else:
            prev_actions_seq = [int(self._bos_token_id) for _ in range(seq_len)]
            for i in range(1, seq_len):
                hist_idx = i - 1
                if hist_idx < len(hist_actions):
                    prev_actions_seq[i] = int(hist_actions[hist_idx])

        start_t = max(0, int(self._step_count) - seq_len + 1)
        timesteps_seq = [start_t + i for i in range(seq_len)]
        rtg_seq = [float(self._target_return) for _ in range(seq_len)]"""

to_build = """    def _build_inputs(self, current_state: List[float], trajectory: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        hist_states = list(self._state_history)
        hist_actions = list(self._action_history)
        
        my_states = hist_states + [list(current_state)]
        my_actions = hist_actions
        
        start_t = max(0, int(self._step_count) - len(my_states) + 1)
        my_timesteps = [start_t + i for i in range(len(my_states))]
        my_rtg = [float(self._target_return) for _ in range(len(my_states))]
        
        demo_states = []
        demo_actions = []
        demo_timesteps = []
        demo_rtg = []
        
        if trajectory:
            for i, st in enumerate(trajectory):
                # parse obs to univ vec or use as is
                raw_obs = st.get("obs")
                from memory.embeddings import obs_to_universal_vector
                try:
                    vec = obs_to_universal_vector(raw_obs, dim=int(self._state_dim))
                    demo_states.append(list(vec))
                except Exception:
                    demo_states.append([0.0]*int(self._state_dim))
                    
                act = st.get("action")
                if self._action_space_type == "continuous":
                    try:
                        act_vec = [float(x) for x in list(act)]
                    except Exception:
                        act_vec = [0.0]*int(self._n_actions)
                    demo_actions.append(act_vec)
                else:
                    try:
                        act_val = int(act)
                    except Exception:
                        act_val = int(self._bos_token_id)
                    demo_actions.append(act_val)
                demo_timesteps.append(int(st.get("step_idx", 0)))
                # We could estimate RTG from rewards, but let's just copy current RTG for demo
                demo_rtg.append(float(self._target_return))

        states_seq = demo_states + my_states
        action_seq = demo_actions + my_actions
        timesteps_seq = demo_timesteps + my_timesteps
        rtg_seq = demo_rtg + my_rtg
        
        if len(states_seq) > self._context_len:
            overflow = len(states_seq) - self._context_len
            states_seq = states_seq[overflow:]
            action_seq = action_seq[overflow:]
            timesteps_seq = timesteps_seq[overflow:]
            rtg_seq = rtg_seq[overflow:]
            
        seq_len = len(states_seq)
        
        if self._action_space_type == "continuous":
            prev_actions_seq = [[0.0] * self._n_actions for _ in range(seq_len)]
            for i in range(1, seq_len):
                if (i-1) < len(action_seq):
                    prev_actions_seq[i] = list(action_seq[i-1])
        else:
            prev_actions_seq = [int(self._bos_token_id) for _ in range(seq_len)]
            for i in range(1, seq_len):
                if (i-1) < len(action_seq):
                    prev_actions_seq[i] = int(action_seq[i-1])"""

content = content.replace(from_build, to_build)

# 3. Update act_with_hint to extract trajectory and pass to _build_inputs
from_act = """    def act_with_hint(self, obs: JSONValue, hint: Optional[Dict[str, Any]]) -> ActionResult:
        self._maybe_reset_on_obs(obs)
        state_vec = obs_to_universal_vector(obs, dim=self._state_dim)
        model_in = self._build_inputs(state_vec)"""

to_act = """    def act_with_hint(self, obs: JSONValue, hint: Optional[Dict[str, Any]]) -> ActionResult:
        self._maybe_reset_on_obs(obs)
        state_vec = obs_to_universal_vector(obs, dim=self._state_dim)
        
        import numpy as np
        recall = self._extract_recall_bundle(hint=hint)
        trajectory = None
        if isinstance(recall, dict):
            matches = recall.get("matches")
            if isinstance(matches, list) and len(matches) > 0:
                top_match = matches[0]
                if isinstance(top_match, dict) and top_match.get("trajectory"):
                    trajectory = top_match.get("trajectory")

        model_in = self._build_inputs(state_vec, trajectory=trajectory)"""

content = content.replace(from_act, to_act)

with open("agents/transformer_agent.py", "w", encoding="utf-8") as f:
    f.write(content)

print("transformer_agent.py patch ok")
