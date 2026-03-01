import os
import re

# 1. Update DecisionTransformer
with open("models/decision_transformer.py", "r") as f:
    dt_content = f.read()

# Add action_space_type to config
if "action_space_type: str" not in dt_content:
    dt_content = dt_content.replace(
        "action_dim: int\n    context_len: int = 20",
        "action_dim: int\n    action_space_type: str = \"discrete\"\n    context_len: int = 20"
    )

# Update embed
embed_old = """        self.return_embed = nn.Linear(1, d_model)
        self.action_embed = nn.Embedding(int(self.config.action_dim) + 1, d_model)
        self.time_embed = nn.Embedding(int(self.config.max_timestep), d_model)"""
embed_new = """        self.return_embed = nn.Linear(1, d_model)
        if self.config.action_space_type == "continuous":
            self.action_embed = nn.Linear(int(self.config.action_dim), d_model)
        else:
            self.action_embed = nn.Embedding(int(self.config.action_dim) + 1, d_model)
        self.time_embed = nn.Embedding(int(self.config.max_timestep), d_model)"""
if "self.action_embed = nn.Embedding(int(self.config.action_dim) + 1, d_model)" in dt_content:
    dt_content = dt_content.replace(embed_old, embed_new)

# Update forward input shape check
fwd_checks_old = """        if returns_to_go.dim() != 2:
            raise ValueError(f"returns_to_go must be rank-2 [B,T], got {tuple(returns_to_go.shape)}")
        if prev_actions.dim() != 2:
            raise ValueError(f"prev_actions must be rank-2 [B,T], got {tuple(prev_actions.shape)}")
        if timesteps.dim() != 2:"""
fwd_checks_new = """        if returns_to_go.dim() != 2:
            raise ValueError(f"returns_to_go must be rank-2 [B,T], got {tuple(returns_to_go.shape)}")
        if self.config.action_space_type == "discrete" and prev_actions.dim() != 2:
            raise ValueError(f"prev_actions must be rank-2 [B,T], got {tuple(prev_actions.shape)}")
        if self.config.action_space_type == "continuous" and prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must be rank-3 [B,T,A], got {tuple(prev_actions.shape)}")
        if timesteps.dim() != 2:"""
if "if prev_actions.dim() != 2" in dt_content:
    dt_content = dt_content.replace(fwd_checks_old, fwd_checks_new)

# Update forward mismatch check
fwd_mismatch_old = """        if returns_to_go.shape != (batch_size, seq_len):
            raise ValueError("returns_to_go shape mismatch against states")
        if prev_actions.shape != (batch_size, seq_len):
            raise ValueError("prev_actions shape mismatch against states")
        if timesteps.shape != (batch_size, seq_len):"""
fwd_mismatch_new = """        if returns_to_go.shape != (batch_size, seq_len):
            raise ValueError("returns_to_go shape mismatch against states")
        if self.config.action_space_type == "discrete" and prev_actions.shape != (batch_size, seq_len):
            raise ValueError("prev_actions shape mismatch against states")
        if self.config.action_space_type == "continuous" and prev_actions.shape != (batch_size, seq_len, int(self.config.action_dim)):
            raise ValueError("prev_actions shape mismatch against states")
        if timesteps.shape != (batch_size, seq_len):"""
if "if prev_actions.shape != (batch_size, seq_len):" in dt_content:
    dt_content = dt_content.replace(fwd_mismatch_old, fwd_mismatch_new)

# Update forward embedding
embed_app_old = """        prev_actions = prev_actions.long().clamp(0, int(self.config.action_dim))
        timesteps = timesteps.long().clamp(0, int(self.config.max_timestep) - 1)

        x_state = self.state_embed(states)
        x_rtg = self.return_embed(returns_to_go.unsqueeze(-1))
        x_prev = self.action_embed(prev_actions)
        x_time = self.time_embed(timesteps)"""
embed_app_new = """        timesteps = timesteps.long().clamp(0, int(self.config.max_timestep) - 1)

        x_state = self.state_embed(states)
        x_rtg = self.return_embed(returns_to_go.unsqueeze(-1))
        if self.config.action_space_type == "continuous":
            x_prev = self.action_embed(prev_actions.float())
        else:
            prev_actions = prev_actions.long().clamp(0, int(self.config.action_dim))
            x_prev = self.action_embed(prev_actions)
        x_time = self.time_embed(timesteps)"""
if "x_prev = self.action_embed(prev_actions)" in dt_content:
    dt_content = dt_content.replace(embed_app_old, embed_app_new)

# Update predict_next_action
predict_old = """        probs = torch.softmax(last_logits, dim=-1)
        if bool(sample):
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(probs, dim=-1)
        conf = probs.gather(1, actions.view(-1, 1)).squeeze(1)
        return actions, conf, probs"""
predict_new = """        if self.config.action_space_type == "continuous":
            actions = last_logits
            if bool(sample):
                actions = actions + torch.randn_like(actions) * 0.1
            conf = torch.ones_like(actions[:, 0])
            probs = actions
            return actions, conf, probs
        else:
            probs = torch.softmax(last_logits, dim=-1)
            if bool(sample):
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                actions = torch.argmax(probs, dim=-1)
            conf = probs.gather(1, actions.view(-1, 1)).squeeze(1)
            return actions, conf, probs"""
if "= torch.argmax(probs, dim=-1)\n        conf =" in dt_content:
    dt_content = dt_content.replace(predict_old, predict_new)

with open("models/decision_transformer.py", "w") as f:
    f.write(dt_content)

print("decision_transformer.py patched!")

# 2. Update TransformerAgent
with open("agents/transformer_agent.py", "r") as f:
    ta_content = f.read()

# Make history hold Any (float or int)
if "self._action_history: Deque[int] = deque" in ta_content:
    ta_content = ta_content.replace(
        "self._action_history: Deque[int] = deque",
        "self._action_history: Deque[Any] = deque"
    )

init_checks_old = """        if action_space.type != "discrete" or not isinstance(action_space.n, int) or int(action_space.n) <= 0:
            raise ValueError("TransformerAgent requires discrete action_space with n > 0")
        self._n_actions = int(action_space.n)"""
init_checks_new = """        self._action_space_type = "discrete"
        if action_space.type in ("vector", "continuous", "box"):
            self._action_space_type = "continuous"
            if not getattr(action_space, "shape", None):
                raise ValueError("Continuous action_space requires a shape attribute")
            self._n_actions = int(action_space.shape[0])
        else:
            if action_space.type != "discrete" or not isinstance(action_space.n, int) or int(action_space.n) <= 0:
                raise ValueError("TransformerAgent requires discrete/vector action_space with valid dimension")
            self._n_actions = int(action_space.n)"""
if "requires discrete action_space with n > 0" in ta_content:
    ta_content = ta_content.replace(init_checks_old, init_checks_new)

build_inputs_old = """        prev_actions_seq = [int(self._bos_token_id) for _ in range(seq_len)]
        for i in range(1, seq_len):
            hist_idx = i - 1
            if hist_idx < len(hist_actions):
                prev_actions_seq[i] = int(hist_actions[hist_idx])

        start_t = max(0, int(self._step_count) - seq_len + 1)
        timesteps_seq = [start_t + i for i in range(seq_len)]
        rtg_seq = [float(self._target_return) for _ in range(seq_len)]

        K = self._context_len
        states_pad = [[0.0] * self._state_dim for _ in range(K)]
        prev_pad = [int(self._bos_token_id) for _ in range(K)]
        rtg_pad = [0.0 for _ in range(K)]
        t_pad = [0 for _ in range(K)]
        m_pad = [0.0 for _ in range(K)]
        for i in range(seq_len):
            states_pad[i] = list(states_seq[i])
            prev_pad[i] = int(prev_actions_seq[i])
            rtg_pad[i] = float(rtg_seq[i])
            t_pad[i] = int(timesteps_seq[i])
            m_pad[i] = 1.0

        return {
            "states": torch.tensor([states_pad], dtype=torch.float32, device=self._device),
            "returns_to_go": torch.tensor([rtg_pad], dtype=torch.float32, device=self._device),
            "prev_actions": torch.tensor([prev_pad], dtype=torch.long, device=self._device),
            "timesteps": torch.tensor([t_pad], dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor([m_pad], dtype=torch.float32, device=self._device),
            "seq_len": torch.tensor([seq_len], dtype=torch.long, device=self._device),
        }"""
build_inputs_new = """        if self._action_space_type == "continuous":
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
        rtg_seq = [float(self._target_return) for _ in range(seq_len)]

        K = self._context_len
        states_pad = [[0.0] * self._state_dim for _ in range(K)]
        rtg_pad = [0.0 for _ in range(K)]
        t_pad = [0 for _ in range(K)]
        m_pad = [0.0 for _ in range(K)]
        
        if self._action_space_type == "continuous":
            prev_pad = [[0.0] * self._n_actions for _ in range(K)]
            for i in range(seq_len):
                prev_pad[i] = list(prev_actions_seq[i])
            prev_tensor = torch.tensor([prev_pad], dtype=torch.float32, device=self._device)
        else:
            prev_pad = [int(self._bos_token_id) for _ in range(K)]
            for i in range(seq_len):
                prev_pad[i] = int(prev_actions_seq[i])
            prev_tensor = torch.tensor([prev_pad], dtype=torch.long, device=self._device)
            
        for i in range(seq_len):
            states_pad[i] = list(states_seq[i])
            rtg_pad[i] = float(rtg_seq[i])
            t_pad[i] = int(timesteps_seq[i])
            m_pad[i] = 1.0

        return {
            "states": torch.tensor([states_pad], dtype=torch.float32, device=self._device),
            "returns_to_go": torch.tensor([rtg_pad], dtype=torch.float32, device=self._device),
            "prev_actions": prev_tensor,
            "timesteps": torch.tensor([t_pad], dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor([m_pad], dtype=torch.float32, device=self._device),
            "seq_len": torch.tensor([seq_len], dtype=torch.long, device=self._device),
        }"""
if "prev_actions_seq = [int(self._bos_token_id) for _ in range(seq_len)]" in ta_content:
    ta_content = ta_content.replace(build_inputs_old, build_inputs_new)

act_old = """        probs_base = probs_t.squeeze(0).cpu().numpy().astype(float)
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
        self._action_history.append(int(action))"""
act_new = """        import numpy as np
        recall = self._extract_recall_bundle(hint=hint)
        recall_prior = self._memory_action_prior(recall)
        recall_eligible = bool(recall_prior is not None and max(recall_prior) > 0.0)
        
        if self._action_space_type == "continuous":
            action_out = action_t.squeeze(0).cpu().numpy().astype(float).tolist()
            # For continuous, we skip simple memory prior addition for now or blend later.
            action_ret = action_out
            self._action_history.append(list(action_ret))
        else:
            probs_base = probs_t.squeeze(0).cpu().numpy().astype(float)
            probs_recall = probs_base.copy()
            if recall_eligible and recall_prior is not None:
                for i in range(self._n_actions):
                    probs_recall[i] += float(self._recall_vote_weight) * float(recall_prior[i])
                    
            if self._sample:
                p_sum = float(np.sum(probs_recall))
                if p_sum > 0:
                    probs_recall = probs_recall / p_sum
                else:
                    probs_recall = probs_base
                action_int = int(np.random.choice(self._n_actions, p=probs_recall))
            else:
                action_int = int(np.argmax(probs_recall))

            action_int = max(0, min(self._n_actions - 1, action_int))
            action_ret = action_int
            self._action_history.append(int(action_int))

        self._state_history.append(list(state_vec))"""
if "action = max(0, min(self._n_actions - 1, action))" in ta_content:
    ta_content = ta_content.replace(act_old, act_new)
    
act_ret_old = """        return ActionResult(
            action=int(action),
            info=info,
        )"""
act_ret_new = """        return ActionResult(
            action=action_ret,
            info=info,
        )"""
if act_ret_old in ta_content:
    ta_content = ta_content.replace(act_ret_old, act_ret_new)

# 3. Handle online learning update (if continuous, MSE loss)
learn_old = """            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                sampled["actions"].reshape(-1),
                ignore_index=-100,
            )"""
learn_new = """            if self._action_space_type == "continuous":
                valid = sampled["attention_mask"] > 0
                loss = F.mse_loss(logits[valid], sampled["actions"][valid].float())
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    sampled["actions"].reshape(-1),
                    ignore_index=-100,
                )"""
if learn_old in ta_content:
    ta_content = ta_content.replace(learn_old, learn_new)

with open("agents/transformer_agent.py", "w") as f:
    f.write(ta_content)

print("transformer_agent.py patched!")

# 4. Also update train_adt.py and prep_adt_data.py
# If time, I'll update those separately via manual replace, or read them here.
