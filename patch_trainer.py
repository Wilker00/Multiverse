import os

with open("tools/prep_adt_data.py", "r") as f:
    prep_content = f.read()

prep_action_parse_old = """            try:
                action = int(row.get("action"))
            except (TypeError, ValueError):
                continue
            if action < 0:
                continue"""
prep_action_parse_new = """            action_raw = row.get("action")
            action = None
            if isinstance(action_raw, (list, tuple)):
                try:
                    action = [float(x) for x in action_raw]
                except (TypeError, ValueError):
                    pass
            else:
                try:
                    action_val = int(action_raw)
                    if action_val >= 0:
                        action = action_val
                except (TypeError, ValueError):
                    pass
            if action is None:
                continue"""
if "action = int(row.get(\"action\"))" in prep_content:
    prep_content = prep_content.replace(prep_action_parse_old, prep_action_parse_new)

prep_infer_old = """    action_dim = 0
    for ep in episode_rows:
        for step in ep.steps:
            action_dim = max(int(action_dim), int(step.action) + 1)
    action_dim = max(int(action_dim), max(0, int(min_action_dim)))
    if action_dim <= 0:
        raise RuntimeError("action_dim inferred as 0; no valid discrete actions found")

    bos_token_id = int(action_dim)"""
prep_infer_new = """    action_dim = 0
    action_space_type = "discrete"
    bos_token_id = 0
    for ep in episode_rows:
        if ep.steps:
            if isinstance(ep.steps[0].action, (list, tuple)):
                action_space_type = "continuous"
                action_dim = len(ep.steps[0].action)
            break

    if action_space_type == "discrete":
        for ep in episode_rows:
            for step in ep.steps:
                try:
                    action_dim = max(int(action_dim), int(step.action) + 1)
                except Exception:
                    pass
        action_dim = max(int(action_dim), max(0, int(min_action_dim)))
        if action_dim <= 0:
            raise RuntimeError("action_dim inferred as 0; no valid discrete actions found")
        bos_token_id = int(action_dim)
    else:
        bos_token_id = [0.0] * action_dim"""
if "action_dim inferred as 0; no valid discrete actions found" in prep_content:
    prep_content = prep_content.replace(prep_infer_old, prep_infer_new)

prep_chunk_old = """            cur_prev = [int(bos_token_id) for _ in range(K)]
            cur_actions = [-100 for _ in range(K)]"""
prep_chunk_new = """            if action_space_type == "continuous":
                cur_prev = [list(bos_token_id) for _ in range(K)]
                cur_actions = [[-100.0] * action_dim for _ in range(K)]
            else:
                cur_prev = [int(bos_token_id) for _ in range(K)]
                cur_actions = [-100 for _ in range(K)]"""
if "cur_prev = [int(bos_token_id) for _ in range(K)]" in prep_content:
    prep_content = prep_content.replace(prep_chunk_old, prep_chunk_new)

prep_chunk_set_old = """                cur_rtg[j] = float(rtg[idx])
                cur_actions[j] = int(actions[idx])
                cur_prev[j] = int(bos_token_id) if idx == 0 else int(actions[idx - 1])"""
prep_chunk_set_new = """                cur_rtg[j] = float(rtg[idx])
                if action_space_type == "continuous":
                    cur_actions[j] = list(actions[idx])
                    cur_prev[j] = list(bos_token_id) if idx == 0 else list(actions[idx - 1])
                else:
                    cur_actions[j] = int(actions[idx])
                    cur_prev[j] = int(bos_token_id) if idx == 0 else int(actions[idx - 1])"""
if "cur_prev[j] = int(bos_token_id) if idx == 0 else int(actions[idx - 1])" in prep_content:
    prep_content = prep_content.replace(prep_chunk_set_old, prep_chunk_set_new)

prep_payload_old = """    payload = {
        "states": torch.tensor(states_all, dtype=torch.float32),
        "returns_to_go": torch.tensor(rtg_all, dtype=torch.float32),
        "prev_actions": torch.tensor(prev_actions_all, dtype=torch.long),
        "actions": torch.tensor(actions_all, dtype=torch.long),
        "timesteps": torch.tensor(timesteps_all, dtype=torch.long),
        "attention_mask": torch.tensor(mask_all, dtype=torch.float32),
        "meta": {"""
prep_payload_new = """    payload = {
        "states": torch.tensor(states_all, dtype=torch.float32),
        "returns_to_go": torch.tensor(rtg_all, dtype=torch.float32),
        "prev_actions": torch.tensor(prev_actions_all, dtype=torch.float32 if action_space_type == "continuous" else torch.long),
        "actions": torch.tensor(actions_all, dtype=torch.float32 if action_space_type == "continuous" else torch.long),
        "timesteps": torch.tensor(timesteps_all, dtype=torch.long),
        "attention_mask": torch.tensor(mask_all, dtype=torch.float32),
        "meta": {
            "action_space_type": str(action_space_type),"""
if "prev_actions_all, dtype=torch.long" in prep_content:
    prep_content = prep_content.replace(prep_payload_old, prep_payload_new)

with open("tools/prep_adt_data.py", "w") as f:
    f.write(prep_content)

print("prep_adt_data.py patched!")


with open("tools/train_adt.py", "r") as f:
    train_content = f.read()

train_loader_old = """    states = payload["states"].float()
    rtg = payload["returns_to_go"].float()
    prev_actions = payload["prev_actions"].long()
    actions = payload["actions"].long()
    timesteps = payload["timesteps"].long()
    attention_mask = payload["attention_mask"].float()"""
train_loader_new = """    states = payload["states"].float()
    rtg = payload["returns_to_go"].float()
    action_space_type = str(meta.get("action_space_type", "discrete"))
    if action_space_type == "continuous":
        prev_actions = payload["prev_actions"].float()
        actions = payload["actions"].float()
    else:
        prev_actions = payload["prev_actions"].long()
        actions = payload["actions"].long()
    timesteps = payload["timesteps"].long()
    attention_mask = payload["attention_mask"].float()"""
if "prev_actions = payload[\"prev_actions\"].long()" in train_content:
    train_content = train_content.replace(train_loader_old, train_loader_new)

train_loss_old = """    logits = model(
        states=states,
        returns_to_go=returns_to_go,
        prev_actions=prev_actions,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    action_dim = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, action_dim),
        targets.reshape(-1),
        weight=class_weights,
        ignore_index=-100,
    )

    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        valid = targets != -100
        correct = (pred == targets) & valid
        n_valid = int(valid.sum().item())
        acc = float(correct.sum().item() / max(1, n_valid))
    return loss, acc"""
train_loss_new = """    logits = model(
        states=states,
        returns_to_go=returns_to_go,
        prev_actions=prev_actions,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    action_dim = logits.shape[-1]
    
    if model.config.action_space_type == "continuous":
        # MSE Loss
        valid = attention_mask > 0
        if valid.any():
            loss = F.mse_loss(logits[valid], targets[valid])
            with torch.no_grad():
                # For continuous, accuracy isn't directly applicable, use pseudo accuracy (error threshold)
                err = torch.abs(logits[valid] - targets[valid])
                acc = float((err < 0.1).float().mean().item())
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            acc = 0.0
    else:
        # Cross Entropy
        loss = F.cross_entropy(
            logits.reshape(-1, action_dim),
            targets.reshape(-1),
            weight=class_weights,
            ignore_index=-100,
        )
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            valid = targets != -100
            correct = (pred == targets) & valid
            n_valid = int(valid.sum().item())
            acc = float(correct.sum().item() / max(1, n_valid))
    return loss, acc"""
if "loss = F.cross_entropy(" in train_content:
    train_content = train_content.replace(train_loss_old, train_loss_new)

train_cfg_old = """    cfg = DecisionTransformerConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        context_len=int(context_len),
        d_model=int(d_model),
        n_head=int(n_head),
        n_layer=int(n_layer),
        dropout=float(dropout),
        max_timestep=max(1, int(max_timestep)),
        bos_token_id=int(bos_token_id),
    )"""
train_cfg_new = """    cfg = DecisionTransformerConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        action_space_type=str(meta.get("action_space_type", "discrete")),
        context_len=int(context_len),
        d_model=int(d_model),
        n_head=int(n_head),
        n_layer=int(n_layer),
        dropout=float(dropout),
        max_timestep=max(1, int(max_timestep)),
        bos_token_id=int(bos_token_id) if str(meta.get("action_space_type", "discrete")) == "discrete" else 0, # bos_token_id not strictly used by continuous, but safe default
    )"""
if "cfg = DecisionTransformerConfig(" in train_content:
    train_content = train_content.replace(train_cfg_old, train_cfg_new)

# disable class weights for continuous
train_cw_old = """    class_weights_cpu, class_weight_meta = _build_class_weights(
        actions=actions,
        action_dim=int(action_dim),
        mode=str(class_weight_mode),
        min_count=max(1, int(class_weight_min_count)),
        max_weight=max(0.0, float(class_weight_max)),
    )"""
train_cw_new = """    if action_space_type == "continuous":
        class_weights_cpu, class_weight_meta = None, {"class_weight_mode_effective": "none"}
    else:
        class_weights_cpu, class_weight_meta = _build_class_weights(
            actions=actions,
            action_dim=int(action_dim),
            mode=str(class_weight_mode),
            min_count=max(1, int(class_weight_min_count)),
            max_weight=max(0.0, float(class_weight_max)),
        )"""
if "class_weights_cpu, class_weight_meta = _build_class_weights(" in train_content:
    train_content = train_content.replace(train_cw_old, train_cw_new)

with open("tools/train_adt.py", "w") as f:
    f.write(train_content)
print("train_adt.py patched!")

