"""
patch_verse_embedding.py

Adds verse embedding token to the Decision Transformer architecture.
Threads verse_id through:
  1. DecisionTransformerConfig (n_verses, verse_to_id mapping)
  2. DecisionTransformer (verse_embed layer, forward/predict signatures)
  3. prep_adt_data.py (emit verse_ids tensor + verse_to_id map)
  4. train_adt.py (pass verse_ids to model + _compute_loss_and_acc)
  5. TransformerAgent (set verse_id at init, pass at inference, action masking)
"""
import os
import sys
import re


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  patched {path}")


def patch_decision_transformer():
    path = "models/decision_transformer.py"
    c = read(path)

    # 1. Add n_verses + verse_action_ranges to config
    c = c.replace(
        "    bos_token_id: Optional[int] = None",
        "    bos_token_id: Optional[int] = None\n"
        "    n_verses: int = 0\n"
        "    verse_to_id: Optional[Dict[str, int]] = None\n"
        "    verse_action_ranges: Optional[Dict[str, int]] = None"
    )

    # Need Dict import
    c = c.replace(
        "from typing import Any, Dict, Optional, Tuple",
        "from typing import Any, Dict, List, Optional, Tuple"
    )

    # 2. Add verse_embed layer in __init__
    c = c.replace(
        "        self.embed_ln = nn.LayerNorm(d_model)\n"
        "        self.embed_drop = nn.Dropout(float(self.config.dropout))",
        "        n_verses = max(1, int(self.config.n_verses)) if int(self.config.n_verses) > 0 else 0\n"
        "        self.verse_embed = nn.Embedding(max(1, n_verses if n_verses > 0 else 1), d_model) if n_verses > 0 else None\n"
        "        self.embed_ln = nn.LayerNorm(d_model)\n"
        "        self.embed_drop = nn.Dropout(float(self.config.dropout))"
    )

    # 3. Add verse_ids to forward signature
    c = c.replace(
        "        attention_mask: Optional[torch.Tensor] = None,\n"
        "    ) -> torch.Tensor:",
        "        attention_mask: Optional[torch.Tensor] = None,\n"
        "        verse_ids: Optional[torch.Tensor] = None,\n"
        "    ) -> torch.Tensor:"
    )

    # 4. Add verse embedding to the sum in forward
    c = c.replace(
        "        x = self.embed_ln(x_state + x_rtg + x_prev + x_time)",
        "        x_verse = torch.zeros_like(x_state)\n"
        "        if self.verse_embed is not None and verse_ids is not None:\n"
        "            # verse_ids: [B] integer tensor\n"
        "            v_ids = verse_ids.long().clamp(0, self.verse_embed.num_embeddings - 1)\n"
        "            x_verse = self.verse_embed(v_ids).unsqueeze(1).expand_as(x_state)\n"
        "        x = self.embed_ln(x_state + x_rtg + x_prev + x_time + x_verse)"
    )

    # 5. Add verse_ids to predict_next_action signature
    c = c.replace(
        "        top_k: int = 0,\n"
        "        sample: bool = False,\n"
        "    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:",
        "        top_k: int = 0,\n"
        "        sample: bool = False,\n"
        "        verse_ids: Optional[torch.Tensor] = None,\n"
        "        valid_action_n: Optional[int] = None,\n"
        "    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:"
    )

    # 6. Pass verse_ids through in predict_next_action
    c = c.replace(
        "        logits = self.forward(\n"
        "            states=states,\n"
        "            returns_to_go=returns_to_go,\n"
        "            prev_actions=prev_actions,\n"
        "            timesteps=timesteps,\n"
        "            attention_mask=attention_mask,\n"
        "        )",
        "        logits = self.forward(\n"
        "            states=states,\n"
        "            returns_to_go=returns_to_go,\n"
        "            prev_actions=prev_actions,\n"
        "            timesteps=timesteps,\n"
        "            attention_mask=attention_mask,\n"
        "            verse_ids=verse_ids,\n"
        "        )"
    )

    # 7. Add action masking after temperature scaling, before softmax
    c = c.replace(
        "        if int(top_k) > 0 and int(top_k) < int(self.config.action_dim):",
        "        # Action masking: zero out logits for invalid actions\n"
        "        if valid_action_n is not None and int(valid_action_n) > 0 and int(valid_action_n) < int(self.config.action_dim):\n"
        "            last_logits[:, int(valid_action_n):] = -1e9\n"
        "        if int(top_k) > 0 and int(top_k) < int(self.config.action_dim):"
    )

    # 8. Handle verse_to_id and verse_action_ranges in get_config
    # (already works via dataclass __dict__)

    # 9. Handle loading with new fields gracefully
    c = c.replace(
        "    model = DecisionTransformer(DecisionTransformerConfig(**cfg))",
        "    # Filter unknown fields for backward compat\n"
        "    import dataclasses as _dc\n"
        "    _valid_fields = {f.name for f in _dc.fields(DecisionTransformerConfig)}\n"
        "    cfg_filtered = {k: v for k, v in cfg.items() if k in _valid_fields}\n"
        "    model = DecisionTransformer(DecisionTransformerConfig(**cfg_filtered))"
    )

    write(path, c)


def patch_prep_adt_data():
    path = "tools/prep_adt_data.py"
    c = read(path)

    # 1. Build verse_to_id mapping and verse_ids_all list alongside states_all etc.
    c = c.replace(
        "    states_all: List[List[List[float]]] = []\n"
        "    rtg_all: List[List[float]] = []\n"
        "    prev_actions_all: List[List[int]] = []\n"
        "    actions_all: List[List[int]] = []\n"
        "    timesteps_all: List[List[int]] = []\n"
        "    mask_all: List[List[float]] = []",
        "    # Build verse_to_id mapping\n"
        "    all_verse_names = sorted({ep.verse_name for ep in episode_rows if ep.verse_name})\n"
        "    verse_to_id = {name: idx for idx, name in enumerate(all_verse_names)}\n"
        "\n"
        "    states_all: List[List[List[float]]] = []\n"
        "    rtg_all: List[List[float]] = []\n"
        "    prev_actions_all: List[List[int]] = []\n"
        "    actions_all: List[List[int]] = []\n"
        "    timesteps_all: List[List[int]] = []\n"
        "    mask_all: List[List[float]] = []\n"
        "    verse_ids_all: List[int] = []"
    )

    # 2. Append verse_id for each chunk
    c = c.replace(
        "            states_all.append(cur_states)\n"
        "            rtg_all.append(cur_rtg)\n"
        "            prev_actions_all.append(cur_prev)\n"
        "            actions_all.append(cur_actions)\n"
        "            timesteps_all.append(cur_t)\n"
        "            mask_all.append(cur_mask)",
        "            states_all.append(cur_states)\n"
        "            rtg_all.append(cur_rtg)\n"
        "            prev_actions_all.append(cur_prev)\n"
        "            actions_all.append(cur_actions)\n"
        "            timesteps_all.append(cur_t)\n"
        "            mask_all.append(cur_mask)\n"
        "            verse_ids_all.append(verse_to_id.get(ep.verse_name, 0))"
    )

    # 3. Handle action balance slicing for verse_ids too
    c = c.replace(
        "                mask_all = [mask_all[i] for i in keep]\n"
        "                action_balance_meta[\"action_balance_applied\"] = True",
        "                mask_all = [mask_all[i] for i in keep]\n"
        "                verse_ids_all = [verse_ids_all[i] for i in keep]\n"
        "                action_balance_meta[\"action_balance_applied\"] = True"
    )

    # 4. Add verse_ids tensor + verse_to_id to payload
    c = c.replace(
        '    payload = {\n'
        '        "states": torch.tensor(states_all, dtype=torch.float32),\n'
        '        "returns_to_go": torch.tensor(rtg_all, dtype=torch.float32),',
        '    payload = {\n'
        '        "states": torch.tensor(states_all, dtype=torch.float32),\n'
        '        "returns_to_go": torch.tensor(rtg_all, dtype=torch.float32),\n'
        '        "verse_ids": torch.tensor(verse_ids_all, dtype=torch.long),'
    )

    # 5. Add verse_to_id and n_verses to meta
    c = c.replace(
        '            "samples": int(len(states_all)),',
        '            "samples": int(len(states_all)),\n'
        '            "n_verses": int(len(verse_to_id)),\n'
        '            "verse_to_id": dict(verse_to_id),'
    )

    write(path, c)


def patch_train_adt():
    path = "tools/train_adt.py"
    c = read(path)

    # 1. Grab verse_ids from dataset
    c = c.replace(
        "    n = int(states.shape[0])\n"
        "    if n <= 0:\n"
        "        raise RuntimeError(\"empty ADT dataset\")",
        "    verse_ids = payload.get(\"verse_ids\")\n"
        "    if verse_ids is not None:\n"
        "        verse_ids = verse_ids.long()\n"
        "    n = int(states.shape[0])\n"
        "    if n <= 0:\n"
        "        raise RuntimeError(\"empty ADT dataset\")"
    )

    # 2. Add n_verses + verse_to_id to config construction
    c = c.replace(
        "        bos_token_id=int(bos_token_id) if str(meta.get(\"action_space_type\", \"discrete\")) == \"discrete\" else 0, # bos_token_id not strictly used by continuous, but safe default\n"
        "    )",
        "        bos_token_id=int(bos_token_id) if str(meta.get(\"action_space_type\", \"discrete\")) == \"discrete\" else 0, # bos_token_id not strictly used by continuous, but safe default\n"
        "        n_verses=_safe_int(meta.get(\"n_verses\"), 0),\n"
        "        verse_to_id=meta.get(\"verse_to_id\"),\n"
        "    )"
    )

    # 3. Add verse_ids to forward signature in _compute_loss_and_acc
    c = c.replace(
        "def _compute_loss_and_acc(\n"
        "    *,\n"
        "    model: DecisionTransformer,\n"
        "    states: torch.Tensor,\n"
        "    returns_to_go: torch.Tensor,\n"
        "    prev_actions: torch.Tensor,\n"
        "    timesteps: torch.Tensor,\n"
        "    attention_mask: torch.Tensor,\n"
        "    targets: torch.Tensor,\n"
        "    class_weights: Optional[torch.Tensor] = None,\n"
        ") -> Tuple[torch.Tensor, float]:",
        "def _compute_loss_and_acc(\n"
        "    *,\n"
        "    model: DecisionTransformer,\n"
        "    states: torch.Tensor,\n"
        "    returns_to_go: torch.Tensor,\n"
        "    prev_actions: torch.Tensor,\n"
        "    timesteps: torch.Tensor,\n"
        "    attention_mask: torch.Tensor,\n"
        "    targets: torch.Tensor,\n"
        "    class_weights: Optional[torch.Tensor] = None,\n"
        "    verse_ids: Optional[torch.Tensor] = None,\n"
        ") -> Tuple[torch.Tensor, float]:"
    )

    # 4. Pass verse_ids in the model call inside _compute_loss_and_acc
    c = c.replace(
        "    logits = model(\n"
        "        states=states,\n"
        "        returns_to_go=returns_to_go,\n"
        "        prev_actions=prev_actions,\n"
        "        timesteps=timesteps,\n"
        "        attention_mask=attention_mask,\n"
        "    )",
        "    logits = model(\n"
        "        states=states,\n"
        "        returns_to_go=returns_to_go,\n"
        "        prev_actions=prev_actions,\n"
        "        timesteps=timesteps,\n"
        "        attention_mask=attention_mask,\n"
        "        verse_ids=verse_ids,\n"
        "    )"
    )

    # 5. Pass verse_ids in training loop batch
    # Training batch
    c = c.replace(
        "            batch_y = actions[bi_t].to(runtime_device)\n\n"
        "            loss, acc = _compute_loss_and_acc(\n"
        "                model=model,\n"
        "                states=batch_states,\n"
        "                returns_to_go=batch_rtg,\n"
        "                prev_actions=batch_prev,\n"
        "                timesteps=batch_t,\n"
        "                attention_mask=batch_mask,\n"
        "                targets=batch_y,\n"
        "                class_weights=class_weights_runtime,\n"
        "            )\n"
        "            optimizer.zero_grad()",
        "            batch_y = actions[bi_t].to(runtime_device)\n"
        "            batch_verse = verse_ids[bi_t].to(runtime_device) if verse_ids is not None else None\n\n"
        "            loss, acc = _compute_loss_and_acc(\n"
        "                model=model,\n"
        "                states=batch_states,\n"
        "                returns_to_go=batch_rtg,\n"
        "                prev_actions=batch_prev,\n"
        "                timesteps=batch_t,\n"
        "                attention_mask=batch_mask,\n"
        "                targets=batch_y,\n"
        "                class_weights=class_weights_runtime,\n"
        "                verse_ids=batch_verse,\n"
        "            )\n"
        "            optimizer.zero_grad()"
    )

    # Validation batch
    c = c.replace(
        "                    batch_y = actions[bi_t].to(runtime_device)\n"
        "                    loss, acc = _compute_loss_and_acc(\n"
        "                        model=model,\n"
        "                        states=batch_states,\n"
        "                        returns_to_go=batch_rtg,\n"
        "                        prev_actions=batch_prev,\n"
        "                        timesteps=batch_t,\n"
        "                        attention_mask=batch_mask,\n"
        "                        targets=batch_y,\n"
        "                        class_weights=class_weights_runtime,\n"
        "                    )",
        "                    batch_y = actions[bi_t].to(runtime_device)\n"
        "                    batch_verse = verse_ids[bi_t].to(runtime_device) if verse_ids is not None else None\n"
        "                    loss, acc = _compute_loss_and_acc(\n"
        "                        model=model,\n"
        "                        states=batch_states,\n"
        "                        returns_to_go=batch_rtg,\n"
        "                        prev_actions=batch_prev,\n"
        "                        timesteps=batch_t,\n"
        "                        attention_mask=batch_mask,\n"
        "                        targets=batch_y,\n"
        "                        class_weights=class_weights_runtime,\n"
        "                        verse_ids=batch_verse,\n"
        "                    )"
    )

    write(path, c)


def patch_transformer_agent():
    path = "agents/transformer_agent.py"
    c = read(path)

    # 1. Add verse_id resolution + action masking config after model load
    c = c.replace(
        "        # Allow checkpoint action_dim > n_actions for cross-verse padded spaces.",
        "        # Resolve verse_id from checkpoint's verse_to_id mapping\n"
        "        self._verse_id: Optional[int] = None\n"
        "        self._valid_action_n: Optional[int] = None\n"
        "        verse_to_id = model_cfg.get(\"verse_to_id\")\n"
        "        verse_action_ranges = model_cfg.get(\"verse_action_ranges\")\n"
        "        if isinstance(verse_to_id, dict) and self._verse_name:\n"
        "            vn = str(self._verse_name).strip().lower()\n"
        "            self._verse_id = verse_to_id.get(vn)\n"
        "        if isinstance(verse_action_ranges, dict) and self._verse_name:\n"
        "            vn = str(self._verse_name).strip().lower()\n"
        "            if vn in verse_action_ranges:\n"
        "                self._valid_action_n = int(verse_action_ranges[vn])\n"
        "        # If no explicit range from checkpoint, use the env's action space\n"
        "        if self._valid_action_n is None and int(self._n_actions) < self._model_action_dim:\n"
        "            self._valid_action_n = int(self._n_actions)\n"
        "        # Allow checkpoint action_dim > n_actions for cross-verse padded spaces."
    )

    # 2. Add verse_ids to _build_inputs output
    c = c.replace(
        '            "seq_len": torch.tensor([seq_len], dtype=torch.long, device=self._device),\n'
        "        }",
        '            "seq_len": torch.tensor([seq_len], dtype=torch.long, device=self._device),\n'
        '            "verse_ids": torch.tensor([self._verse_id if self._verse_id is not None else 0], dtype=torch.long, device=self._device),\n'
        "        }"
    )

    # 3. Pass verse_ids + valid_action_n to predict_next_action in act_with_hint
    c = c.replace(
        '            action_t, conf_t, probs_t = self.model.predict_next_action(\n'
        '                states=model_in["states"],\n'
        '                returns_to_go=model_in["returns_to_go"],\n'
        '                prev_actions=model_in["prev_actions"],\n'
        '                timesteps=model_in["timesteps"],\n'
        '                attention_mask=model_in["attention_mask"],\n'
        '                temperature=float(self._temperature),\n'
        '                top_k=int(self._top_k),\n'
        '                sample=bool(self._sample),\n'
        "            )",
        '            action_t, conf_t, probs_t = self.model.predict_next_action(\n'
        '                states=model_in["states"],\n'
        '                returns_to_go=model_in["returns_to_go"],\n'
        '                prev_actions=model_in["prev_actions"],\n'
        '                timesteps=model_in["timesteps"],\n'
        '                attention_mask=model_in["attention_mask"],\n'
        '                temperature=float(self._temperature),\n'
        '                top_k=int(self._top_k),\n'
        '                sample=bool(self._sample),\n'
        '                verse_ids=model_in.get("verse_ids"),\n'
        '                valid_action_n=self._valid_action_n,\n'
        "            )"
    )

    # 4. Also patch the _old_act fallback
    c = c.replace(
        '            action_t, conf_t, _ = self.model.predict_next_action(\n'
        '                states=model_in["states"],\n'
        '                returns_to_go=model_in["returns_to_go"],\n'
        '                prev_actions=model_in["prev_actions"],\n'
        '                timesteps=model_in["timesteps"],\n'
        '                attention_mask=model_in["attention_mask"],\n'
        '                temperature=float(self._temperature),\n'
        '                top_k=int(self._top_k),\n'
        '                sample=bool(self._sample),\n'
        "            )",
        '            action_t, conf_t, _ = self.model.predict_next_action(\n'
        '                states=model_in["states"],\n'
        '                returns_to_go=model_in["returns_to_go"],\n'
        '                prev_actions=model_in["prev_actions"],\n'
        '                timesteps=model_in["timesteps"],\n'
        '                attention_mask=model_in["attention_mask"],\n'
        '                temperature=float(self._temperature),\n'
        '                top_k=int(self._top_k),\n'
        '                sample=bool(self._sample),\n'
        '                verse_ids=model_in.get("verse_ids"),\n'
        '                valid_action_n=self._valid_action_n,\n'
        "            )"
    )

    write(path, c)


if __name__ == "__main__":
    print("Patching verse embedding across stack...")
    patch_decision_transformer()
    patch_prep_adt_data()
    patch_train_adt()
    patch_transformer_agent()
    print("Done! All 4 files patched.")
