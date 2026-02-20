"""
models/micro_selector.py

Contains the MicroSelector, a small, CPU-friendly Transformer model for
selecting the best Skill Gene (Lesson) to execute.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MicroSelector(nn.Module):
    """
    A small Decision Transformer-style model that predicts a Skill ID.
    It takes the current state and a goal state, and outputs logits over the
    vocabulary of known skills (lessons).
    """
    def __init__(
        self,
        state_dim: int,
        skill_vocab_size: int,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        dropout: float = 0.0,
        use_interaction_tokens: bool = False,
        use_cls_token: bool = False,
        ff_mult: int = 4,
        use_deep_stem: bool = False,
        pooling: str = "last",
    ):
        super().__init__()

        if skill_vocab_size <= 0:
            raise ValueError("skill_vocab_size must be greater than 0.")
        if n_embd <= 0:
            raise ValueError("n_embd must be greater than 0.")
        if n_head <= 0:
            raise ValueError("n_head must be greater than 0.")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if ff_mult <= 0:
            raise ValueError("ff_mult must be greater than 0.")
        pool_mode = str(pooling or "last").strip().lower()
        if pool_mode not in {"last", "mean", "cls"}:
            raise ValueError("pooling must be one of: last, mean, cls")

        def _enc() -> nn.Module:
            if not bool(use_deep_stem):
                return nn.Linear(state_dim, n_embd)
            return nn.Sequential(
                nn.Linear(state_dim, n_embd),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                nn.Linear(n_embd, n_embd),
            )

        # 1. State & Goal Embedding
        self.state_encoder = _enc()
        self.goal_encoder = _enc()
        self.use_interaction_tokens = bool(use_interaction_tokens)
        self.use_cls_token = bool(use_cls_token)
        self.pooling = pool_mode

        if self.use_interaction_tokens:
            self.delta_encoder = _enc()
            self.abs_delta_encoder = _enc()
        else:
            self.delta_encoder = None
            self.abs_delta_encoder = None

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        else:
            self.cls_token = None

        self.num_tokens = 2 + (2 if self.use_interaction_tokens else 0) + (1 if self.use_cls_token else 0)

        # 2. Positional Encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, n_embd))
        self.token_dropout = nn.Dropout(float(max(0.0, dropout)))

        # 3. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_mult,
            dropout=float(max(0.0, dropout)),
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 4. Skill Head
        self.head_norm = nn.LayerNorm(n_embd)
        self.skill_head = nn.Linear(n_embd, skill_vocab_size)
        self._config = {
            "n_embd": int(n_embd),
            "n_head": int(n_head),
            "n_layer": int(n_layer),
            "dropout": float(max(0.0, dropout)),
            "use_interaction_tokens": bool(self.use_interaction_tokens),
            "use_cls_token": bool(self.use_cls_token),
            "ff_mult": int(ff_mult),
            "use_deep_stem": bool(use_deep_stem),
            "pooling": str(pool_mode),
        }
        self._reset_parameters()

        print(f"MicroSelector initialized with ~{self._calculate_params()/1e6:.2f}M parameters.")

    def _calculate_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_emb, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def get_config(self) -> Dict[str, object]:
        return dict(self._config)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # Ensure input tensors are 2D (batch, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        tokens = []
        if self.cls_token is not None:
            tokens.append(self.cls_token.expand(state.shape[0], -1, -1).squeeze(1))

        # Base prompt is [goal, current_state]
        tokens.append(self.goal_encoder(goal))
        tokens.append(self.state_encoder(state))

        if self.use_interaction_tokens and self.delta_encoder is not None and self.abs_delta_encoder is not None:
            delta = state - goal
            tokens.append(self.delta_encoder(delta))
            tokens.append(self.abs_delta_encoder(torch.abs(delta)))

        x = torch.stack(tokens, dim=1)
        x = x + self.pos_emb[:, :x.shape[1], :]
        x = self.token_dropout(x)
        features = self.transformer(x)

        if self.pooling == "cls":
            if self.cls_token is not None:
                pooled = features[:, 0, :]
            else:
                pooled = features.mean(dim=1)
        elif self.pooling == "mean":
            pooled = features.mean(dim=1)
        elif self.cls_token is not None:
            pooled = features[:, 0, :]
        else:
            pooled = features[:, -1, :]
        pooled = self.head_norm(pooled)
        return self.skill_head(pooled)

def create_lesson_vocab(lessons_dir: str) -> Dict[str, int]:
    """Creates a mapping from lesson filename (Skill ID) to an integer index."""
    if not os.path.isdir(lessons_dir):
        return {}
    
    # Sort for deterministic ordering
    lesson_files = sorted([f for f in os.listdir(lessons_dir) if f.endswith(".txt")])
    return {lesson_name: i for i, lesson_name in enumerate(lesson_files)}

def _hash_bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(1, dim)


def _tokens_to_vector(tokens: List[str], dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    if not tokens:
        return vec
    for token in tokens:
        idx = _hash_bucket(token, dim)
        vec[idx] += 1.0
    norm = torch.linalg.vector_norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _parse_lesson(path: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "title": "",
        "context": "",
        "utility_score": 1.0,
        "actions": [],
    }
    action_re = re.compile(r"DO_ACTION\((.*)\)")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("TITLE:"):
                fields["title"] = line.split(":", 1)[1].strip()
                continue
            if line.startswith("CONTEXT:"):
                fields["context"] = line.split(":", 1)[1].strip()
                continue
            if line.startswith("UTILITY_SCORE:"):
                try:
                    fields["utility_score"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    fields["utility_score"] = 1.0
                continue

            m = action_re.search(line)
            if m:
                payload = m.group(1).strip()
                fields["actions"].append(payload)
    return fields


def _build_training_batch(lessons_dir: str, state_dim: int, lesson_vocab: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states: List[torch.Tensor] = []
    goals: List[torch.Tensor] = []
    labels: List[int] = []

    for lesson_name, lesson_idx in lesson_vocab.items():
        path = os.path.join(lessons_dir, lesson_name)
        if not os.path.isfile(path):
            continue
        meta = _parse_lesson(path)
        context = str(meta.get("context", "")).strip()
        title = str(meta.get("title", "")).strip()
        actions = [str(a) for a in (meta.get("actions") or []) if str(a).strip()]
        utility = float(meta.get("utility_score", 1.0) or 1.0)

        if not actions:
            state_tokens = [f"context:{context}", f"title:{title}", "step:0"]
            goal_tokens = [f"context:{context}", f"title:{title}", "goal:unknown", "len:0"]
            states.append(_tokens_to_vector(state_tokens, state_dim))
            goals.append(_tokens_to_vector(goal_tokens, state_dim))
            labels.append(int(lesson_idx))
            continue

        upsample = max(1, min(5, int(round(utility))))
        final_action = actions[-1]
        for step_idx, action in enumerate(actions):
            state_tokens = [
                f"context:{context}",
                f"title:{title}",
                f"step:{step_idx}",
                f"action:{action}",
            ]
            goal_tokens = [
                f"context:{context}",
                f"title:{title}",
                f"goal:{final_action}",
                f"len:{len(actions)}",
            ]
            for _ in range(upsample):
                states.append(_tokens_to_vector(state_tokens, state_dim))
                goals.append(_tokens_to_vector(goal_tokens, state_dim))
                labels.append(int(lesson_idx))

    if not labels:
        return (
            torch.zeros((0, state_dim), dtype=torch.float32),
            torch.zeros((0, state_dim), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    return (
        torch.stack(states, dim=0).to(dtype=torch.float32),
        torch.stack(goals, dim=0).to(dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def train_selector_stub(lessons_dir: str, state_dim: int, model_save_path: str) -> Dict[str, Any]:
    """
    Trains a lightweight selector from lesson text files.

    For compatibility the function name remains `train_selector_stub`, but it now
    performs real supervised fitting on hashed lesson features.
    """
    print("\\n--- Training MicroSelector (Lesson-Based) ---")
    lesson_vocab = create_lesson_vocab(lessons_dir)

    if not lesson_vocab:
        print("No lessons found. Skipping MicroSelector training.")
        return {"saved": False, "reason": "no_lessons"}

    print(f"Found {len(lesson_vocab)} skills in the vocabulary.")
    state_dim = max(1, int(state_dim))
    states, goals, labels = _build_training_batch(lessons_dir, state_dim, lesson_vocab)
    if labels.numel() == 0:
        print("No valid training samples parsed from lessons. Skipping training.")
        return {"saved": False, "reason": "no_samples"}

    model = MicroSelector(
        state_dim=state_dim,
        skill_vocab_size=len(lesson_vocab),
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    dataset = TensorDataset(states, goals, labels)
    batch_size = max(1, min(64, int(len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_loss = float("inf")
    epochs = min(20, max(5, 3 + len(lesson_vocab)))
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_states, batch_goals, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_states, batch_goals)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(loader))
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    model.eval()
    with torch.no_grad():
        logits = model(states, goals)
        pred = torch.argmax(logits, dim=-1)
        train_acc = float((pred == labels).float().mean().item())

    out_dir = os.path.dirname(model_save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "saved": True,
        "num_lessons": int(len(lesson_vocab)),
        "num_samples": int(labels.numel()),
        "epochs": int(epochs),
        "best_loss": float(best_loss if best_loss < float("inf") else 0.0),
        "train_accuracy": float(train_acc),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lesson_vocab": lesson_vocab,
            "vocab": lesson_vocab,
            "state_dim": state_dim,
            "model_config": model.get_config(),
            "train_metrics": metrics,
        },
        model_save_path,
    )

    print(f"MicroSelector model and vocabulary saved to: {model_save_path}")
    print(
        f"Samples={metrics['num_samples']} "
        f"epochs={metrics['epochs']} "
        f"loss={metrics['best_loss']:.4f} "
        f"acc={metrics['train_accuracy']:.3f}"
    )
    print("------------------------------------")
    return metrics
