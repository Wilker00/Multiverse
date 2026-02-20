"""
tools/train_selector.py

Trains the MicroSelector model using the pre-compiled training batch.
This script performs behavioral cloning to teach the selector to pick the
right skill for a given state and goal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import sys
from typing import Optional, Tuple

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from models.micro_selector import MicroSelector

def _make_dataloaders(
    states: torch.Tensor,
    goals: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset = TensorDataset(states, goals, labels)
    if val_split <= 0.0 or len(dataset) < 2:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None

    val_count = int(len(dataset) * val_split)
    if val_count <= 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None
    if val_count >= len(dataset):
        val_count = len(dataset) - 1

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_ds = TensorDataset(states[train_idx], goals[train_idx], labels[train_idx])
    val_ds = TensorDataset(states[val_idx], goals[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _class_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=classes).float()
    nonzero = counts > 0
    weights = torch.ones_like(counts)
    if nonzero.any():
        inv = counts[nonzero].sum() / counts[nonzero]
        inv = inv / inv.mean()
        weights[nonzero] = inv
    return weights


def _evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_states, batch_goals, batch_labels in dataloader:
            logits = model(batch_states, batch_goals)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == batch_labels).sum().item())
            total_examples += int(batch_labels.numel())
    denom = max(1, len(dataloader))
    acc = float(total_correct / max(1, total_examples))
    return total_loss / denom, acc


def train(
    data_path: str,
    model_save_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_embd: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    use_interaction_tokens: bool,
    use_cls_token: bool,
    ff_mult: int,
    use_deep_stem: bool,
    pooling: str,
    weight_decay: float,
    label_smoothing: float,
    grad_clip: float,
    val_split: float,
    patience: int,
    class_balance: bool,
    seed: int,
):
    """
    Loads the training data and runs the training loop for the MicroSelector.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at: {data_path}. Please run prepare_selector_data.py first.")

    # 1. Load Data
    print(f"Loading training data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    states = data['states']
    goals = data['goals']
    labels = data['labels']
    vocab = data['vocab']
    
    # The model expects state and goal to have the same dimension. We'll use the max of the two.
    state_dim = max(data['state_dim'], data['goal_dim'])
    
    # Pad the smaller tensor if dimensions are mismatched
    if states.shape[1] < state_dim:
        padding = torch.zeros(states.shape[0], state_dim - states.shape[1])
        states = torch.cat([states, padding], dim=1)
    if goals.shape[1] < state_dim:
        padding = torch.zeros(goals.shape[0], state_dim - goals.shape[1])
        goals = torch.cat([goals, padding], dim=1)

    train_loader, val_loader = _make_dataloaders(
        states=states,
        goals=goals,
        labels=labels,
        batch_size=batch_size,
        val_split=float(max(0.0, min(0.9, val_split))),
        seed=seed,
    )

    print(f"Loaded {len(states)} total training samples.")
    if val_loader is not None:
        train_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
        val_samples = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
        print(f"Train samples: {train_samples}, Val samples: {val_samples}")
    print(f"State dimension (padded): {state_dim}")
    print(f"Vocabulary size: {len(vocab)}")

    # 2. Initialize Model, Loss, and Optimizer
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
        use_deep_stem=bool(use_deep_stem),
        pooling=str(pooling),
    )
    class_w = _class_weights(labels, len(vocab)) if class_balance else None
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(max(0.0, min(0.5, label_smoothing))))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(max(0.0, weight_decay)))

    print("\\n--- Starting Training ---")
    best_metric = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0
        for batch_states, batch_goals, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_states, batch_goals)

            # Compute loss
            loss = criterion(logits, batch_labels)

            # Backward pass and optimization
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == batch_labels).sum().item())
            total_examples += int(batch_labels.numel())

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_acc = float(total_correct / max(1, total_examples))

        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion)
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )
            tracked = val_loss
        else:
            print(f"Epoch [{epoch+1}/{epochs}] train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}")
            tracked = avg_train_loss

        if tracked < best_metric:
            best_metric = tracked
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience}).")
                break

    print("--- Training Finished ---")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # 3. Save the trained model and its associated vocabulary
    out_dir = os.path.dirname(model_save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model_config = model.get_config()
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'state_dim': state_dim,
        'model_config': model_config,
        'train_config': {
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'lr': float(learning_rate),
            'weight_decay': float(max(0.0, weight_decay)),
            'label_smoothing': float(max(0.0, min(0.5, label_smoothing))),
            'grad_clip': float(max(0.0, grad_clip)),
            'val_split': float(max(0.0, min(0.9, val_split))),
            'patience': int(max(0, patience)),
            'class_balance': bool(class_balance),
            'seed': int(seed),
            'best_metric': float(best_metric),
            'use_deep_stem': bool(use_deep_stem),
            'pooling': str(pooling),
        },
    }, model_save_path)
    
    print(f"Trained MicroSelector model saved to: {model_save_path}")
    
    # --- Optimization Note ---
    # To get to 10MB, you would add these steps:
    # model.half() # Convert to FP16
    # scripted_model = torch.jit.script(model) # Export to TorchScript
    # torch.jit.save(scripted_model, "micro_selector_v1_10mb.pt")
    # print("FP16 optimized model saved to micro_selector_v1_10mb.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_batch.pt", help="Path to the pre-compiled training data.")
    parser.add_argument("--model_save_path", type=str, default="models/micro_selector.pt", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--n_embd", type=int, default=256, help="Transformer embedding width.")
    parser.add_argument("--n_head", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--n_layer", type=int, default=4, help="Transformer depth.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--use_interaction_tokens", action="store_true", help="Add delta/abs-delta tokens.")
    parser.add_argument("--use_cls_token", action="store_true", help="Use CLS pooling token.")
    parser.add_argument("--ff_mult", type=int, default=4, help="Feed-forward width multiplier.")
    parser.add_argument("--use_deep_stem", action="store_true", help="Use 2-layer token stems before transformer.")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "cls"], help="Pooling mode for selector head.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Cross-entropy label smoothing.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0 disables).")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs (0 disables).")
    parser.add_argument("--class_balance", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/validation split.")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        use_interaction_tokens=bool(args.use_interaction_tokens),
        use_cls_token=bool(args.use_cls_token),
        ff_mult=args.ff_mult,
        use_deep_stem=bool(args.use_deep_stem),
        pooling=str(args.pooling),
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        patience=args.patience,
        class_balance=bool(args.class_balance),
        seed=args.seed,
    )




