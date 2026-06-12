"""Train and save best_model.pt for the tuned GCN config (single run, no grid search)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train_gnn import MoleculeDataset, load_split, make_loader, metrics, predict, run_epoch, set_seed
from tune_gnn import FlexibleGCNRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data/splits"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "reports/tuning/gnn/best_config.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports/tuning/gnn/best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Missing config: {args.config}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MoleculeDataset(load_split(args.splits_dir, "random", "train"))
    val_dataset = MoleculeDataset(load_split(args.splits_dir, "random", "val"))
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    input_dim = train_dataset[0].x.shape[1]

    model = FlexibleGCNRegressor(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.MSELoss()

    best_val_r2 = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_epoch(model, val_loader, loss_fn, None, device)
        y_val, pred_val = predict(model, val_loader, device)
        val_r2 = metrics(y_val, pred_val)["R2"]
        print(f"Epoch {epoch}: train_mse={train_loss:.4f}, val_mse={val_loss:.4f}, val_R2={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"Saved weights to: {args.output}")


if __name__ == "__main__":
    main()
