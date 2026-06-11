"""Tune simple GCN hyperparameters on the random validation split."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_gnn import (
    GCNLayer,
    MoleculeDataset,
    load_split,
    make_loader,
    metrics,
    predict,
    run_epoch,
    set_seed,
)


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("reports/tuning/gnn")


class FlexibleGCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = [GCNLayer(input_dim, hidden_dim)]
        layers.extend(GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 1))
        self.layers = nn.ModuleList(layers)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, adj))
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.regressor(pooled).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def config_grid() -> list[dict[str, object]]:
    hidden_options = [64, 128]
    layer_options = [2, 3]
    dropout_options = [0.0, 0.1]
    lr_options = [1e-3]
    return [
        {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": 1e-4,
        }
        for hidden_dim, num_layers, dropout, lr in product(
            hidden_options,
            layer_options,
            dropout_options,
            lr_options,
        )
    ]


def plot_best_loss(history: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_mse"], label="train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="validation MSE")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title("Best tuned GCN loss - random split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, s=12)
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Rzeczywiste pIC50")
    plt.ylabel("Predykcja pIC50")
    plt.title("Best tuned GCN predictions - random test")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def train_config(
    config_id: int,
    config: dict[str, object],
    train_loader,
    val_loader,
    test_loader,
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, torch.Tensor]]:
    model = FlexibleGCNRegressor(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    best_val_r2 = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_epoch(model, val_loader, loss_fn, None, device)
        y_val, pred_val = predict(model, val_loader, device)
        val_r2 = metrics(y_val, pred_val)["R2"]

        history_rows.append(
            {
                "config_id": config_id,
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "val_r2": val_r2,
            }
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(best_state)
    y_train, pred_train = predict(model, train_loader, device)
    y_val, pred_val = predict(model, val_loader, device)
    y_test, pred_test = predict(model, test_loader, device)

    row = {
        "config_id": config_id,
        **config,
        "best_epoch": best_epoch,
        "train_R2": metrics(y_train, pred_train)["R2"],
        "val_R2": metrics(y_val, pred_val)["R2"],
        "test_R2": metrics(y_test, pred_test)["R2"],
        "test_MAE": metrics(y_test, pred_test)["MAE"],
        "test_RMSE": metrics(y_test, pred_test)["RMSE"],
    }

    return row, pd.DataFrame(history_rows), best_state


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MoleculeDataset(load_split(args.splits_dir, "random", "train"))
    val_dataset = MoleculeDataset(load_split(args.splits_dir, "random", "val"))
    test_dataset = MoleculeDataset(load_split(args.splits_dir, "random", "test"))

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)
    input_dim = train_dataset[0].x.shape[1]

    metrics_rows = []
    histories = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    for config_id, config in enumerate(config_grid(), start=1):
        row, history, state = train_config(
            config_id=config_id,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dim=input_dim,
            args=args,
            device=device,
        )
        metrics_rows.append(row)
        histories.append(history)
        states[config_id] = state
        print(f"Finished config {config_id}: val_R2={row['val_R2']:.4f}, test_R2={row['test_R2']:.4f}")

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("val_R2", ascending=False)
    history_frame = pd.concat(histories, ignore_index=True)
    best = metrics_frame.iloc[0].to_dict()
    best_config_id = int(best["config_id"])
    best_history = history_frame[history_frame["config_id"] == best_config_id].copy()

    best_model = FlexibleGCNRegressor(
        input_dim=input_dim,
        hidden_dim=int(best["hidden_dim"]),
        num_layers=int(best["num_layers"]),
        dropout=float(best["dropout"]),
    ).to(device)
    best_model.load_state_dict(states[best_config_id])
    y_test, pred_test = predict(best_model, test_loader, device)

    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    history_frame.to_csv(args.output_dir / "all_loss_history.csv", index=False)
    best_history.to_csv(args.output_dir / "best_loss_history.csv", index=False)
    (args.output_dir / "best_config.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    torch.save(states[best_config_id], args.output_dir / "best_model.pt")

    plot_best_loss(best_history, args.output_dir / "best_loss.png")
    plot_predictions(y_test, pred_test, args.output_dir / "best_test_predictions.png")

    print("Best tuned GCN config:")
    print(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
