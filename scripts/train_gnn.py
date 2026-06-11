"""Train a small pure-PyTorch GCN baseline on molecular graphs from SMILES."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("reports/gnn")
DEFAULT_SUMMARY = Path("reports/gnn_summary.md")

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]


@dataclass
class GraphSample:
    x: torch.Tensor
    adj: torch.Tensor
    y: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atom_features(atom: Chem.Atom) -> list[float]:
    atomic_num = atom.GetAtomicNum()
    atom_type = [1.0 if atomic_num == item else 0.0 for item in ATOM_TYPES]
    atom_type.append(1.0 if atomic_num not in ATOM_TYPES else 0.0)

    degree = atom.GetDegree()
    degree_features = [1.0 if degree == item else 0.0 for item in range(6)]
    degree_features.append(1.0 if degree >= 6 else 0.0)

    return [
        *atom_type,
        *degree_features,
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def smiles_to_graph(smiles: str, target: float) -> GraphSample | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(features, dtype=torch.float32)

    num_atoms = mol.GetNumAtoms()
    adj = torch.eye(num_atoms, dtype=torch.float32)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adj[begin, end] = 1.0
        adj[end, begin] = 1.0

    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    normalized_adj = degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]

    return GraphSample(
        x=x,
        adj=normalized_adj,
        y=torch.tensor(float(target), dtype=torch.float32),
    )


class MoleculeDataset(Dataset[GraphSample]):
    def __init__(self, frame: pd.DataFrame) -> None:
        samples = []
        for row in frame.itertuples(index=False):
            graph = smiles_to_graph(row.canonical_smiles, row.pIC50)
            if graph is not None:
                samples.append(graph)
        if not samples:
            raise ValueError("No valid molecular graphs found.")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


def collate_graphs(batch: list[GraphSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_nodes = max(sample.x.shape[0] for sample in batch)
    feature_dim = batch[0].x.shape[1]

    x = torch.zeros(len(batch), max_nodes, feature_dim, dtype=torch.float32)
    adj = torch.zeros(len(batch), max_nodes, max_nodes, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_nodes, dtype=torch.float32)
    y = torch.zeros(len(batch), dtype=torch.float32)

    for idx, sample in enumerate(batch):
        num_nodes = sample.x.shape[0]
        x[idx, :num_nodes] = sample.x
        adj[idx, :num_nodes, :num_nodes] = sample.adj
        mask[idx, :num_nodes] = 1.0
        y[idx] = sample.y

    return x, adj, mask, y


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.bmm(adj, x))


class GCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gcn1(x, adj))
        h = torch.relu(self.gcn2(h, adj))
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.regressor(pooled).squeeze(-1)


def load_split(splits_dir: Path, split_type: str, split_name: str) -> pd.DataFrame:
    path = splits_dir / f"{split_type}_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def make_loader(dataset: Dataset[GraphSample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graphs)


def run_epoch(
    model: GCNRegressor,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    model.train(optimizer is not None)
    losses = []

    for x, adj, mask, y in loader:
        x = x.to(device)
        adj = adj.to(device)
        mask = mask.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        pred = model(x, adj, mask)
        loss = loss_fn(pred, y)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()) * len(y))

    return float(sum(losses) / len(loader.dataset))


def predict(model: GCNRegressor, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, adj, mask, y in loader:
            pred = model(x.to(device), adj.to(device), mask.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())

    return np.concatenate(targets), np.concatenate(preds)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def train_one_split(
    split_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset = MoleculeDataset(load_split(args.splits_dir, split_type, "train"))
    val_dataset = MoleculeDataset(load_split(args.splits_dir, split_type, "val"))
    test_dataset = MoleculeDataset(load_split(args.splits_dir, split_type, "test"))

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)

    input_dim = train_dataset[0].x.shape[1]
    model = GCNRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_epoch(model, val_loader, loss_fn, None, device)
        history_rows.append(
            {
                "split_type": split_type,
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
            }
        )

    metric_rows = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = predict(model, loader, device)
        metric_rows.append(
            {
                "split_type": split_type,
                "split": split_name,
                "rows": len(y_true),
                **metrics(y_true, y_pred),
            }
        )

    history = pd.DataFrame(history_rows)
    metric_frame = pd.DataFrame(metric_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / f"{split_type}_loss_history.csv", index=False)
    metric_frame.to_csv(args.output_dir / f"{split_type}_metrics.csv", index=False)
    plot_loss(history, args.output_dir / f"{split_type}_loss.png", split_type)

    y_true, y_pred = predict(model, test_loader, device)
    plot_predictions(y_true, y_pred, args.output_dir / f"{split_type}_test_predictions.png", split_type)

    return metric_frame, history


def plot_loss(history: pd.DataFrame, path: Path, split_type: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_mse"], label="train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="validation MSE")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title(f"GCN loss - {split_type} split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path, split_type: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, s=12)
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Rzeczywiste pIC50")
    plt.ylabel("Predykcja pIC50")
    plt.title(f"Predykcje GCN na teście - {split_type} split")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build_summary(metrics_frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> str:
    test_metrics = metrics_frame[metrics_frame["split"] == "test"].copy()
    return f"""# Krok 5: prosty model GCN na grafach cząsteczek

## Cel

Celem tego kroku było wytrenowanie pierwszego modelu grafowego na strukturach cząsteczek zapisanych jako `SMILES`.

Model nie używa jeszcze PyTorch Geometric. Dla prostoty projektowej zaimplementowano mały GCN w czystym PyTorch:

```text
GCN layers: 2
hidden_dim: {args.hidden_dim}
epochs: {args.epochs}
batch_size: {args.batch_size}
learning_rate: {args.lr}
```

## Wejście modelu

Dla każdego `SMILES` tworzony jest graf:

- atomy są węzłami,
- wiązania są krawędziami,
- dodawane są self-loopy,
- macierz sąsiedztwa jest normalizowana stopniami węzłów.

Cechy atomów obejmują m.in. typ atomu, stopień, ładunek formalny, liczbę atomów wodoru, aromatyczność i informację o pierścieniu.

## Metryki testowe

```text
{test_metrics.round(4).to_string(index=False)}
```

## Zapisane artefakty

Metryki:

```text
{output_dir / "metrics.csv"}
{output_dir / "random_metrics.csv"}
{output_dir / "scaffold_metrics.csv"}
```

Historia loss:

```text
{output_dir / "loss_history.csv"}
{output_dir / "random_loss_history.csv"}
{output_dir / "scaffold_loss_history.csv"}
```

Wykresy:

```text
{output_dir / "random_loss.png"}
{output_dir / "scaffold_loss.png"}
{output_dir / "random_test_predictions.png"}
{output_dir / "scaffold_test_predictions.png"}
```

## Wniosek

To jest pierwszy, prosty model grafowy. Jego wynik należy porównać z baseline MLP na deskryptorach RDKit, a nie traktować jako finalny model.
"""


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_metrics = []
    all_history = []
    for split_type in ["random", "scaffold"]:
        metric_frame, history = train_one_split(split_type, args, device)
        all_metrics.append(metric_frame)
        all_history.append(history)

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    history_frame = pd.concat(all_history, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    history_frame.to_csv(args.output_dir / "loss_history.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    args.summary.write_text(build_summary(metrics_frame, args.output_dir, args), encoding="utf-8")

    print(f"Saved GCN metrics in: {args.output_dir}")
    print(f"Saved GCN summary: {args.summary}")


if __name__ == "__main__":
    main()
