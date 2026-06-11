"""Train a simple physicochemical-feature baseline for pIC50 regression."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("reports/baseline")
DEFAULT_SUMMARY = Path("reports/baseline_summary.md")

FEATURE_COLUMNS = ["MW", "LogP", "TPSA", "HBD", "HBA", "rotatable_bonds", "heavy_atoms"]
TARGET_COLUMN = "pIC50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(splits_dir: Path, split_type: str, split_name: str) -> pd.DataFrame:
    path = splits_dir / f"{split_type}_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in [TARGET_COLUMN, *FEATURE_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(df)
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    batch_size=128,
                    max_iter=1,
                    warm_start=True,
                    shuffle=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def train_one_split(
    splits_dir: Path,
    split_type: str,
    output_dir: Path,
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_split(splits_dir, split_type, "train")
    val_df = load_split(splits_dir, split_type, "val")
    test_df = load_split(splits_dir, split_type, "test")

    x_train, y_train = split_xy(train_df)
    x_val, y_val = split_xy(val_df)
    x_test, y_test = split_xy(test_df)

    model = make_model(seed)
    history_rows = []

    for epoch in range(1, epochs + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)

        history_rows.append(
            {
                "split_type": split_type,
                "epoch": epoch,
                "train_mse": mean_squared_error(y_train, train_pred),
                "val_mse": mean_squared_error(y_val, val_pred),
                "train_r2": r2_score(y_train, train_pred),
                "val_r2": r2_score(y_val, val_pred),
            }
        )

    metrics_rows = []
    for split_name, x_data, y_data in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        pred = model.predict(x_data)
        metrics_rows.append(
            {
                "split_type": split_type,
                "split": split_name,
                "rows": len(y_data),
                **regression_metrics(y_data, pred),
            }
        )

    history = pd.DataFrame(history_rows)
    metrics = pd.DataFrame(metrics_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / f"{split_type}_loss_history.csv", index=False)
    metrics.to_csv(output_dir / f"{split_type}_metrics.csv", index=False)

    plot_loss(history, output_dir / f"{split_type}_loss.png", split_type)
    plot_predictions(model.predict(x_test), y_test, output_dir / f"{split_type}_test_predictions.png", split_type)

    return metrics, history


def plot_loss(history: pd.DataFrame, path: Path, split_type: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_mse"], label="train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="validation MSE")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title(f"Baseline MLP loss - {split_type} split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_predictions(y_pred: np.ndarray, y_true: pd.Series, path: Path, split_type: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, s=12)
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Rzeczywiste pIC50")
    plt.ylabel("Predykcja pIC50")
    plt.title(f"Predykcje baseline na teście - {split_type} split")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def build_summary(metrics: pd.DataFrame, output_dir: Path, epochs: int) -> str:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    return f"""# Krok 4: baseline na cechach fizykochemicznych

## Cel

Celem tego kroku było wytrenowanie prostego modelu regresyjnego przewidującego `pIC50` wyłącznie na podstawie cech fizykochemicznych RDKit.

Model:

```text
MLPRegressor
hidden layers: 64, 32
epochs: {epochs}
features: {", ".join(FEATURE_COLUMNS)}
target: pIC50
```

Model został wytrenowany osobno dla dwóch wariantów podziału danych:

- `random split`,
- `scaffold split`.

## Metryki testowe

```text
{test_metrics.round(4).to_string(index=False)}
```

Pełne metryki dla `train`, `validation` i `test` zapisano w:

```text
{output_dir / "random_metrics.csv"}
{output_dir / "scaffold_metrics.csv"}
```

## Krzywe loss

Historia loss została zapisana w:

```text
{output_dir / "random_loss_history.csv"}
{output_dir / "scaffold_loss_history.csv"}
```

Wykresy loss:

```text
{output_dir / "random_loss.png"}
{output_dir / "scaffold_loss.png"}
```

## Wykresy predykcji

Wykresy `predykcja pIC50` vs `rzeczywiste pIC50` dla zbiorów testowych:

```text
{output_dir / "random_test_predictions.png"}
{output_dir / "scaffold_test_predictions.png"}
```

## Wniosek

Ten baseline jest punktem odniesienia dla przyszłego modelu GNN. Jeżeli GNN nie przebije tego wyniku, to sama reprezentacja grafowa nie dała jeszcze praktycznej poprawy w obecnej konfiguracji.
"""


def main() -> None:
    args = parse_args()

    all_metrics = []
    all_history = []
    for split_type in ["random", "scaffold"]:
        metrics, history = train_one_split(
            splits_dir=args.splits_dir,
            split_type=split_type,
            output_dir=args.output_dir,
            epochs=args.epochs,
            seed=args.seed,
        )
        all_metrics.append(metrics)
        all_history.append(history)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    history_df = pd.concat(all_history, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    history_df.to_csv(args.output_dir / "loss_history.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    args.summary.write_text(build_summary(metrics_df, args.output_dir, args.epochs), encoding="utf-8")

    print(f"Saved baseline metrics in: {args.output_dir}")
    print(f"Saved baseline summary: {args.summary}")


if __name__ == "__main__":
    main()
