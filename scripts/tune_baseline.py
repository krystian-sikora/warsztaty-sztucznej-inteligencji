"""Tune MLP baseline hyperparameters on the random validation split."""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["MW", "LogP", "TPSA", "HBD", "HBA", "rotatable_bonds", "heavy_atoms"]
TARGET_COLUMN = "pIC50"

DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_DIR = Path("reports/tuning/baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def load_xy(splits_dir: Path, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(splits_dir / f"random_{split_name}.csv")
    return frame[FEATURE_COLUMNS], frame[TARGET_COLUMN]


def metric_row(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def make_model(
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    batch_size=128,
                    max_iter=1,
                    warm_start=True,
                    shuffle=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def config_grid() -> list[dict[str, object]]:
    hidden_options = [(64, 32), (128, 64), (256, 128), (128, 64, 32)]
    alpha_options = [1e-4, 1e-3]
    lr_options = [1e-3]
    return [
        {
            "hidden_layer_sizes": hidden,
            "alpha": alpha,
            "learning_rate_init": lr,
        }
        for hidden, alpha, lr in product(hidden_options, alpha_options, lr_options)
    ]


def plot_best_loss(history: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_mse"], label="train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="validation MSE")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title("Best tuned MLP loss - random split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, s=12)
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Rzeczywiste pIC50")
    plt.ylabel("Predykcja pIC50")
    plt.title("Best tuned MLP predictions - random test")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_xy(args.splits_dir, "train")
    x_val, y_val = load_xy(args.splits_dir, "val")
    x_test, y_test = load_xy(args.splits_dir, "test")

    metrics_rows = []
    histories: dict[int, pd.DataFrame] = {}
    best_models: dict[int, Pipeline] = {}
    configs = config_grid()

    for config_id, config in enumerate(configs, start=1):
        model = make_model(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            alpha=float(config["alpha"]),
            learning_rate_init=float(config["learning_rate_init"]),
            seed=args.seed,
        )
        history_rows = []
        best_model = copy.deepcopy(model)
        best_val_r2 = -np.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)

            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            history_rows.append(
                {
                    "config_id": config_id,
                    "epoch": epoch,
                    "train_mse": mean_squared_error(y_train, train_pred),
                    "val_mse": mean_squared_error(y_val, val_pred),
                    "train_r2": r2_score(y_train, train_pred),
                    "val_r2": r2_score(y_val, val_pred),
                }
            )

            current_val_r2 = r2_score(y_val, val_pred)
            if current_val_r2 > best_val_r2:
                best_val_r2 = current_val_r2
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                break

        val_pred = best_model.predict(x_val)
        test_pred = best_model.predict(x_test)
        metrics_rows.append(
            {
                "config_id": config_id,
                **config,
                "best_epoch": best_epoch,
                "val_MAE": mean_absolute_error(y_val, val_pred),
                "val_RMSE": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "val_R2": r2_score(y_val, val_pred),
                "test_MAE": mean_absolute_error(y_test, test_pred),
                "test_RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "test_R2": r2_score(y_test, test_pred),
            }
        )
        histories[config_id] = pd.DataFrame(history_rows)
        best_models[config_id] = best_model

    metrics = pd.DataFrame(metrics_rows).sort_values("val_R2", ascending=False)
    best = metrics.iloc[0].to_dict()
    best_config_id = int(best["config_id"])
    best_history = histories[best_config_id]

    best_model = best_models[best_config_id]

    test_pred = best_model.predict(x_test)
    final_test_metrics = metric_row(y_test, test_pred)

    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(histories.values(), ignore_index=True).to_csv(args.output_dir / "all_loss_history.csv", index=False)
    best_history.to_csv(args.output_dir / "best_loss_history.csv", index=False)
    (args.output_dir / "best_config.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "best_test_metrics.json").write_text(
        json.dumps(final_test_metrics, indent=2),
        encoding="utf-8",
    )

    plot_best_loss(best_history, args.output_dir / "best_loss.png")
    plot_predictions(y_test, test_pred, args.output_dir / "best_test_predictions.png")

    print("Best tuned MLP config:")
    print(json.dumps(best, indent=2, default=str))
    print("Final random_test metrics:")
    print(json.dumps(final_test_metrics, indent=2))


if __name__ == "__main__":
    main()
