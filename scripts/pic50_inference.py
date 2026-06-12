"""Load tuned GCN weights and predict pIC50 from canonical SMILES."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pic50_tool_logging import log_tool_event
from train_gnn import collate_graphs, smiles_to_inference_graph
from tune_gnn import FlexibleGCNRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "reports/tuning/gnn/best_config.json"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "reports/tuning/gnn/best_model.pt"

TARGET_ID = "CHEMBL203"
TARGET_NAME = "EGFR"
MODEL_NAME = "tuned_gcn"
DISCLAIMER = (
    "Wynik dotyczy random-split EGFR (CHEMBL203); model GCN nie jest gotowy do odkrywania leków."
)

_CACHED_PREDICTOR: tuple[nn.Module, torch.device] | None = None


class WeightsNotFoundError(FileNotFoundError):
    """Raised when tuned model weights are missing."""


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved = _resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing model config: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_gcn_predictor(
    config_path: Path = DEFAULT_CONFIG_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    resolved_config = _resolve_path(config_path)
    resolved_weights = _resolve_path(weights_path)

    if not resolved_weights.exists():
        raise WeightsNotFoundError(
            f"Missing model weights: {resolved_weights}\n"
            "Run: python scripts/export_gcn_weights.py"
        )

    config = load_config(resolved_config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample = smiles_to_inference_graph("C")
    if sample is None:
        raise RuntimeError("Failed to build reference graph for input_dim inference.")

    model = FlexibleGCNRegressor(
        input_dim=sample.x.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    state = torch.load(resolved_weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def get_cached_predictor(
    config_path: Path = DEFAULT_CONFIG_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> tuple[nn.Module, torch.device]:
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None:
        _CACHED_PREDICTOR = load_gcn_predictor(config_path, weights_path)
    return _CACHED_PREDICTOR


def predict_pic50(
    smiles: str,
    model: nn.Module | None = None,
    device: torch.device | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> float:
    if model is None or device is None:
        model, device = get_cached_predictor(config_path, weights_path)

    graph = smiles_to_inference_graph(smiles)
    if graph is None:
        raise ValueError("RDKit could not parse SMILES")

    x, adj, mask, _ = collate_graphs([graph])
    with torch.no_grad():
        pred = model(x.to(device), adj.to(device), mask.to(device))
    return float(pred.cpu().item())


def predict_pic50_batch(
    smiles_list: list[str],
    config_path: Path = DEFAULT_CONFIG_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    *,
    source: str = "unknown",
    caller: str | None = None,
) -> dict[str, Any]:
    log_tool_event(
        "prediction_batch_start",
        source=source,
        caller=caller,
        smiles_count=len(smiles_list),
        smiles=smiles_list,
    )

    model, device = get_cached_predictor(config_path, weights_path)

    predictions: list[dict[str, Any]] = []
    for smiles in smiles_list:
        try:
            pic50 = predict_pic50(
                smiles,
                model=model,
                device=device,
                config_path=config_path,
                weights_path=weights_path,
            )
            row = {"smiles": smiles, "pic50": round(pic50, 4), "status": "ok"}
            predictions.append(row)
            log_tool_event(
                "prediction_ok",
                source=source,
                caller=caller,
                smiles=smiles,
                pic50=row["pic50"],
            )
        except ValueError as error:
            row = {
                "smiles": smiles,
                "pic50": None,
                "status": "error",
                "message": str(error),
            }
            predictions.append(row)
            log_tool_event(
                "prediction_error",
                source=source,
                caller=caller,
                smiles=smiles,
                message=str(error),
            )

    result = {
        "target": TARGET_ID,
        "target_name": TARGET_NAME,
        "model": MODEL_NAME,
        "predictions": predictions,
        "disclaimer": DISCLAIMER,
    }
    ok_count = sum(1 for item in predictions if item["status"] == "ok")
    log_tool_event(
        "prediction_batch_done",
        source=source,
        caller=caller,
        ok_count=ok_count,
        error_count=len(predictions) - ok_count,
    )
    return result
