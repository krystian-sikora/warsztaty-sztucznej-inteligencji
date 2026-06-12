"""CLI tool for pIC50 prediction from canonical SMILES (for LLM / agent use)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pic50_inference import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WEIGHTS_PATH,
    WeightsNotFoundError,
    predict_pic50_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", type=str, help="Single canonical SMILES string")
    group.add_argument("--input", type=Path, help="Text file with one SMILES per line")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (default is compact JSON on stdout)",
    )
    return parser.parse_args()


def read_smiles_list(args: argparse.Namespace) -> list[str]:
    if args.smiles is not None:
        return [args.smiles.strip()]
    lines = args.input.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> int:
    args = parse_args()
    smiles_list = read_smiles_list(args)

    try:
        result = predict_pic50_batch(
            smiles_list,
            config_path=args.config,
            weights_path=args.weights,
        )
    except WeightsNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))

    has_error = any(item["status"] == "error" for item in result["predictions"])
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
