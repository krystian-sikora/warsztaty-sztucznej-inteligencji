"""Structured logging for pIC50 prediction tool invocations."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "pic50_tool.jsonl"

_LOGGER_NAME = "pic50_tool"
_configured = False


def _get_logger() -> logging.Logger:
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if not _configured:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s pic50_tool: %(message)s")
            )
            logger.addHandler(handler)
        _configured = True
    return logger


def log_tool_event(
    event: str,
    *,
    log_file: Path = DEFAULT_LOG_FILE,
    **fields: Any,
) -> None:
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = ", ".join(f"{key}={value}" for key, value in fields.items())
    _get_logger().info("%s | %s", event, summary)
