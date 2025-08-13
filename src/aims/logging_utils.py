"""
Structured logging utilities for AIMS demo (ISO/IEC 42001 evidence intent).

Features (opt-in via env flags):
- Rotating JSON logs at logs/forecast_audit_trail.json
- Daily SHA256 hash append at logs/daily_hashes.txt

Guard enablement via flags in src.aims.flags.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .flags import aims_demo_logs_enabled


LOGS_DIR = Path("logs")
AUDIT_LOG_PATH = LOGS_DIR / "forecast_audit_trail.json"
DAILY_HASHES_PATH = LOGS_DIR / "daily_hashes.txt"


class JsonLineFormatter(logging.Formatter):
    """Simple JSON lines formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extras if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        # Include standard attributes if provided via 'extra'
        for key in ("event", "category", "run_id", "user", "component"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def _ensure_log_paths() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.touch()
    if not DAILY_HASHES_PATH.exists():
        DAILY_HASHES_PATH.touch()


def enable_structured_logging(max_bytes: int = 1_000_000, backup_count: int = 5) -> None:
    """Enable rotating JSON logging if demo logs flag is on. No-op otherwise."""
    if not aims_demo_logs_enabled():
        return

    _ensure_log_paths()

    handler = RotatingFileHandler(
        AUDIT_LOG_PATH, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(JsonLineFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    logging.getLogger(__name__).info(
        "Structured logging enabled (AIMS demo)", extra={"component": "aims.logging"}
    )


def append_daily_hash() -> str:
    """Append a daily SHA256 hash line for the audit log and return the hex digest.

    Format: YYYY-MM-DD <space> <sha256_hex>
    """
    _ensure_log_paths()
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Compute hash of current audit log file contents (intent: tamper-evidence demo)
    h = hashlib.sha256()
    try:
        with open(AUDIT_LOG_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        # If missing, hash empty content
        pass

    digest = h.hexdigest()
    line = f"{today} {digest}\n"

    # Idempotent append (avoid duplicate for same day)
    try:
        existing = DAILY_HASHES_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    if today not in existing:
        with open(DAILY_HASHES_PATH, "a", encoding="utf-8") as f:
            f.write(line)

    logging.getLogger(__name__).info(
        "Daily log hash recorded",
        extra={"component": "aims.logging", "event": "daily_hash", "digest": digest},
    )
    return digest


__all__ = [
    "enable_structured_logging",
    "append_daily_hash",
    "AUDIT_LOG_PATH",
    "DAILY_HASHES_PATH",
]


