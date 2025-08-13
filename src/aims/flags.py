"""
AIMS feature flags helpers.

All new governance/demo functionality must be opt-in and disabled by default.
This module centralizes flag parsing from environment variables.
"""

from __future__ import annotations

import os


def _env_truthy(var_name: str, default: str = "0") -> bool:
    """Return True if env var is set to a truthy value ("1", "true", "yes")."""
    val = os.getenv(var_name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def aims_demo_enabled() -> bool:
    return _env_truthy("AIMS_DEMO", "0")


def aims_demo_logs_enabled() -> bool:
    # If AIMS_DEMO is enabled, logs can be considered enabled unless explicitly off
    if aims_demo_enabled():
        return _env_truthy("AIMS_DEMO_LOGS", "1")
    return _env_truthy("AIMS_DEMO_LOGS", "0")


def aims_input_safety_enabled() -> bool:
    return _env_truthy("AIMS_INPUT_SAFETY", "0")


def aims_qc_enabled() -> bool:
    return _env_truthy("AIMS_QC", "0")


def aims_dashboard_enabled() -> bool:
    if aims_demo_enabled():
        return _env_truthy("AIMS_DASHBOARD", "1")
    return _env_truthy("AIMS_DASHBOARD", "0")


def aims_encrypt_demo_enabled() -> bool:
    return _env_truthy("AIMS_ENCRYPT_DEMO", "0")


__all__ = [
    "aims_demo_enabled",
    "aims_demo_logs_enabled",
    "aims_input_safety_enabled",
    "aims_qc_enabled",
    "aims_dashboard_enabled",
    "aims_encrypt_demo_enabled",
]


