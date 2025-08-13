"""
Input safety utilities (opt-in) for UI/API parameters.

Features (behind AIMS_INPUT_SAFETY=1):
- Basic prompt-like token blocking
- Max length enforcement for strings
- Numeric bounds checks helper

This module provides pure functions to be optionally called by API/UI layers
without changing their interfaces.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .flags import aims_input_safety_enabled


PROMPTLIKE_PATTERN = re.compile(r"(?i)\b(system:|user:|assistant:|<\/?(system|user|assistant)>|\\n\\n)\b")


def sanitize_text(value: str, max_len: int = 256) -> str:
    """Return a sanitized string if safety enabled; otherwise passthrough."""
    if not aims_input_safety_enabled():
        return value
    if value is None:
        return ""
    v = str(value)
    v = PROMPTLIKE_PATTERN.sub(" ", v)
    if len(v) > max_len:
        v = v[:max_len]
    return v


def clamp_number(num: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    """Clamp a numeric value within [min_value, max_value] if safety enabled."""
    if not aims_input_safety_enabled():
        return num
    x = float(num)
    if min_value is not None:
        x = max(min_value, x)
    if max_value is not None:
        x = min(max_value, x)
    return x


def validate_params(params: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sanitize/validate a dict of params based on optional limits spec.

    limits example:
        {
            "store_nbr": {"min": 1, "max": 1000},
            "family": {"type": "text", "max_len": 64},
        }
    """
    if not aims_input_safety_enabled():
        return params
    if not params:
        return {}
    res: Dict[str, Any] = dict(params)
    limits = limits or {}
    for key, spec in limits.items():
        if key not in res:
            continue
        val = res[key]
        typ = spec.get("type")
        if typ == "text" or isinstance(val, str):
            res[key] = sanitize_text(str(val), max_len=int(spec.get("max_len", 256)))
        elif typ == "number" or isinstance(val, (int, float)):
            res[key] = clamp_number(float(val), spec.get("min"), spec.get("max"))
    return res


__all__ = [
    "sanitize_text",
    "clamp_number",
    "validate_params",
]


