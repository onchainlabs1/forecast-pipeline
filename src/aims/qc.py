"""
Forecast safeguards (read-only) for AIMS demo.

Features (AIMS_QC=1):
- Bias drift check by category/store (signed error aggregates)
- Baseline comparison (naive / moving-average) vs model forecasts

These helpers return dict reports and do not block flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .flags import aims_qc_enabled


@dataclass
class BiasDriftConfig:
    group_by: Sequence[str] = ("store_nbr", "family")
    threshold: float = 0.1  # relative signed error threshold


def bias_drift_report(df: pd.DataFrame, actual_col: str, pred_col: str, cfg: Optional[BiasDriftConfig] = None) -> Dict:
    if not aims_qc_enabled():
        return {"enabled": False}
    cfg = cfg or BiasDriftConfig()
    req_cols = set(cfg.group_by) | {actual_col, pred_col}
    if not req_cols.issubset(df.columns):
        return {"enabled": True, "error": "missing_columns", "required": sorted(req_cols)}

    g = df.groupby(list(cfg.group_by), dropna=False)
    agg = g.apply(lambda x: pd.Series({
        "n": len(x),
        "signed_err": float((x[pred_col] - x[actual_col]).mean()),
        "rel_signed_err": float(((x[pred_col] - x[actual_col]) / (np.abs(x[actual_col]) + 1e-8)).mean()),
    })).reset_index()
    agg["flag"] = (agg["rel_signed_err"].abs() >= cfg.threshold).astype(int)
    return {"enabled": True, "config": cfg.__dict__, "rows": agg.to_dict(orient="records")}


def naive_baseline(y: np.ndarray, window: int = 7) -> np.ndarray:
    if len(y) == 0:
        return y
    # simple moving average with window, fallback to cumulative mean for warmup
    out = np.zeros_like(y, dtype=float)
    cumsum = np.cumsum(y, dtype=float)
    for i in range(len(y)):
        if i + 1 < window:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - window]) / window
    return out


def baseline_comparison(y_true: np.ndarray, y_pred: np.ndarray, window: int = 7) -> Dict:
    if not aims_qc_enabled():
        return {"enabled": False}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_base = naive_baseline(y_true, window=window)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def mape(a, b):
        denom = np.clip(np.abs(a), 1e-8, None)
        return float(np.mean(np.abs((a - b) / denom)))

    report = {
        "enabled": True,
        "window": window,
        "scores": {
            "model": {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "mape": mape(y_true, y_pred)},
            "baseline": {"rmse": rmse(y_true, y_base), "mae": mae(y_true, y_base), "mape": mape(y_true, y_base)},
        },
        "outlier_flags": {
            "rmse_outlier": rmse(y_true, y_pred) > 1.5 * rmse(y_true, y_base),
            "mae_outlier": mae(y_true, y_pred) > 1.5 * mae(y_true, y_base),
        },
    }
    return report


__all__ = ["BiasDriftConfig", "bias_drift_report", "baseline_comparison", "naive_baseline"]


