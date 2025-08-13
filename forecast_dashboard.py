#!/usr/bin/env python
"""
Minimal governance dashboard (AIMS demo) – Streamlit.

Tabs:
- Overview
- Models & Data
- Forecast Quality
- Audit Preparation

Buttons:
- [Run Simulated Cycle]
- [Record Daily Log Hash]

Activation flags (environment):
- AIMS_DEMO=1 and AIMS_DASHBOARD=1 (defaulted on when AIMS_DEMO is set)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import streamlit as st
except Exception as _e:  # pragma: no cover – optional import for demo only
    raise SystemExit("Streamlit is required to run forecast_dashboard.py")

from src.aims.flags import (
    aims_demo_enabled,
    aims_dashboard_enabled,
    aims_demo_logs_enabled,
)
from src.aims.logging_utils import enable_structured_logging, append_daily_hash


def _load_metrics() -> dict:
    # Try common locations
    for p in (
        Path("reports/metrics.json"),
        Path("metrics.json"),
    ):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _kpi(value, label, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def main() -> None:
    if aims_demo_logs_enabled():
        enable_structured_logging()

    st.set_page_config(page_title="AIMS Demo Dashboard", layout="wide")

    st.title("AIMS Governance Demo Dashboard")
    st.caption("Functional demo with simulated portfolio evidence.")

    if not aims_demo_enabled() or not aims_dashboard_enabled():
        st.warning(
            "AIMS demo flags are OFF. Set AIMS_DEMO=1 (and optionally AIMS_DASHBOARD=1) to enable full functionality.")

    # Top KPIs
    metrics = _load_metrics()
    col1, col2, col3 = st.columns(3)
    with col1:
        _kpi(metrics.get("mape", "—"), "MAPE")
    with col2:
        _kpi(metrics.get("mae", "—"), "MAE")
    with col3:
        _kpi(metrics.get("rmse", "—"), "RMSE")

    # Action buttons
    st.divider()
    colA, colB = st.columns(2)
    with colA:
        if st.button("Run Simulated Cycle", type="primary"):
            try:
                from scripts.simulate_cycle import run as sim_run

                sim_run()
                st.success("Simulated cycle appended to docs/evidence/")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
    with colB:
        if st.button("Record Daily Log Hash"):
            try:
                digest = append_daily_hash()
                st.success(f"Daily hash recorded: {digest[:12]}…")
            except Exception as e:
                st.error(f"Hash record failed: {e}")

    st.divider()

    tabs = st.tabs(["Overview", "Models & Data", "Forecast Quality", "Audit Preparation"])

    with tabs[0]:
        st.subheader("Overview")
        st.write(
            "This demo showcases ISO/IEC 42001-aligned governance primitives: structured logs, simulated evidence, and quality checks."
        )

    with tabs[1]:
        st.subheader("Models & Data")
        st.write("List of datasets/models is illustrative for demo. Evidence is stored under docs/evidence/ (Simulated).")

    with tabs[2]:
        st.subheader("Forecast Quality")
        st.write("Quality checks (bias drift, baseline comparison) are available via AIMS_QC helpers for read-only reporting.")

    with tabs[3]:
        st.subheader("Audit Preparation")
        st.write("Use the buttons above to simulate daily cycles and record log hashes for tamper-evidence intent.")


if __name__ == "__main__":
    main()


