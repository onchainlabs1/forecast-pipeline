#!/usr/bin/env python
"""
Simulated governance cycle writer (AIMS demo).

Appends one dated cycle of evidence files into docs/evidence/ and generates two
markdown reports (Internal Audit and Management Review). All artifacts labeled "Simulated".

Usage:
  AIMS_DEMO=1 python scripts/simulate_cycle.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import uuid
import os

from src.aims.flags import aims_demo_enabled


BASE = Path("docs/evidence")


def _now_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _ensure_dirs() -> None:
    (BASE).mkdir(parents=True, exist_ok=True)


def _append_csv(path: Path, headers: list[str], row: list[str]) -> None:
    write_headers = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_headers:
            w.writerow(headers)
        w.writerow(row)


def run() -> None:
    if not aims_demo_enabled():
        print("AIMS_DEMO flag is OFF; nothing to simulate.")
        return

    _ensure_dirs()
    ts = _now_str()
    cycle_id = str(uuid.uuid4())[:8]

    # CSV logs
    _append_csv(
        BASE / "experiments_log.csv",
        ["timestamp", "run_id", "metric_mape", "metric_rmse", "label"],
        [ts, cycle_id, "0.20", "300.5", "Simulated"],
    )
    _append_csv(
        BASE / "data_change_log.csv",
        ["timestamp", "dataset", "change", "owner", "label"],
        [ts, "train.csv", "schema: +feature holiday_flag", "dataops", "Simulated"],
    )
    _append_csv(
        BASE / "data_incident_log.csv",
        ["timestamp", "incident", "severity", "status", "label"],
        [ts, "late data arrival", "low", "resolved", "Simulated"],
    )
    _append_csv(
        BASE / "capa_log.csv",
        ["timestamp", "issue", "action", "owner", "due_date", "effectiveness_check", "label"],
        [ts, "bias drift store 5", "retrain + add feature", "ml", "30d", "pending", "Simulated"],
    )
    _append_csv(
        BASE / "internal_audit_log.csv",
        ["timestamp", "scope", "independence_attested", "findings", "label"],
        [ts, "forecast process", "Yes", "no major findings", "Simulated"],
    )

    # Markdown reports
    audit_md = BASE / f"Internal_Audit_Report_{cycle_id}.md"
    audit_md.write_text(
        f"""# Internal Audit Report (Simulated)\n\n- ID: {cycle_id}\n- Timestamp: {ts}\n- Scope: Forecasting process and controls\n- Independence Attested: Yes\n\nFindings:\n- No major findings. Minor observation on input safety thresholds.\n\nThis document is generated for demo purposes (Simulated).\n""",
        encoding="utf-8",
    )

    mgmt_md = BASE / f"Management_Review_Minutes_{cycle_id}.md"
    mgmt_md.write_text(
        f"""# Management Review Minutes (Simulated)\n\n- ID: {cycle_id}\n- Timestamp: {ts}\n- Attendees: Head of Data, MLOps Lead, QA\n- KPIs Reviewed: MAPE, RMSE, data incidents, CAPA status\n\nDecisions:\n- Continue monitoring drift; plan retrain in 2 weeks.\n\nThis document is generated for demo purposes (Simulated).\n""",
        encoding="utf-8",
    )

    print(f"Simulated cycle recorded. ID={cycle_id} in {BASE}")


if __name__ == "__main__":
    run()


