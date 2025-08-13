# ISO/IEC 42001 – Demo Compliance Summary (Simulated)

Scope: Retail Forecast AIMS demo. All artifacts in this folder are simulated for portfolio demonstration.

Retention & Access (Demo):
- Evidence artifacts retained for 90 days.
- Access restricted to project maintainers.
- Daily SHA256 hashes recorded in `logs/daily_hashes.txt` for tamper-evidence intent.
- Optional encryption demo available via `scripts/encrypt_evidence_demo.py` (Fernet key stored locally, not committed).

Controls Mapping (High-level):
- Clause 6 (Planning / Risks): bias drift monitoring, baseline comparisons, risk-to-action logging (simulated).
- Clause 8 (Operation): input safety utilities, change logs for datasets/models (simulated).
- Clause 9 (Performance): metrics (MAPE/RMSE) surfaced in dashboard; audits and management reviews (simulated).
- Clause 10 (Improvement): CAPA log with effectiveness checks (simulated).

All items here are marked "Simulated" and intended only for demo purposes.
